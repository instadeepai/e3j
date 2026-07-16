# Copyright (c) 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as np
import jax.random as random
import numpy.testing as testing
import pytest
from jax import Array

import e3j
from e3j.ops import CUDATensorProductParams as Params
from e3j.ops import tensor_product
from e3j.ops.coef import Coef
from e3j.utils.sparse import narrow_index_dtype

e3j.config(
    debug_level=0,
)


@pytest.fixture(scope="class", params=[32, 64, 128, 512])
def channels(request):
    request.cls.channels_x = request.param
    if getattr(request.cls, "mode", None) in ("INNER", "MAP"):
        request.cls.channels_y = request.param
    elif getattr(request.cls, "mode", None) == "OUTER":
        request.cls.channels_y = None


@pytest.fixture(scope="class", params=["OUTER", "INNER", "MAP"])
def mode(request):
    request.cls.mode = request.param


np.set_printoptions(
    precision=4,
    suppress=True,
    formatter={"bool": lambda t: "1" if t else "0"},
    linewidth=90,
)


def pack_coef(val, idx, val_dtype="float32", idx_dtype="int32"):
    """Pack (nnz,) values + (3, nnz) COO indices into an opaque JAX array."""
    return Coef(val, idx.T, val_dtype=val_dtype, idx_dtype=idx_dtype).pack_jax()


def assert_allclose(expect, result, rtol=5e-6, atol=5e-6, debug: int = 1):
    """Show more on assertion errors to help diagnose addressing errors."""
    try:
        testing.assert_allclose(expect, result, rtol=rtol, atol=atol)
    except AssertionError as err:
        if debug >= 1:
            print("expect == result\n", abs(expect - result) < atol)
        if debug >= 2:
            print("expect\n", expect)
            print("result\n", result)
        raise err


def tensor_product_reference(
    idx, coef, x, y, num_out, mode="OUTER", layout="LEADING_CHANNELS"
):
    """Reference tensor_product implementation."""

    if layout == "TRAILING_CHANNELS":
        x_t = x if x.ndim < 3 else np.matrix_transpose(x)
        y_t = y if y.ndim < 3 else np.matrix_transpose(y)
        z_t = tensor_product_reference(idx, coef, x_t, y_t, num_out, mode)
        if z_t.ndim < 3:
            return z_t
        return np.matrix_transpose(z_t)

    n_rows = x.shape[0]

    if mode == "MAP":
        assert (
            x.ndim == 3 and y.ndim == 3
        ), "MAP mode requires channel dims on both x and y"
        assert x.shape[1] == y.shape[1], "MAP mode requires channels_x == channels_y"
        shape_out = (n_rows, x.shape[1], num_out)
        out = np.zeros(shape_out, dtype=x.dtype)
        cxy = coef * x[..., idx[1]] * y[..., idx[2]]
        return out.at[..., idx[0]].add(cxy)

    if mode == "OUTER":
        has_cx, has_cy = x.ndim > 2, y.ndim > 2

        if not (has_cx or has_cy):
            shape_out = (n_rows, num_out)

        elif has_cx:
            shape_out = (n_rows, x.shape[1], num_out)
            y = y[:, None]

        elif has_cy:
            shape_out = (n_rows, y.shape[1], num_out)
            x = x[:, None]

        out = np.zeros(shape_out, dtype=x.dtype)
        cxy = coef * x[..., idx[1]] * y[..., idx[2]]
        return out.at[..., idx[0]].add(cxy)

    if mode == "INNER":
        assert (
            x.ndim == 3 and y.ndim == 3
        ), "INNER mode requires channel dims on both x and y"
        assert x.shape[1] == y.shape[1], "INNER mode requires channels_x == channels_y"
        shape_out = (n_rows, num_out)
        out = np.zeros(shape_out, dtype=x.dtype)
        cxy = coef * x[..., idx[1]] * y[..., idx[2]]
        sum_cxy = np.sum(cxy, axis=-2)
        return out.at[..., idx[0]].add(sum_cxy)


class _TestTensorProductOp:
    mode: str
    num_idx: int = 530
    num_x: int = 42
    num_y: int = 25
    num_out: int
    channels_x: int | None = None
    channels_y: int | None = None
    num_rows = 100
    layout: str = "TRAILING_CHANNELS"

    def closure(self):
        """Return coefficient buffers and tensor product descriptor.

        Closing over these parameters yields a tensor-product
        with signature (x, y) -> z.
        """
        nx, ny, nz = self.num_x, self.num_y, self.num_out
        nnz = self.num_idx

        keys = (k for k in random.split(random.key(1234), 4))

        # NOTE: makes sure I/O indices are covered, since we
        #        don't flush rows by default with (lm,k) layout.
        def make_idx(dim: int, n: int, key=None) -> Array:
            if key is None:
                key = random.key(n + dim)
            idx_all = np.arange(dim)
            # Last index has only one occurence, to detect a previous
            # bug where last store would have been skipped.
            idx_rdm = random.randint(key, (n - dim,), 0, dim - 1)
            return np.concat((idx_all, idx_rdm))

        indices = [
            make_idx(nz, nnz, next(keys)),
            make_idx(nx, nnz, next(keys)),
            make_idx(ny, nnz, next(keys)),
        ]

        sigma = np.argsort(indices[0])
        idx = np.stack(indices)[:, sigma]
        idx = idx.astype(narrow_index_dtype((nz, nx, ny)))
        val = random.normal(next(keys), (nnz,))

        kwargs = {
            "num_out": self.num_out,
            "mode": self.mode,
            "layout": self.layout,
        }

        return (idx, val, kwargs)

    @property
    def fwd_ref(self):
        idx, val, kwargs = self.closure()
        return lambda x, y: tensor_product_reference(idx, val, x, y, **kwargs)

    @property
    def fwd_op(self):
        idx, val, kwargs = self.closure()
        coef = pack_coef(val, idx)
        params = Params(**kwargs)
        return lambda x, y: tensor_product(coef, x, y, params)

    @property
    def bwd_ref(self):
        return jax.grad(
            lambda x, y: np.sum(self.fwd_ref(x, y)),
            argnums=(0, 1),
        )

    @property
    def bwd_op(self):
        return jax.grad(
            lambda x, y: np.sum(self.fwd_op(x, y)),
            argnums=(0, 1),
        )

    @property
    def bwd_bwd_op(self):
        def sum_bwd(x, y, dz):
            """A backward operation consuming dz cotangents."""
            df = jax.grad(
                lambda x, y, dz: np.sum(self.fwd_op(x, y) * dz),
                argnums=(0, 1),
            )
            dx, dy = df(x, y, dz)
            return np.sum(dx) + np.sum(dy)

        # Differentiate once more, yield 3 cotangents
        return jax.grad(sum_bwd, argnums=(0, 1, 2))

    @property
    def bwd_bwd_ref(self):
        def sum_bwd(x, y, dz):
            """A backward operation consuming dz cotangents."""
            df = jax.grad(
                lambda x, y, dz: np.sum(self.fwd_ref(x, y) * dz),
                argnums=(0, 1),
            )
            dx, dy = df(x, y, dz)
            return np.sum(dx) + np.sum(dy)

        # Differentiate once more, yield 3 cotangents
        return jax.grad(sum_bwd, argnums=(0, 1, 2))

    def inputs(self) -> tuple[Array, Array]:
        n = self.num_rows
        nx, ny = self.num_x, self.num_y
        cx, cy = self.channels_x, self.channels_y
        if self.layout == "LEADING_CHANNELS":
            shape_x = (n, nx) if cx is None else (n, cx, nx)
            shape_y = (n, ny) if cy is None else (n, cy, ny)
        elif self.layout == "TRAILING_CHANNELS":
            shape_x = (n, nx) if cx is None else (n, nx, cx)
            shape_y = (n, ny) if cy is None else (n, ny, cy)

        keys = (k for k in random.split(random.key(123), 2))
        x = random.normal(next(keys), shape_x)
        y = random.normal(next(keys), shape_y)
        return (x, y)

    def test_index_dtype(self):
        """Check that closure indices have the expected narrow dtype."""
        idx, coef, kwargs = self.closure()
        shape = (kwargs["num_out"], self.num_x, self.num_y)
        expected_dtype = narrow_index_dtype(shape)
        assert idx.dtype == expected_dtype

    def test_forward(self):
        x, y = self.inputs()
        expect = self.fwd_ref(x, y)
        result = self.fwd_op(x, y)
        assert_allclose(expect, result, atol=5e-5, rtol=5e-5)

    def test_backward(self):
        x, y = self.inputs()
        expect_dx, expect_dy = self.bwd_ref(x, y)
        result_dx, result_dy = self.bwd_op(x, y)
        assert_allclose(expect_dx, result_dx, atol=1e-4, rtol=2e-3)
        assert_allclose(expect_dy, result_dy, atol=1e-4, rtol=2e-3)

    def test_backward_backward(self):
        x, y = self.inputs()
        dz = self.fwd_ref(x, y)
        expect_Dx, expect_Dy, expect_Ddz = self.bwd_bwd_ref(x, y, dz)
        result_Dx, result_Dy, result_Ddz = self.bwd_bwd_op(x, y, dz)
        assert_allclose(expect_Dx, result_Dx, atol=1e-4, rtol=2e-3)
        assert_allclose(expect_Dy, result_Dy, atol=1e-4, rtol=2e-3)
        assert_allclose(expect_Ddz, result_Ddz, atol=1e-4, rtol=2e-3)


@pytest.mark.usefixtures("channels")
class TestTensorProductOuter(_TestTensorProductOp):
    layout = "TRAILING_CHANNELS"
    mode = "OUTER"
    num_x = 16
    num_y = 25
    num_out = 93
    num_rows = 100


@pytest.mark.usefixtures("channels")
class TestTensorProductInner(_TestTensorProductOp):
    layout = "TRAILING_CHANNELS"
    mode = "INNER"
    num_x = 16
    num_y = 25
    num_out = 93
    num_rows = 100


@pytest.mark.usefixtures("channels")
class TestTensorProductMap(_TestTensorProductOp):
    layout = "TRAILING_CHANNELS"
    mode = "MAP"
    num_x = 16
    num_y = 25
    num_out = 93
    num_rows = 100


@pytest.mark.usefixtures("channels")
class TestTensorProductOuterOneIn(_TestTensorProductOp):
    layout = "TRAILING_CHANNELS"
    mode = "OUTER"
    num_x = 16
    num_y = 1
    num_out = 16
    num_idx = 30
    num_rows = 64


class TestTensorProductInnerOneOut(_TestTensorProductOp):
    layout = "TRAILING_CHANNELS"
    mode = "INNER"
    num_x = 25
    num_y = 25
    channels_x = 512
    channels_y = 512
    num_out = 1
    num_idx = 42
    num_rows = 8


class TestTensorProductOuterCY(_TestTensorProductOp):
    layout = "TRAILING_CHANNELS"
    mode = "OUTER"
    channels_x = None
    channels_y = 128
    num_rows = 22
    num_idx = 512
    num_out = 210


class TestTensorProductLeadingOuter(_TestTensorProductOp):
    layout = "LEADING_CHANNELS"
    mode = "OUTER"
    num_idx = 780
    num_x = 16
    num_y = 25
    channels_x = 256
    channels_y = None
    num_out = 93
    num_rows = 100


class TestTensorProductLeadingInner(_TestTensorProductOp):
    layout = "LEADING_CHANNELS"
    mode = "INNER"
    num_idx = 780
    num_x = 16
    num_y = 25
    channels_x = 256
    channels_y = 256
    num_out = 93
    num_rows = 100


class TestTensorProductLeadingMap(_TestTensorProductOp):
    layout = "LEADING_CHANNELS"
    mode = "MAP"
    num_idx = 780
    num_x = 16
    num_y = 25
    channels_x = 256
    channels_y = 256
    num_out = 93
    num_rows = 100
