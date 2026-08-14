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

"""Tests for _zero_gap_rows correctness in tensor_product backward.

The irrep combination used here has real CG gap rows:

    '4x0e + 6x1o + 3x1e + 7x2e + 4x2o + 6x3o + 4x3e'
      x  '1x0e + 1x1o + 1x2e + 1x3o'  ->  23x0e

x's 1e (9 coords), 2o (20 coords), and 3e (28 coords) irreps cannot couple
to any y irrep to produce 0e output, leaving 57 gap rows in ct_x that
_zero_gap_rows must zero explicitly.  If missing or broken, those rows
contain uninitialized memory (NaN) and the gradient diverges from reference.
"""

import jax
import jax.numpy as np
import jax.random as random
import numpy.testing as testing

import e3j
from e3j.core.tensor_product import TensorProduct as CoreTP
from e3j.ops import CUDATensorProductParams as Params
from e3j.ops import tensor_product
from e3j.ops.coef import Coef

e3j.config(debug_level=0)

IN1 = "4x0e + 6x1o + 3x1e + 7x2e + 4x2o + 6x3o + 4x3e"
IN2 = "1x0e + 1x1o + 1x2e + 1x3o"


def pack_coef(val, idx, val_dtype="float32", idx_dtype="int32"):
    return Coef(val, idx.T, val_dtype=val_dtype, idx_dtype=idx_dtype).pack_jax()


def assert_allclose(expect, result, rtol=5e-6, atol=5e-6, debug: int = 1):
    try:
        testing.assert_allclose(expect, result, rtol=rtol, atol=atol)
    except AssertionError as err:
        if debug >= 1:
            print("expect == result\n", abs(expect - result) < atol)
        raise err


def tensor_product_reference(
    idx, coef, x, y, num_out, mode="OUTER", layout="LEADING_CHANNELS"
):
    """Reference tensor_product (pure scatter-add; naturally zeros gap rows)."""
    if layout == "TRAILING_CHANNELS":
        x_t = x if x.ndim < 3 else np.matrix_transpose(x)
        y_t = y if y.ndim < 3 else np.matrix_transpose(y)
        z_t = tensor_product_reference(idx, coef, x_t, y_t, num_out, mode)
        return z_t if z_t.ndim < 3 else np.matrix_transpose(z_t)

    n_rows = x.shape[0]
    if mode == "MAP":
        out = np.zeros((n_rows, x.shape[1], num_out), dtype=x.dtype)
        return out.at[..., idx[0]].add(coef * x[..., idx[1]] * y[..., idx[2]])
    if mode == "OUTER":
        has_cx, has_cy = x.ndim > 2, y.ndim > 2
        if not (has_cx or has_cy):
            out = np.zeros((n_rows, num_out), dtype=x.dtype)
        elif has_cx:
            out = np.zeros((n_rows, x.shape[1], num_out), dtype=x.dtype)
            y = y[:, None]
        else:
            out = np.zeros((n_rows, y.shape[1], num_out), dtype=x.dtype)
            x = x[:, None]
        return out.at[..., idx[0]].add(coef * x[..., idx[1]] * y[..., idx[2]])


class _TestGapRows:
    """Backward correctness for a tensor product with real CG gap rows.

    Subclasses set ``mode``, ``channels_x``, and ``channels_y``.
    ``closure()`` returns the same ``(idx, val, kwargs)`` triple as
    ``_TestTensorProductOp.closure()``, but built from actual CG coefficients
    so that gap rows exist and _zero_gap_rows is exercised.
    """

    mode: str
    layout: str = "TRAILING_CHANNELS"
    channels_x: int | None = None
    channels_y: int | None = None
    num_rows: int = 16
    num_x: int = 156
    num_y: int = 16
    num_out: int = 23

    def closure(self):
        with e3j.config.use(tensor_product="FUSED_CUDA"):
            tp = CoreTP((IN1, IN2), "0e", sort=True, layout=self.layout, mode=self.mode)
        coef = tp.coef
        idx = np.array(coef.indices).T  # (3, nnz)
        val = coef.data
        kwargs = {"num_out": tp.target.dim, "mode": self.mode, "layout": self.layout}
        return idx, val, kwargs

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

    def inputs(self):
        n = self.num_rows
        nx, ny = self.num_x, self.num_y
        cx, cy = self.channels_x, self.channels_y
        if self.layout == "LEADING_CHANNELS":
            shape_x = (n, nx) if cx is None else (n, cx, nx)
            shape_y = (n, ny) if cy is None else (n, cy, ny)
        else:
            shape_x = (n, nx) if cx is None else (n, nx, cx)
            shape_y = (n, ny) if cy is None else (n, ny, cy)
        keys = (k for k in random.split(random.key(123), 2))
        x = random.normal(next(keys), shape_x)
        y = random.normal(next(keys), shape_y)
        return x, y

    def test_backward(self):
        x, y = self.inputs()
        expect_dx, expect_dy = self.bwd_ref(x, y)
        result_dx, result_dy = self.bwd_op(x, y)
        assert_allclose(expect_dx, result_dx, atol=1e-4, rtol=2e-3)
        assert_allclose(expect_dy, result_dy, atol=1e-4, rtol=2e-3)


class TestGapRowsOUTER(_TestGapRows):
    layout = "TRAILING_CHANNELS"
    mode = "OUTER"
    channels_x = 32
    channels_y = None


class TestGapRowsMAP(_TestGapRows):
    layout = "TRAILING_CHANNELS"
    mode = "MAP"
    channels_x = 32
    channels_y = 32


class TestGapRowsOUTERLeading(_TestGapRows):
    layout = "LEADING_CHANNELS"
    mode = "OUTER"
    channels_x = 32
    channels_y = None
