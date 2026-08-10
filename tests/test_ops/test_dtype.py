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

"""float16 / float32 / float64 value dtypes for the fused CUDA kernels.

GPU only. `float64` needs `jax_enable_x64`, scoped locally by `x64_if`.
"""

import contextlib

import jax
import jax.numpy as jnp
import jax.random as random
import numpy
import numpy.testing as testing
import pytest

import e3j
from e3j.ops import CUDATensorProductParams as Params
from e3j.ops import scatter_add_1, tensor_product
from e3j.ops.coef import Coef, Coef4D, resolve_val_dtype
from e3j.ops.convolution import CUDAConvolutionParams, convolution
from e3j.utils.options import Layout, MixingMode

DTYPES = ["float16", "float32", "float64"]

# Small problem sizes, so every index fits in uint8, the narrow dtype
# `narrow_index_dtype` picks. int32 indices are covered separately.
NUM_OUT, NUM_X, NUM_Y, NNZ, NUM_ROWS = 40, 16, 25, 120, 32

# The backward pass reduces channels over a full warp, so the channel count
# must be a multiple of 32.
CHANNELS = 32


def x64_if(dtype):
    """float64 is silently truncated to float32 unless x64 is enabled."""
    return jax.enable_x64() if dtype == "float64" else contextlib.nullcontext()


def tolerances(dtype, reference, nnz=NNZ):
    """Error budget in dtype ULP: both sides accumulate in the same dtype, so
    they only differ by summation order, which grows like `sqrt(nnz)`.
    """
    eps = float(numpy.finfo(numpy.dtype(dtype)).eps)
    scale = max(float(jnp.max(jnp.abs(reference.astype("float32")))), 1.0)
    return {"atol": 8 * eps * scale, "rtol": 8 * eps * nnz**0.5}


def assert_close(expect, result, dtype, nnz=NNZ):
    tol = tolerances(dtype, expect, nnz)
    testing.assert_allclose(
        numpy.asarray(result, dtype="float64"),
        numpy.asarray(expect, dtype="float64"),
        **tol,
    )


def make_idx(dim, n, key):
    """Random COO coordinates in [0, dim), each appearing at least once.

    The kernel does not flush output rows no coefficient writes to.
    """
    return jnp.concatenate(
        [jnp.arange(dim), random.randint(key, (n - dim,), 0, dim - 1)]
    )


# --- Tensor product -------------------------------------------------------


def tp_problem():
    """Return uint8 COO indices (3, nnz) covering every output coord, + values."""
    keys = iter(random.split(random.key(7), 4))
    idx = jnp.stack(
        [
            make_idx(NUM_OUT, NNZ, next(keys)),
            make_idx(NUM_X, NNZ, next(keys)),
            make_idx(NUM_Y, NNZ, next(keys)),
        ]
    )
    idx = idx[:, jnp.argsort(idx[0])].astype("uint8")
    return idx, random.normal(next(keys), (NNZ,))


def tp_reference(idx, val, x, y):
    """OUTER + TRAILING_CHANNELS reference, in the operand dtype.

    `x` is `(rows, num_x[, channels])`; OUTER leaves `y` single-channel.
    """
    has_cx = x.ndim > 2
    x_t = jnp.matrix_transpose(x) if has_cx else x
    y_t = y[:, None] if has_cx else y
    shape = (x.shape[0], x.shape[2], NUM_OUT) if has_cx else (x.shape[0], NUM_OUT)
    out = jnp.zeros(shape, dtype=x.dtype)
    z = out.at[..., idx[0]].add(val * x_t[..., idx[1]] * y_t[..., idx[2]])
    return jnp.matrix_transpose(z) if has_cx else z


def tp_inputs(dtype, channels):
    kx, ky = random.split(random.key(11))
    shape_x = (NUM_ROWS, NUM_X) if channels is None else (NUM_ROWS, NUM_X, channels)
    x = random.normal(kx, shape_x).astype(dtype)
    y = random.normal(ky, (NUM_ROWS, NUM_Y)).astype(dtype)
    return x, y


def pack_tp(val, idx, dtype, idx_dtype="uint8"):
    return Coef(val, idx.T, val_dtype=dtype, idx_dtype=idx_dtype).pack_jax()


def tp_params(layout):
    """Build params inside a test: without `e3j_ops`, `Params` is `object`, so
    constructing it at import time would break collection on CPU.
    """
    return Params(num_out=NUM_OUT, mode="OUTER", layout=layout)


@pytest.mark.parametrize("dtype", DTYPES)
class TestTensorProductDtype:
    """Fused tensor product parity over {float16, float32, float64}."""

    @pytest.mark.parametrize("channels", [None, CHANNELS])
    def test_forward(self, dtype, channels):
        with x64_if(dtype):
            idx, val = tp_problem()
            x, y = tp_inputs(dtype, channels)
            coef = pack_tp(val, idx, dtype)

            params = tp_params("TRAILING_CHANNELS")
            result = tensor_product(coef, x, y, params)
            expect = tp_reference(idx, val.astype(dtype), x, y)

            assert result.dtype == jnp.dtype(dtype)
            assert result.shape == expect.shape
            assert_close(expect, result, dtype)

    def test_backward(self, dtype):
        with x64_if(dtype):
            idx, val = tp_problem()
            x, y = tp_inputs(dtype, CHANNELS)
            coef = pack_tp(val, idx, dtype)
            val_dt = val.astype(dtype)

            # Summing in the operand dtype gives a cotangent of exact ones, so
            # op and reference see the same incoming gradient.
            params = tp_params("TRAILING_CHANNELS")
            dx, dy = jax.grad(
                lambda a, b: jnp.sum(tensor_product(coef, a, b, params)),
                argnums=(0, 1),
            )(x, y)
            rx, ry = jax.grad(
                lambda a, b: jnp.sum(tp_reference(idx, val_dt, a, b)),
                argnums=(0, 1),
            )(x, y)

            assert dx.dtype == jnp.dtype(dtype) and dy.dtype == jnp.dtype(dtype)
            assert_close(rx, dx, dtype)
            assert_close(ry, dy, dtype)


@pytest.mark.parametrize("idx_dtype", ["uint8", "int32"])
def test_float16_index_dtypes(idx_dtype):
    """float16 pairs with either index dtype the coef table may use."""
    idx, val = tp_problem()
    x, y = tp_inputs("float16", CHANNELS)
    coef = pack_tp(val, idx.astype(idx_dtype), "float16", idx_dtype=idx_dtype)
    params = Params(num_out=NUM_OUT, mode="OUTER", layout="TRAILING_CHANNELS")

    result = tensor_product(coef, x, y, params)
    expect = tp_reference(idx, val.astype("float16"), x, y)

    assert result.dtype == jnp.float16
    assert_close(expect, result, "float16")


# --- Convolution ----------------------------------------------------------

CONV_NNZ, CONV_X, CONV_Y, CONV_OUT, CONV_SCALARS = 200, 9, 16, 9, 3
CONV_NODES, CONV_EDGES = 8, 24


def conv_problem():
    """COO indices + scalar-mixing index for the fused convolution."""
    keys = iter(random.split(random.key(42), 5))
    idx = jnp.stack(
        [
            make_idx(CONV_OUT, CONV_NNZ, next(keys)),
            make_idx(CONV_X, CONV_NNZ, next(keys)),
            make_idx(CONV_Y, CONV_NNZ, next(keys)),
        ]
    )
    idx = idx[:, jnp.argsort(idx[0])]
    val = random.normal(next(keys), (CONV_NNZ,))
    s_index = jnp.sort(
        jnp.concatenate(
            [
                jnp.arange(CONV_SCALARS),
                random.randint(next(keys), (CONV_OUT - CONV_SCALARS,), 0, CONV_SCALARS),
            ]
        )
    )
    return idx, val, s_index


def conv_graph():
    # int32 pinned: the FFI declares the adjacency as int32, and x64 would
    # default these to int64. `core.Convolution` casts, the raw op does not.
    k1, k2 = random.split(random.key(99))
    sender = random.randint(k1, (CONV_EDGES,), 0, CONV_NODES, dtype="int32")
    receiver = random.randint(k2, (CONV_EDGES,), 0, CONV_NODES, dtype="int32")
    perm = jnp.argsort(receiver)
    return sender[perm], receiver[perm]


def conv_reference(idx, val, x, y, s, s_index, sender, receiver):
    """Unfused gather -> TP -> scalar mix -> scatter, in the operand dtype."""
    channels = x.shape[-1]
    x_e = x[sender]
    z_e = jnp.zeros((CONV_EDGES, CONV_OUT, channels), dtype=x.dtype)
    z_e = z_e.at[:, idx[0], :].add(
        val[:, None] * x_e[:, idx[1], :] * y[:, idx[2], None]
    )
    z = jnp.zeros((CONV_NODES, CONV_OUT, channels), dtype=x.dtype)
    return z.at[receiver].add(z_e * s[:, s_index, :])


def conv_inputs(dtype):
    keys = iter(random.split(random.key(123), 3))
    x = random.normal(next(keys), (CONV_NODES, CONV_X, CHANNELS)).astype(dtype)
    y = random.normal(next(keys), (CONV_EDGES, CONV_Y)).astype(dtype)
    s = random.normal(next(keys), (CONV_EDGES, CONV_SCALARS, CHANNELS)).astype(dtype)
    return x, y, s


@pytest.mark.parametrize("dtype", DTYPES)
class TestConvolutionDtype:
    """Fused convolution parity over {float16, float32, float64}."""

    def _closure(self, dtype):
        idx, val, s_index = conv_problem()
        sender, receiver = conv_graph()
        idx4 = jnp.stack([idx[0], idx[1], idx[2], s_index[idx[0]]]).astype("uint8")
        coef = Coef4D(val, idx4.T, val_dtype=dtype, idx_dtype="uint8").pack_jax()
        params = CUDAConvolutionParams(num_out=CONV_OUT, num_scalars=CONV_SCALARS)

        def op(a, b, c):
            return convolution(coef, a, b, c, sender, receiver, params)

        def ref(a, b, c):
            return conv_reference(
                idx, val.astype(dtype), a, b, c, s_index, sender, receiver
            )

        return op, ref

    def test_forward(self, dtype):
        with x64_if(dtype):
            op, ref = self._closure(dtype)
            x, y, s = conv_inputs(dtype)

            result, expect = op(x, y, s), ref(x, y, s)

            assert result.dtype == jnp.dtype(dtype)
            assert result.shape == expect.shape
            assert_close(expect, result, dtype, nnz=CONV_NNZ)

    def test_backward(self, dtype):
        with x64_if(dtype):
            op, ref = self._closure(dtype)
            x, y, s = conv_inputs(dtype)

            grads = jax.grad(lambda *a: jnp.sum(op(*a)), argnums=(0, 1, 2))(x, y, s)
            refs = jax.grad(lambda *a: jnp.sum(ref(*a)), argnums=(0, 1, 2))(x, y, s)

            for got, want in zip(grads, refs):
                assert got.dtype == jnp.dtype(dtype)
                assert_close(want, got, dtype, nnz=CONV_NNZ)


# --- Public API -----------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_core_inherits_operand_dtype(dtype):
    """`core.TensorProduct` runs the fused kernel in the dtype of its operands."""
    with x64_if(dtype):
        with e3j.config.use(tensor_product="FUSED_CUDA"):
            otimes = e3j.core.TensorProduct(
                ("1x0e + 1x1o", "1x0e + 1x1o"), None, layout="TRAILING_CHANNELS"
            )

        sx, sy = otimes.source
        x = random.normal(random.key(1), (4, sx.dim)).astype(dtype)
        y = random.normal(random.key(2), (4, sy.dim)).astype(dtype)
        z = otimes(x, y)

        assert z.dtype == jnp.dtype(dtype)
        assert jnp.all(jnp.isfinite(z.astype("float32")))


def test_core_promotes_mixed_operands():
    """Mixed operand dtypes are promoted, so coef packing matches the buffers."""
    with e3j.config.use(tensor_product="FUSED_CUDA"):
        otimes = e3j.core.TensorProduct(
            ("1x0e + 1x1o", "1x0e + 1x1o"), None, layout="TRAILING_CHANNELS"
        )
    sx, sy = otimes.source
    x = random.normal(random.key(1), (4, sx.dim)).astype("float16")
    y = random.normal(random.key(2), (4, sy.dim)).astype("float32")

    assert otimes(x, y).dtype == jnp.float32


def test_unsupported_val_dtype_raises():
    """Non-float value dtypes are rejected with a clear error."""
    with pytest.raises(TypeError, match="support value dtypes"):
        resolve_val_dtype(numpy.dtype("int32"))


# --- LEADING_CHANNELS layout ----------------------------------------------
#
# A separate kernel, reducing with `atomicAdd`: no float16, see
# TestFloat16GuardedPaths.

LEADING_DTYPES = ["float32", "float64"]


def leading_inputs(dtype):
    """LEADING_CHANNELS puts the channel axis first: (rows, channels, dim)."""
    kx, ky = random.split(random.key(11))
    x = random.normal(kx, (NUM_ROWS, CHANNELS, NUM_X)).astype(dtype)
    y = random.normal(ky, (NUM_ROWS, NUM_Y)).astype(dtype)
    return x, y


def leading_reference(idx, val, x, y):
    """OUTER + LEADING_CHANNELS, channels on x only, in the operand dtype."""
    out = jnp.zeros((x.shape[0], x.shape[1], NUM_OUT), dtype=x.dtype)
    return out.at[..., idx[0]].add(val * x[..., idx[1]] * y[:, None, idx[2]])


@pytest.mark.parametrize("dtype", LEADING_DTYPES)
class TestLeadingChannelsDtype:
    """Fused LEADING_CHANNELS tensor product parity over {float32, float64}."""

    def test_forward(self, dtype):
        with x64_if(dtype):
            idx, val = tp_problem()
            x, y = leading_inputs(dtype)
            coef = pack_tp(val, idx, dtype)

            params = tp_params("LEADING_CHANNELS")
            result = tensor_product(coef, x, y, params)
            expect = leading_reference(idx, val.astype(dtype), x, y)

            assert result.dtype == jnp.dtype(dtype)
            assert result.shape == expect.shape
            assert_close(expect, result, dtype)

    def test_backward(self, dtype):
        # There is no fused backward kernel for this layout: the AD rule falls
        # back to two forward tensor products, unpacking the coefficients again.
        with x64_if(dtype):
            idx, val = tp_problem()
            x, y = leading_inputs(dtype)
            coef = pack_tp(val, idx, dtype)
            val_dt = val.astype(dtype)

            params = tp_params("LEADING_CHANNELS")
            dx, dy = jax.grad(
                lambda a, b: jnp.sum(tensor_product(coef, a, b, params)),
                argnums=(0, 1),
            )(x, y)
            rx, ry = jax.grad(
                lambda a, b: jnp.sum(leading_reference(idx, val_dt, a, b)),
                argnums=(0, 1),
            )(x, y)

            assert dx.dtype == jnp.dtype(dtype) and dy.dtype == jnp.dtype(dtype)
            assert_close(rx, dx, dtype)
            assert_close(ry, dy, dtype)


# --- scatter_add_1 --------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scatter_add_dtypes(dtype):
    """scatter_add_1 supports float32 and float64, see below for float16.

    Sorted 1-D index and 2-D source, as in test_scatter_add_op.py.
    """
    with x64_if(dtype):
        rows = 8
        # int32 pinned: x64 would default `arange` to int64, rejected by the FFI.
        segments = jnp.arange(64, dtype="int32")
        idx = jnp.repeat(segments, segments % 6)
        num_idx = idx.shape[0]
        num_out = int(jnp.max(idx)) + 1
        src = random.normal(random.key(4), (rows, num_idx)).astype(dtype)
        out = jnp.zeros((rows, num_out), dtype=dtype)

        result = scatter_add_1(idx, src, out)
        expect = out.at[:, idx].add(src)

        assert result.dtype == jnp.dtype(dtype)
        assert result.shape == expect.shape
        assert_close(expect, result, dtype, nnz=num_idx)


# --- float16 guarded paths ------------------------------------------------


class TestFloat16GuardedPaths:
    """float16 is not supported yet on the atomicAdd based paths."""

    def test_leading_channels_raises(self):
        idx, val = tp_problem()
        coef = pack_tp(val, idx, "float16")
        # Shapes are valid for the layout, so the guard is what rejects the call.
        x, y = leading_inputs("float16")
        params = Params(
            num_out=NUM_OUT, mode=MixingMode.OUTER, layout=Layout.LEADING_CHANNELS
        )
        with pytest.raises(Exception, match="float16"):
            jax.block_until_ready(tensor_product(coef, x, y, params))

    @pytest.mark.parametrize("idx_dtype", ["int32", "uint8"])
    def test_scatter_add_raises(self, idx_dtype):
        """Both index dtypes dispatch, then hit the atomicAdd(__half*) guard."""
        n_out, n_in = 8, 20
        idx = random.randint(random.key(3), (n_in,), 0, n_out).astype(idx_dtype)
        val = random.normal(random.key(4), (n_in,)).astype("float16")
        out = jnp.zeros((n_out,), dtype="float16")
        with pytest.raises(Exception, match="float16"):
            jax.block_until_ready(scatter_add_1(idx, val, out))
