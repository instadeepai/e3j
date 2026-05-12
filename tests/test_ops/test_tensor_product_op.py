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

import re

import jax
import jax.numpy as np
import jax.random as random
import numpy.testing as testing
import pytest
from jax import Array
from jax.experimental.topologies import get_topology_desc

import e3j
from e3j.ops import TensorProductParams as Params
from e3j.ops import tensor_product
from e3j.ops.coef import Coef
from e3j.utils.sparse import narrow_index_dtype

e3j.config(
    debug_level=0,
    tensor_product_bwd=True,
)

NUM_STRIPS = 10
LEN_STRIPS = 6
NUM_ROWS = 128
NUM_CHANNELS = 32

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


def generate_tp_data(
    channels: tuple[int | None, int | None] = (None, None),
) -> tuple[int, Array, Array, Array, Array]:
    """
    Generate common test inputs for the forward and backward tests.

    Returns:
        num_out: target space dimension.
        idx: A 2D index array of leading dimension 3.
        coef: A 1D value vector of length `idx.shape[-1]`.
        x: A 2D array of shape `(..., num_x)`
        y: A 2D array of shape `(..., num_y)`
    """
    num_rows = NUM_ROWS

    # lexsorted indices
    num_out, num_x, num_y = 2, 2, 3
    idx_0 = np.tile(np.array([0, 0, 0, 1, 1, 1]), NUM_STRIPS)
    idx_1 = np.tile(np.array([0, 0, 0, 1, 1, 1]), NUM_STRIPS)
    idx_2 = np.tile(np.array([0, 1, 2, 0, 1, 2]), NUM_STRIPS)
    #   num_out, num_x, num_y = 4, 3, 5
    #   idx_0 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3])
    #   idx_1 = np.array([0, 0, 1, 2, 0, 1, 2, 2, 0, 1, 2, 1, 2])
    #   idx_2 = np.array([0, 1, 2, 3, 0, 0, 2, 4, 0, 2, 3, 0, 4])
    offsets = np.array([num_out, num_x, num_y])[:, None]
    offsets *= np.arange(NUM_STRIPS).repeat(LEN_STRIPS)
    idx = np.stack([idx_0, idx_1, idx_2]) + offsets

    x = np.stack([np.tile(np.array([1.0, 2.0]), NUM_STRIPS)] * num_rows)
    y = np.stack([np.tile(np.array([3.0, 4.0, 5.0]), NUM_STRIPS)] * num_rows)
    #   x = np.stack([np.linspace(1, num_x, num_x)] * num_rows)
    #   y = np.stack([np.linspace(1, num_y, num_y)] * num_rows)

    if channels[0] is not None:
        x = np.tile(x[:, None, :], (1, channels[0], 1))
    if channels[1] is not None:
        y = np.tile(y[:, None, :], (1, channels[1], 1))

    #   key = random.key(789)
    #   coef = random.normal(key, (idx.shape[-1],))
    #   x = random.normal(key, (num_rows, num_x))
    #   y = random.normal(key, (num_rows, num_y))
    coef = np.ones(idx.shape[-1])
    return num_out * NUM_STRIPS, idx, coef, x, y


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
        assert x.shape[-2] == y.shape[-2]
        shape_out = (n_rows, num_out)
        out = np.zeros(shape_out, dtype=x.dtype)
        cxy = coef * x[..., idx[1]] * y[..., idx[2]]
        sum_cxy = np.sum(cxy, axis=-2)
        return out.at[..., idx[0]].add(sum_cxy)


def test_forward_tensor_product():
    """Check forward pass."""
    num_out, idx, val, x, y = generate_tp_data()
    params = Params(num_out=num_out, mode="OUTER")
    expect = tensor_product_reference(idx, val, x, y, num_out)
    result = tensor_product(pack_coef(val, idx), x, y, params)
    assert np.allclose(expect, result, atol=1e-5)


def test_forward_tensor_product_inner():
    """Check forward pass."""
    num_out, idx, val, x, y = generate_tp_data(channels=[NUM_CHANNELS] * 2)
    params = Params(num_out=num_out, mode="INNER")
    expect = tensor_product_reference(idx, val, x, y, num_out, mode="INNER")
    result = tensor_product(pack_coef(val, idx), x, y, params)
    assert np.allclose(expect, result, atol=1e-5)


def test_jit_tensor_product():
    """Check that e3j.ops.tensor_product can be compiled."""
    num_out, idx, val, x, y = generate_tp_data()
    params = Params(num_out=num_out, mode="OUTER")
    coef = pack_coef(val, idx)

    @jax.jit
    def tp(x, y, num_out):
        return tensor_product(coef, x, y, params)

    assert tp(x, y, num_out).shape[-1] == num_out


def test_jit_tensor_product_vmap():
    """Check that e3j.ops.tensor_product can be compiled."""
    num_out, idx, val, x, y = generate_tp_data()
    params = Params(num_out=num_out, mode="OUTER")
    coef = pack_coef(val, idx)
    xs = np.stack([x, x])
    ys = np.stack([y, y])

    def tp(x, y):
        return tensor_product(coef, x, y, params)

    out_normal = tp(x, y)
    out_vmap = jax.vmap(tp)(xs, ys)
    assert out_normal.shape[-1] == num_out
    assert out_vmap.shape[-1] == num_out
    assert out_vmap[0].shape == out_normal.shape
    assert np.allclose(out_vmap[0], out_normal, atol=1e-5)
    assert np.allclose(out_vmap[1], out_normal, atol=1e-5)


# Target config to simulate H100 GPU for testing
h100_gpu_target_config = """
gpu_device_info {
  threads_per_block_limit: 1024
  threads_per_warp: 32
  shared_memory_per_block: 49152
  shared_memory_per_core: 233472
  threads_per_core_limit: 2048
  core_count: 132
  fpus_per_core: 128
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65535
  block_dim_limit_z: 65535
  memory_bandwidth: 3352320000000
  l2_cache_size: 52428800
  clock_rate_ghz: 1.98
  device_memory_size: 84929347584
  shared_memory_per_block_optin: 232448
  cuda_compute_capability {
    major: 9
  }
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}
runtime_version: {
  major: 12
  minor: 8
  patch: 0
}
platform_name: "CUDA"
dnn_version_info {
  major: 9
  minor: 7
}
device_description_str: "NVIDIA H100 80GB HBM3"
"""


@pytest.mark.skip("TODO: fix vmap support with kernel_bwd")
def test_vmap_tensor_product_multi_devices():
    """Check that the operation is sharded as expetected when vmapped and jitted over 2 device.
    This test use precompilation so it does not require a GPU to run but need to have jax version
    with gpu support.

    TODO: generalize this setup so we can use it for other tests and run them in the CI.
    """
    num_out, idx, val, x, y = generate_tp_data()
    coef = pack_coef(val, idx)
    params = Params(num_out=num_out, mode="OUTER")

    topology = get_topology_desc(
        "name",
        "cuda",
        target_config=h100_gpu_target_config,
        topology="1x1x2",
    )
    mesh = jax.sharding.Mesh(topology.devices, axis_names=["batch"])
    sharding = jax.sharding.NamedSharding(
        mesh=mesh,
        spec=jax.sharding.PartitionSpec("batch", None, None),
    )

    def tp(x, y):
        return tensor_product(coef, x, y, params)

    tp_jitted = jax.jit(tp, in_shardings=(sharding, sharding), out_shardings=sharding)
    tp_compiled = tp_jitted.lower(
        jax.core.ShapedArray((128, *x.shape), x.dtype),
        jax.core.ShapedArray((128, *y.shape), y.dtype),
        # xs, ys,
    ).compile()  # Check that it compiles without error
    tp_compiled_string = tp_compiled.as_text()

    # We expect a single custom call with batch of 64.
    regex = re.compile(
        r"= f32\[64,2,2,4\].+custom_call_target=\"tensor_product\".+operand_layout_constraints={s32\[12,4\]\{1,0\}, f32\[64,2,4\]{2,1,0}, f32\[64,2,6\]\{2,1,0\}\}",
        flags=re.MULTILINE,
    )
    matches = regex.finditer(tp_compiled_string)
    assert len(list(matches)) == 1


def test_vmap_grad_tensor_product_multi_devices():
    """Check that the operation is sharded as expetected when vmapped and jitted over 2 device.
    This test use precompilation so it does not require a GPU to run but need to have jax version
    with gpu support.

    TODO: generalize this setup so we can use it for other tests and run them in the CI.
    """
    num_out, idx, val, x, y = generate_tp_data(channels=(128, None))
    coef = pack_coef(val, idx)
    params = Params(num_out=num_out, unroll=UNROLL, mode="OUTER")

    topology = get_topology_desc(
        "name",
        "cuda",
        target_config=h100_gpu_target_config,
        topology="1x1x2",
    )
    mesh = jax.sharding.Mesh(topology.devices, axis_names=["batch"])
    sharding = jax.sharding.NamedSharding(
        mesh=mesh,
        spec=jax.sharding.PartitionSpec("batch", None, None),
    )

    def tp(x, y):
        return tensor_product(coef, x, y, params)

    def l(x, y):
        return np.sum(jax.vmap(tp)(x, y))

    tp_jitted = jax.jit(
        jax.grad(l), in_shardings=(sharding, sharding), out_shardings=sharding
    )
    tp_compiled = tp_jitted.lower(
        jax.core.ShapedArray((128, *x.shape), x.dtype),
        jax.core.ShapedArray((128, *y.shape), y.dtype),
        # xs, ys,
    ).compile()  # Check that it compiles without error
    tp_compiled_string = tp_compiled.as_text()

    # We expect a single custom call with batch of 64.
    regex = re.compile(
        r"= f32\[128,128,4\].+custom_call_target=\"tensor_product\".+operand_layout_constraints={s32\[12,4\]\{1,0\}, f32\[128,128,4\]\{2,1,0\}, f32\[128,6\]\{1,0\}\}",
        flags=re.MULTILINE,
    )
    matches = regex.finditer(tp_compiled_string)
    assert len(list(matches)) == 1


_VMAP_ASSERT = "Only batching over x and y is supported"


def test_vmap_raises_on_batched_coef():
    num_out, idx, val, x, y = generate_tp_data()
    coef = pack_coef(val, idx)
    params = Params(num_out=num_out, mode="OUTER")
    coef_batch = np.stack([coef, coef, coef])
    with pytest.raises(AssertionError, match=_VMAP_ASSERT):
        jax.vmap(lambda c: tensor_product(c, x, y, params))(coef_batch)


def test_vmap_raises_on_x_only():
    num_out, idx, val, x, y = generate_tp_data()
    coef = pack_coef(val, idx)
    params = Params(num_out=num_out, mode="OUTER")
    xs = np.stack([x, x, x])
    with pytest.raises(AssertionError, match=_VMAP_ASSERT):
        jax.vmap(lambda xi: tensor_product(coef, xi, y, params))(xs)


def test_vmap_raises_on_y_only():
    num_out, idx, val, x, y = generate_tp_data()
    coef = pack_coef(val, idx)
    params = Params(num_out=num_out, mode="OUTER")
    ys = np.stack([y, y, y])
    with pytest.raises(AssertionError, match=_VMAP_ASSERT):
        jax.vmap(lambda yi: tensor_product(coef, x, yi, params))(ys)


def test_backward_tensor_product():
    """Check backward pass on x and y gradients."""
    num_out, idx, val, x, y = generate_tp_data()
    params = Params(num_out=num_out, mode="OUTER")
    coef = pack_coef(val, idx)

    def f_custom(x, y):
        return np.sum(tensor_product(coef, x, y, params))

    def f_ref(x, y):
        return np.sum(tensor_product_reference(idx, val, x, y, num_out=num_out))

    grads_custom = jax.grad(f_custom, argnums=(0, 1))(x, y)
    grads_ref = jax.grad(f_ref, argnums=(0, 1))(x, y)

    ct_x, ct_y = grads_custom
    ct_x_ref, ct_y_ref = grads_ref
    assert np.allclose(ct_x, ct_x_ref)
    assert np.allclose(ct_y, ct_y_ref)


@pytest.mark.skip("NYI: backward tensor_product_bwd")
def test_backward2_tensor_product():
    """Check two TP backward passes on x and y."""
    num_out, idx, val, x, y = generate_tp_data()
    params = Params(num_out=num_out, mode="OUTER")
    coef = pack_coef(val, idx)

    def prod_custom(x, y):
        return np.sum(tensor_product(coef, x, y, params))

    def prod_ref(x, y):
        return np.sum(tensor_product_reference(idx, val, x, y, num_out=num_out))

    def force_grads(prod: callable) -> callable:
        force_norm_x = lambda x, y: np.sum((jax.grad(prod)(x, y)) ** 2)
        return jax.grad(force_norm_x, argnums=(0, 1))

    d2x, d2y = force_grads(prod_custom)(x, y)
    d2x_ref, d2y_ref = force_grads(prod_ref)(x, y)

    assert np.allclose(d2x, d2x_ref)
    assert np.allclose(d2y, d2y_ref)


class _TestTensorProductOp:
    mode: str
    num_idx: int
    num_x: int
    num_y: int
    num_out: int
    channels_x: int | None = None
    channels_y: int | None = None
    num_rows = 16
    layout: str = "LEADING_CHANNELS"

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
        assert_allclose(expect, result, rtol=1e-5, atol=1e-5)

    def test_backward(self):
        x, y = self.inputs()
        expect_dx, expect_dy = self.bwd_ref(x, y)
        result_dx, result_dy = self.bwd_op(x, y)
        assert_allclose(expect_dx, result_dx, atol=5e-5, rtol=5e-5)
        assert_allclose(expect_dy, result_dy, atol=5e-5, rtol=5e-5)


class TestTensorProductOuter(_TestTensorProductOp):
    mode = "OUTER"
    num_idx = 530
    num_x = 16
    num_y = 25
    num_out = 12
    channels_x = 8
    channels_y = None
    num_rows = 6


class TestTensorProductInner(_TestTensorProductOp):
    mode = "INNER"
    num_idx = 535
    num_x = 12
    num_y = 43
    num_out = 31
    channels_x = 4
    channels_y = 4
    num_rows = 12


class TestTensorProductOuterTrailing(TestTensorProductOuter):
    layout = "TRAILING_CHANNELS"
    # FIXME: seems to require num_channels multiple of 32
    channels_x = 32
    channels_y = None
    num_rows = 22
    num_idx = 320


class TestTensorProductOuterTrailingCY(TestTensorProductOuterTrailing):
    layout = "TRAILING_CHANNELS"
    channels_x = 32
    channels_y = None
    num_rows = 22
    num_idx = 320


class TestTensorProductInnerTrailing(TestTensorProductInner):
    layout = "TRAILING_CHANNELS"
    channels_x = 32
    channels_y = 32


class TestTensorProductOuterOneIn(TestTensorProductOuter):
    layout = "TRAILING_CHANNELS"
    mode = "OUTER"
    channels_x = 32
    channels_y = None
    num_x = 16
    num_y = 1
    num_out = 16
    num_idx = 30
    num_rows = 2


class TestTensorProductInnerOneOut(TestTensorProductInner):
    layout = "TRAILING_CHANNELS"
    mode = "INNER"
    channels_x = 32
    channels_y = 32
    num_x = 16
    num_y = 16
    num_out = 1
    num_idx = 28
    num_rows = 2


class TestTensorProductMapTrailing(_TestTensorProductOp):
    layout = "TRAILING_CHANNELS"
    mode = "MAP"
    channels_x = 32
    channels_y = 32
    num_x = 16
    num_y = 18
    num_out = 12
    num_idx = 132
    num_rows = 4


if __name__ == "__main__":
    num_out, idx, coef, x, y = generate_tp_data()
