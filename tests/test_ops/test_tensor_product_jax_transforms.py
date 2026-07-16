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

"""JAX integration tests for tensor_product: jit, vmap, sharding, AD."""

import re

import jax
import jax.numpy as np
import numpy.testing as testing
import pytest
from jax import Array
from jax.experimental.topologies import get_topology_desc

import e3j
from e3j.ops import CUDATensorProductParams as Params
from e3j.ops import tensor_product
from e3j.ops.coef import Coef

e3j.config(
    debug_level=0,
)

NUM_STRIPS = 10
LEN_STRIPS = 6
NUM_ROWS = 128
NUM_CHANNELS = 32


def pack_coef(val, idx, val_dtype="float32", idx_dtype="int32"):
    """Pack (nnz,) values + (3, nnz) COO indices into an opaque JAX array."""
    return Coef(val, idx.T, val_dtype=val_dtype, idx_dtype=idx_dtype).pack_jax()


def generate_tp_data(
    channels: tuple[int | None, int | None] = (None, None),
) -> tuple[int, Array, Array, Array, Array]:
    """
    Generate common test inputs for the forward and backward tests.

    Returns:
        num_out: target space dimension.
        idx: A 2D index array of leading dimension 3.
        coef: A 1D value vector of length `idx.shape[-1]`.
        x: Array of shape `(num_rows, num_x)`, or `(num_rows, channels[0], num_x)`.
        y: Array of shape `(num_rows, num_y)`, or `(num_rows, channels[1], num_y)`.
    """
    num_rows = NUM_ROWS

    # lexsorted indices
    num_out, num_x, num_y = 2, 2, 3
    idx_0 = np.tile(np.array([0, 0, 0, 1, 1, 1]), NUM_STRIPS)
    idx_1 = np.tile(np.array([0, 0, 0, 1, 1, 1]), NUM_STRIPS)
    idx_2 = np.tile(np.array([0, 1, 2, 0, 1, 2]), NUM_STRIPS)
    offsets = np.array([num_out, num_x, num_y])[:, None]
    offsets *= np.arange(NUM_STRIPS).repeat(LEN_STRIPS)
    idx = np.stack([idx_0, idx_1, idx_2]) + offsets

    x = np.stack([np.tile(np.array([1.0, 2.0]), NUM_STRIPS)] * num_rows)
    y = np.stack([np.tile(np.array([3.0, 4.0, 5.0]), NUM_STRIPS)] * num_rows)

    if channels[0] is not None:
        x = np.tile(x[:, None, :], (1, channels[0], 1))
    if channels[1] is not None:
        y = np.tile(y[:, None, :], (1, channels[1], 1))

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


def test_vmap_fwd_tensor_product_multi_devices():
    """Real jax.vmap + sharding: leading vmap axis is split across 2 devices,
    and the custom_vmap rule flattens (axis_size, num_rows) into a single
    batch dim seen by the ffi.
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

    tp_jitted = jax.jit(
        jax.vmap(tp),
        in_shardings=(sharding, sharding),
        out_shardings=sharding,
    )
    tp_compiled = tp_jitted.lower(
        jax.core.ShapedArray((128, *x.shape), x.dtype),
        jax.core.ShapedArray((128, *y.shape), y.dtype),
    ).compile()
    tp_compiled_string = tp_compiled.as_text()

    # What we expect to see in the HLO on each shard:
    #   - The vmapped batch dimension (size 128) is sharded over 2 devices,
    #     so each shard handles 64 samples. vmap then folds those into the
    #     kernel's own row axis (NUM_ROWS=128), giving 64 * 128 = 8192 rows
    #     fed into the forward kernel.
    #   - Exactly one `tensor_product` custom call per shard, with inputs
    #     (coef, x, y) and output z; coef is replicated across shards.
    #
    # Dimension legend (also referenced by the backward test below):
    #   8192  = rows_per_shard = 64 (sharded vmap) * NUM_ROWS (128)
    #   20    = x.shape[-1] = out.shape[-1] (from generate_tp_data)
    #   30    = y.shape[-1] (from generate_tp_data)
    #   60    = #non-zeros in coef = NUM_STRIPS (10) * LEN_STRIPS (6)
    #   4     = packed Coef entry (1 value + 3 index components)
    regex = re.compile(
        r"= f32\[8192,20\]\{1,0\}.+"  # output: z
        r'custom_call_target="tensor_product".+'
        r"operand_layout_constraints=\{"
        r"s32\[60,4\]\{1,0\}, "  # coef
        r"f32\[8192,20\]\{1,0\}, "  # x
        r"f32\[8192,30\]\{1,0\}\}",  # y
        flags=re.MULTILINE,
    )
    assert len(list(regex.finditer(tp_compiled_string))) == 1, (
        f"Expected exactly one tensor_product custom call, found "
        f"{len(list(regex.finditer(tp_compiled_string)))}.\n\n"
        f"HLO:\n{tp_compiled_string}"
    )


def test_vmap_grad_tensor_product_multi_devices():
    """Check that the operation is sharded as expected when vmapped and jitted over 2 device.
    This test use precompilation so it does not require a GPU to run but need to have jax version
    with gpu support.

    TODO: generalize this setup so we can use it for other tests and run them in the CI.
    """
    num_out, idx, val, x, y = generate_tp_data(channels=(128, None))
    coef = pack_coef(val, idx)

    # Use TRAILING_CHANNELS so the VJP routes through the tensor_product_bwd
    params = Params(num_out=num_out, mode="OUTER", layout="TRAILING_CHANNELS")
    x_shape = (x.shape[0], x.shape[2], x.shape[1])

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
    y_sharding = jax.sharding.NamedSharding(
        mesh=mesh,
        spec=jax.sharding.PartitionSpec("batch", None),
    )

    def tp(x, y):
        return tensor_product(coef, x, y, params)

    def sum_vmap_tp(x, y):
        return np.sum(jax.vmap(tp)(x, y))

    # A function with forward + backward tensor product passes
    fn = jax.grad(sum_vmap_tp, argnums=(0, 1))

    jitted_fn = jax.jit(
        fn,
        in_shardings=(sharding, y_sharding),
        out_shardings=(sharding, y_sharding),
    )
    compiled_fn = jitted_fn.lower(
        jax.core.ShapedArray((128, *x_shape), x.dtype),
        jax.core.ShapedArray((128, *y.shape), y.dtype),
    ).compile()  # Check that it compiles without error

    compiled_fn_string = compiled_fn.as_text()

    # Same shard / batch reasoning as the forward test
    # (`test_vmap_fwd_tensor_product_multi_devices`); see the dimension
    # legend there for 8192, 20, 30, 60, 4. Extras that only appear in the
    # backward HLO:
    #   128 = x channels (TRAILING_CHANNELS layout); y has no channel dim
    #   2   = the two transposed COO orderings stacked in coef_bwd (xzy, yzx)
    #
    # Exactly one `tensor_product_bwd` custom call per shard, with inputs
    # (coef_bwd, x, y, ct_z) and outputs (ct_x, ct_y).
    regex = re.compile(
        # outputs: (ct_x, ct_y)
        r"\(f32\[8192,20,128\]\{2,1,0\}, f32\[8192,30\]\{1,0\}\) custom-call.+"
        r'custom_call_target="tensor_product_bwd".+'
        r"operand_layout_constraints=\{"
        r"s32\[2,60,4\]\{2,1,0\}, "  # coef_bwd
        r"f32\[8192,20,128\]\{2,1,0\}, "  # x
        r"f32\[8192,30\]\{1,0\}, "  # y
        r"f32\[8192,20,128\]\{2,1,0\}\}",  # ct_z
        flags=re.MULTILINE,
    )
    assert len(list(regex.finditer(compiled_fn_string))) == 1, (
        f"Expected exactly one tensor_product_bwd custom call, found "
        f"{len(list(regex.finditer(compiled_fn_string)))}.\n\n"
        f"HLO:\n{compiled_fn_string}"
    )


def test_vmap_raises_on_batched_coef():
    num_out, idx, val, x, y = generate_tp_data()
    coef = pack_coef(val, idx)
    params = Params(num_out=num_out, mode="OUTER")
    coef_batch = np.stack([coef, coef, coef])
    with pytest.raises(jax.errors.UnexpectedTracerError):
        jax.vmap(lambda c: tensor_product(c, x, y, params))(coef_batch)


def test_vmap_batches_on_x_only():
    num_out, idx, val, x, y = generate_tp_data()
    coef = pack_coef(val, idx)
    params = Params(num_out=num_out, mode="OUTER")
    xs = np.stack([x, x, x])
    zs = jax.vmap(lambda xi: tensor_product(coef, xi, y, params))(xs)
    assert zs.shape[0] == xs.shape[0]
    testing.assert_allclose(zs[0], zs[-1])


def test_vmap_batches_on_y_only():
    num_out, idx, val, x, y = generate_tp_data()
    coef = pack_coef(val, idx)
    params = Params(num_out=num_out, mode="OUTER")
    ys = np.stack([y, y, y])
    zs = jax.vmap(lambda yi: tensor_product(coef, x, yi, params))(ys)
    assert zs.shape[0] == ys.shape[0]
    testing.assert_allclose(zs[0], zs[-1])


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
