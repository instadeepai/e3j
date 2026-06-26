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

"""JAX integration tests for convolution: vmap, sharding, AD composition."""

import re

import jax
import jax.numpy as np
import jax.random as random
import numpy.testing as testing
import pytest
from jax.experimental.topologies import get_topology_desc

import e3j
from e3j.ops.coef import Coef4D
from e3j.ops.convolution import ConvolutionParams, convolution
from e3j.utils.sparse import narrow_index_dtype

e3j.config(debug_level=0)

pytestmark = pytest.mark.e3j_ops

# Small dimensions for fast Hessian computation.
NUM_NODES = 4
NUM_EDGES = 12
NUM_X = 4
NUM_Y = 4
NUM_OUT = 4
NUM_SCALARS = 2
CHANNELS_X = 32


def pack_coef4d(val, idx4, val_dtype="float32", idx_dtype="int32"):
    return Coef4D(val, idx4.T, val_dtype=val_dtype, idx_dtype=idx_dtype).pack_jax()


def make_graph(num_nodes, num_edges, key):
    k1, k2 = random.split(key)
    sender = random.randint(k1, (num_edges,), 0, num_nodes)
    receiver = random.randint(k2, (num_edges,), 0, num_nodes)
    perm = np.argsort(receiver)
    return sender[perm], receiver[perm]


def convolution_reference(idx, val, x, y, s, s_index, sender, receiver, num_out):
    """Unfused gather -> TP -> mix -> scatter, pure JAX (COO adjacency)."""
    num_nodes = x.shape[0]
    num_edges = sender.shape[0]
    channels_x = x.shape[-1]
    x_e = x[sender]
    z_e = np.zeros((num_edges, num_out, channels_x), dtype=x.dtype)
    cxy = val[:, None] * x_e[:, idx[1], :] * y[:, idx[2], None]
    z_e = z_e.at[:, idx[0], :].add(cxy)
    z = np.zeros((num_nodes, num_out, channels_x), dtype=x.dtype)
    z = z.at[receiver].add(z_e * s[:, s_index, :])
    return z


def generate_conv_data():
    """Generate convolution test inputs with fixed seed."""
    keys = iter(random.split(random.key(42), 10))

    nnz = 20
    idx_0 = np.sort(random.randint(next(keys), (nnz,), 0, NUM_OUT))
    idx_1 = random.randint(next(keys), (nnz,), 0, NUM_X)
    idx_2 = random.randint(next(keys), (nnz,), 0, NUM_Y)
    idx = np.stack([idx_0, idx_1, idx_2])
    idx = idx.astype(narrow_index_dtype((NUM_OUT, NUM_X, NUM_Y)))
    val = random.normal(next(keys), (nnz,))

    s_index = np.sort(random.randint(next(keys), (NUM_OUT,), 0, NUM_SCALARS))
    idx4 = np.stack([idx_0, idx_1, idx_2, s_index[idx_0]])
    coef = pack_coef4d(val, idx4)
    params = ConvolutionParams(num_out=NUM_OUT, num_scalars=NUM_SCALARS)

    sender, receiver = make_graph(NUM_NODES, NUM_EDGES, next(keys))

    x = random.normal(next(keys), (NUM_NODES, NUM_X, CHANNELS_X))
    y = random.normal(next(keys), (NUM_EDGES, NUM_Y))
    s = random.normal(next(keys), (NUM_EDGES, NUM_SCALARS, CHANNELS_X))

    return coef, x, y, s, sender, receiver, params, idx, val, s_index


# --- vmap ---


def test_jit_convolution_vmap():
    """Vmap of convolution produces correct shapes and values."""
    coef, x, y, s, sender, receiver, params, *_ = generate_conv_data()

    def conv(x, y, s):
        return convolution(coef, x, y, s, sender, receiver, params)

    out_single = conv(x, y, s)
    xs, ys, ss = np.stack([x, x]), np.stack([y, y]), np.stack([s, s])
    out_vmap = jax.vmap(conv)(xs, ys, ss)

    assert out_vmap.shape == (2,) + out_single.shape
    testing.assert_allclose(out_vmap[0], out_single, atol=1e-5)
    testing.assert_allclose(out_vmap[1], out_single, atol=1e-5)


# --- sharding ---

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


def test_vmap_fwd_convolution_multi_devices():
    """Forward sharding: vmap + jit compiles over 2 simulated H100 devices."""
    coef, x, y, s, sender, receiver, params, *_ = generate_conv_data()

    topology = get_topology_desc(
        "name",
        "cuda",
        target_config=h100_gpu_target_config,
        topology="1x1x2",
    )
    mesh = jax.sharding.Mesh(topology.devices, axis_names=["batch"])
    x_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch", None, None)
    )
    y_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch", None)
    )
    s_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch", None, None)
    )

    def conv(x, y, s):
        return convolution(coef, x, y, s, sender, receiver, params)

    jitted_fn = jax.jit(
        jax.vmap(conv),
        in_shardings=(x_sharding, y_sharding, s_sharding),
        out_shardings=x_sharding,
    )

    batch = 4
    compiled = jitted_fn.lower(
        jax.core.ShapedArray((batch, *x.shape), x.dtype),
        jax.core.ShapedArray((batch, *y.shape), y.dtype),
        jax.core.ShapedArray((batch, *s.shape), s.dtype),
    ).compile()

    hlo = compiled.as_text()

    # The batch (4) is sharded over 2 devices and the custom_vmap rule folds the
    # 2 per-shard elements into the kernel row axis: node arrays get 2 * 4 = 8
    # rows, edge arrays 2 * 12 = 24. The CSR adjacency is replicated, hence tiled
    # for the full batch: sender 4 * 12 = 48, receiver_ptr 4 * 4 + 1 = 17.
    #
    # Dimension legend (also used by the backward test):
    #   8  = sharded nodes, 24 = sharded edges, 4 = NUM_OUT/X/Y, 32 = CHANNELS_X
    #   2  = NUM_SCALARS, 20 = nnz, 8 (coef) = packed Coef4D entry (1 val + 4 idx)
    regex = re.compile(
        r"= f32\[8,4,32\]\{2,1,0\} custom-call.+"  # output: messages z
        r'custom_call_target="convolution".+'
        r"operand_layout_constraints=\{"
        r"s32\[20,8\]\{1,0\}, "  # coef (packed Coef4D)
        r"f32\[8,4,32\]\{2,1,0\}, "  # x (node features)
        r"f32\[24,4\]\{1,0\}, "  # y (edge embeddings)
        r"f32\[24,2,32\]\{2,1,0\}, "  # s (edge scalars)
        r"s32\[48\]\{0\}, "  # sender (CSR, replicated)
        r"s32\[17\]\{0\}\}",  # receiver_ptr (CSR, replicated)
        flags=re.MULTILINE,
    )
    assert len(list(regex.finditer(hlo))) == 1, (
        f"Expected exactly one convolution custom call, found "
        f"{len(list(regex.finditer(hlo)))}.\n\nHLO:\n{hlo}"
    )


def test_vmap_grad_convolution_multi_devices():
    """Backward sharding: grad(sum(vmap(conv))) compiles over 2 devices."""
    coef, x, y, s, sender, receiver, params, *_ = generate_conv_data()

    topology = get_topology_desc(
        "name",
        "cuda",
        target_config=h100_gpu_target_config,
        topology="1x1x2",
    )
    mesh = jax.sharding.Mesh(topology.devices, axis_names=["batch"])
    x_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch", None, None)
    )
    y_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch", None)
    )
    s_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("batch", None, None)
    )

    def conv(x, y, s):
        return convolution(coef, x, y, s, sender, receiver, params)

    def sum_vmap_conv(x, y, s):
        return np.sum(jax.vmap(conv)(x, y, s))

    fn = jax.grad(sum_vmap_conv, argnums=(0, 1, 2))
    jitted_fn = jax.jit(
        fn,
        in_shardings=(x_sharding, y_sharding, s_sharding),
        out_shardings=(x_sharding, y_sharding, s_sharding),
    )

    batch = 4
    compiled = jitted_fn.lower(
        jax.core.ShapedArray((batch, *x.shape), x.dtype),
        jax.core.ShapedArray((batch, *y.shape), y.dtype),
        jax.core.ShapedArray((batch, *s.shape), s.dtype),
    ).compile()

    hlo = compiled.as_text()

    # Same shard/batch folding as the forward test; see its dimension legend.
    # Extras here: coef_bwd stacks the 3 transposed COO orderings (dx, dy, ds),
    # and perm (the edge sort) joins the replicated CSR adjacency operands.
    regex = re.compile(
        # outputs: (dx, dy, ds)
        r"\(f32\[8,4,32\]\{2,1,0\}, f32\[24,4\]\{1,0\}, f32\[24,2,32\]\{2,1,0\}\)"
        r" custom-call.+"
        r'custom_call_target="convolution_bwd".+'
        r"operand_layout_constraints=\{"
        r"s32\[3,20,8\]\{2,1,0\}, "  # coef_bwd (dx, dy, ds orderings)
        r"f32\[8,4,32\]\{2,1,0\}, "  # x
        r"f32\[24,4\]\{1,0\}, "  # y
        r"f32\[24,2,32\]\{2,1,0\}, "  # s
        r"f32\[8,4,32\]\{2,1,0\}, "  # dm (output cotangent)
        r"s32\[48\]\{0\}, "  # sender_t (transposed CSR)
        r"s32\[17\]\{0\}, "  # receiver_ptr_t
        r"s32\[48\]\{0\}\}",  # perm (edge permutation)
        flags=re.MULTILINE,
    )
    assert len(list(regex.finditer(hlo))) == 1, (
        f"Expected exactly one convolution_bwd custom call, found "
        f"{len(list(regex.finditer(hlo)))}.\n\nHLO:\n{hlo}"
    )


# --- Hessian (jacrev ∘ grad) ---


def test_hessian_convolution():
    """Hessian via jacrev(grad(f)) matches pure-JAX reference.

    Uses `sum(z²)` so that f is quadratic in x, yielding a non-trivial
    second derivative that exercises the double backward and vmap rules.
    """
    coef, x, y, s, sender, receiver, params, idx, val, s_index = generate_conv_data()

    def f_op(x):
        z = convolution(coef, x, y, s, sender, receiver, params)
        return np.sum(z**2)

    def f_ref(x):
        z = convolution_reference(
            idx, val, x, y, s, s_index, sender, receiver, params.num_out
        )
        return np.sum(z**2)

    H_op = jax.jacrev(jax.grad(f_op))(x)
    H_ref = jax.jacrev(jax.grad(f_ref))(x)

    testing.assert_allclose(H_op, H_ref, atol=1e-4, rtol=1e-4)
