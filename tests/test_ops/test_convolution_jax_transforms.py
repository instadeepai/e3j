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
import numpy as onp
import numpy.testing as testing
import pytest
from jax.experimental.topologies import get_topology_desc

import e3j
from e3j.core.convolution import Convolution
from e3j.ops.coef import Coef4D
from e3j.ops.convolution import (
    ConvolutionBatchingWarning,
    CUDAConvolutionParams,
    convolution,
)
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
    params = CUDAConvolutionParams(num_out=NUM_OUT, num_scalars=NUM_SCALARS)

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


# --- vmap over a padded (node_mask) graph ---
#
# `node_mask` marks edges touching a padding node with `DUMMY_INDEX`, dropping
# them from the CSR (no kernel work). Folding several graphs onto one device
# interleaves each graph's sentinels with the next graph's real edges, so the
# vmap fold reassigns every dummy's grouping endpoint to its graph's last node,
# keeping the CSR contiguous while the guard endpoint stays out of range and the
# kernel skips it. These pin the fused kernels against the plain JAX path on that
# configuration, for both orderings.

_MASK_NODES = 5
_MASK_CHANNELS = 32
_MASK_SENDER = np.array([1, 2, 0, 2, 3, 1, 4, 2], dtype=np.int32)
_MASK_RECEIVER = np.array([0, 0, 1, 2, 2, 3, 3, 4], dtype=np.int32)


def _masked_module(impl):
    with e3j.config.use(convolution=impl, tensor_product=impl):
        return Convolution(
            ("2x0e + 1x1o", "1x0e + 1x1o"),
            None,
            layout="TRAILING_CHANNELS",
            graph_ordering="RECEIVER",
        )


def _masked_features(seed=0):
    node_space, edge_space, scalar_space = _masked_module("UNFUSED").source
    num_edges = _MASK_SENDER.shape[0]
    k = random.split(random.key(seed), 3)
    return (
        random.normal(k[0], (_MASK_NODES, node_space.dim, _MASK_CHANNELS)),
        random.normal(k[1], (num_edges, edge_space.dim)),
        random.normal(k[2], (num_edges, scalar_space.dim, _MASK_CHANNELS)),
    )


def _vmap_masked(impl, x, y, s, node_mask):
    """Vmap `impl` over two copies of the same masked graph."""
    xs, ys, ss = np.stack([x, x]), np.stack([y, y]), np.stack([s, s])
    senders = np.stack([_MASK_SENDER, _MASK_SENDER])
    receivers = np.stack([_MASK_RECEIVER, _MASK_RECEIVER])
    masks = np.stack([node_mask, node_mask])

    def f(x, y, s, sender, receiver, node_mask):
        return _masked_module(impl)(x, y, s, sender, receiver, node_mask=node_mask)

    return jax.vmap(f)(xs, ys, ss, senders, receivers, masks)


def test_masked_vmap_forward_matches_unfused():
    """Fused forward over a padded, vmapped graph equals the plain-JAX path."""
    x, y, s = _masked_features()
    node_mask = np.ones(_MASK_NODES, dtype=bool).at[-1].set(False)

    with pytest.warns(ConvolutionBatchingWarning):
        fused = _vmap_masked("FUSED_CUDA", x, y, s, node_mask)
    unfused = _vmap_masked("UNFUSED", x, y, s, node_mask)

    testing.assert_allclose(fused, unfused, atol=1e-4, rtol=1e-4)


def test_masked_vmap_backward_matches_unfused():
    """Fused gradients over a padded, vmapped graph equal the plain-JAX path."""
    x, y, s = _masked_features()
    node_mask = np.ones(_MASK_NODES, dtype=bool).at[-1].set(False)

    def loss(impl):
        return lambda x, y, s: np.sum(_vmap_masked(impl, x, y, s, node_mask))

    with pytest.warns(ConvolutionBatchingWarning):
        g_fused = jax.grad(loss("FUSED_CUDA"), argnums=(0, 1, 2))(x, y, s)
    g_unfused = jax.grad(loss("UNFUSED"), argnums=(0, 1, 2))(x, y, s)

    for a, b in zip(g_fused, g_unfused):
        testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)


def _sender_module(impl):
    with e3j.config.use(convolution=impl, tensor_product=impl):
        return Convolution(
            ("2x0e + 1x1o", "1x0e + 1x1o"),
            None,
            layout="TRAILING_CHANNELS",
            graph_ordering="SENDER",
        )


def test_masked_vmap_sender_matches_unfused():
    """SENDER-ordered fused fwd+bwd over a padded, vmapped, symmetric graph.

    SENDER ordering aggregates over reversed edges and skips the backward edge
    permutation, so its backward CSR is built directly and has no `transpose`
    re-sort. The vmap fold reassigns every dummy's sender to its graph's last
    node so the direct CSR stays contiguous and the kernel guard skips the
    sentinels; without it the interleaved dummies would corrupt the adjacency.
    """
    node_space, edge_space, scalar_space = _sender_module("UNFUSED").source

    # Symmetric real graph (nodes 0-3) plus a masked self-loop on padding node 4.
    pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    edges = pairs + [(b, a) for a, b in pairs] + [(4, 4)]
    sender = np.array([a for a, b in edges], np.int32)
    receiver = np.array([b for a, b in edges], np.int32)
    order = np.argsort(sender, stable=True)
    sender, receiver = sender[order], receiver[order]

    # Reversed edges carry the parity-signed feature that SENDER ordering assumes.
    parity = onp.concatenate(
        [onp.full(m * ir.dim, float(ir.p), onp.float32) for m, ir in edge_space]
    )
    k = random.split(random.key(0), 3)
    yh = random.normal(k[0], (len(pairs), edge_space.dim))
    y = np.concatenate([yh, yh * parity[None], np.zeros((1, edge_space.dim))])[order]
    sh = random.normal(k[1], (len(pairs), scalar_space.dim, _MASK_CHANNELS))
    s = np.concatenate([sh, sh, np.zeros((1, scalar_space.dim, _MASK_CHANNELS))])[order]
    x = random.normal(k[2], (_MASK_NODES, node_space.dim, _MASK_CHANNELS))
    node_mask = np.ones(_MASK_NODES, dtype=bool).at[-1].set(False)

    def run(impl, x, y, s):
        xs, ys, ss = np.stack([x, x]), np.stack([y, y]), np.stack([s, s])
        sm, rm = np.stack([sender, sender]), np.stack([receiver, receiver])
        mm = np.stack([node_mask, node_mask])

        def f(x, y, s, se, re, nm):
            return _sender_module(impl)(x, y, s, se, re, node_mask=nm)

        return jax.vmap(f)(xs, ys, ss, sm, rm, mm)

    with pytest.warns(ConvolutionBatchingWarning):
        fused = run("FUSED_CUDA", x, y, s)
    unfused = run("UNFUSED", x, y, s)
    testing.assert_allclose(fused, unfused, atol=1e-4, rtol=1e-4)

    with pytest.warns(ConvolutionBatchingWarning):
        g_fused = jax.grad(lambda *a: np.sum(run("FUSED_CUDA", *a)), argnums=(0, 1, 2))(
            x, y, s
        )
    g_unfused = jax.grad(lambda *a: np.sum(run("UNFUSED", *a)), argnums=(0, 1, 2))(
        x, y, s
    )
    for a, b in zip(g_fused, g_unfused):
        testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)


def _single_masked(impl, x, y, s, node_mask):
    """Single-graph masked convolution (no vmap)."""
    return _masked_module(impl)(
        x, y, s, _MASK_SENDER, _MASK_RECEIVER, node_mask=node_mask
    )


def test_masked_forward_zeros_padding_node():
    """Forward: a padding node aggregates no edge, so its message is exact zero.

    Every path drops edges touching the masked node, leaving it with no
    receiver; the kernel visits it, finds an empty range and stores zeros,
    matching the plain-JAX scatter into a zero-initialized buffer.
    """
    x, y, s = _masked_features()
    node_mask = np.ones(_MASK_NODES, dtype=bool).at[-1].set(False)

    fused = _single_masked("FUSED_CUDA", x, y, s, node_mask)
    unfused = _single_masked("UNFUSED", x, y, s, node_mask)

    testing.assert_array_equal(jax.device_get(fused[-1]), 0.0)
    testing.assert_allclose(fused, unfused, atol=1e-4, rtol=1e-4)


def test_masked_backward_zeros_disconnected_dx():
    """Backward: dx of a padding node (no outgoing edge) is exact zero.

    With its edges dropped the node is never a sender, so its feature
    cotangent has no contribution. The kernel's grid-stride loop still visits
    every node and writes a zeroed `dx` row, matching the plain-JAX gradient.
    """
    x, y, s = _masked_features()
    node_mask = np.ones(_MASK_NODES, dtype=bool).at[-1].set(False)

    def dx(impl):
        return jax.grad(lambda x: np.sum(_single_masked(impl, x, y, s, node_mask)))(x)

    dx_fused = dx("FUSED_CUDA")
    dx_unfused = dx("UNFUSED")

    testing.assert_array_equal(jax.device_get(dx_fused[-1]), 0.0)
    testing.assert_allclose(dx_fused, dx_unfused, atol=1e-4, rtol=1e-4)


@pytest.mark.multi_devices
def test_vmap_convolution_execute_multi_devices():
    "Check that running on multiple device give the same result as each run individualy."
    coef, x, y, s, sender, receiver, params, idx, val, s_index = generate_conv_data()

    batch = 3 * jax.device_count()
    keys = random.split(random.key(5), batch)
    graphs = [make_graph(NUM_NODES, NUM_EDGES, k) for k in keys]
    senders = np.stack([g[0] for g in graphs])
    receivers = np.stack([g[1] for g in graphs])

    kx, ky, ks = random.split(random.key(11), 3)
    xs = random.normal(kx, (batch, *x.shape))
    ys = random.normal(ky, (batch, *y.shape))
    ss = random.normal(ks, (batch, *s.shape))

    def conv(x, y, s, sender, receiver):
        return convolution(coef, x, y, s, sender, receiver, params)

    mesh = jax.sharding.Mesh(jax.devices(), ("batch",))
    with mesh:
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))

        jitted_fn = jax.jit(
            jax.vmap(conv),
            in_shardings=(sharding, sharding, sharding, sharding, sharding),
            out_shardings=sharding,
        )
        out = jitted_fn(*jax.device_put((xs, ys, ss, senders, receivers), sharding))

        assert out.shape == (batch, NUM_NODES, NUM_OUT, CHANNELS_X)
        for b in range(batch):
            ref = convolution_reference(
                idx,
                val,
                xs[b],
                ys[b],
                ss[b],
                s_index,
                senders[b],
                receivers[b],
                NUM_OUT,
            )
            testing.assert_allclose(out[b], ref, atol=1e-4, rtol=1e-4)


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
    # rows, edge arrays 2 * 12 = 24. The adjacency is passed COO and sharded
    # edge-wise like the edge features (sender 24), and the shard-local CSR
    # pointer is rebuilt from those local edges (receiver_ptr 2 * 4 + 1 = 9), so
    # each device convolves only its own block of the batched graph.
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
        r"s32\[24\]\{0\}, "  # sender (COO, sharded edge-wise)
        r"s32\[9\]\{0\}\}",  # receiver_ptr (shard-local CSR)
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
    # The adjacency is sharded COO edge-wise (sender/receiver 24 each), and the
    # transposed shard-local CSR is rebuilt inside the kernel region: sender_t
    # 24 (edges), receiver_ptr_t 9 (nodes + 1), perm 24 (edge sort). Extra:
    # coef_bwd stacks the 3 transposed COO orderings (dx, dy, ds).
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
        r"s32\[24\]\{0\}, "  # sender_t (shard-local transposed CSR)
        r"s32\[9\]\{0\}, "  # receiver_ptr_t
        r"s32\[24\]\{0\}\}",  # perm (edge permutation)
        flags=re.MULTILINE,
    )
    assert len(list(regex.finditer(hlo))) == 1, (
        f"Expected exactly one convolution_bwd custom call, found "
        f"{len(list(regex.finditer(hlo)))}.\n\nHLO:\n{hlo}"
    )


# --- sharding (numerical, real multi-device) ---
#
# The HLO tests above pin the *shape* of the sharded custom call but never
# execute it. These run the real kernels on >= 2 devices and check that data
# parallelism over a batch of *distinct* graphs matches single-device vmap
# exactly. They are the regression guard for the cross-shard adjacency bleed
# that a replicated global CSR caused (every shard but the first reused the
# first shard's adjacency), which shape-only checks cannot catch.


def _batched_conv_data(batch, key):
    """A batch of `batch` distinct random graphs sharing packed coefficients."""
    coef, x0, y0, s0, *_rest, params, _idx, _val, _s_index = generate_conv_data()
    xs, ys, ss, senders, receivers = [], [], [], [], []
    for k in random.split(key, batch):
        kg, kx, ky, ks = random.split(k, 4)
        sender, receiver = make_graph(NUM_NODES, NUM_EDGES, kg)
        senders.append(sender)
        receivers.append(receiver)
        xs.append(random.normal(kx, x0.shape))
        ys.append(random.normal(ky, y0.shape))
        ss.append(random.normal(ks, s0.shape))
    stack = lambda arrs: np.stack(arrs)
    return (
        coef,
        stack(xs),
        stack(ys),
        stack(ss),
        stack(senders),
        stack(receivers),
        params,
    )


def test_vmap_fwd_convolution_multi_devices_numerical():
    """Sharded forward equals single-device vmap over distinct graphs."""
    if jax.device_count() < 2:
        pytest.skip("requires >= 2 devices")
    coef, x, y, s, sender, receiver, params = _batched_conv_data(
        batch=2 * jax.device_count(), key=random.key(0)
    )

    def conv(x, y, s, sender, receiver):
        return convolution(coef, x, y, s, sender, receiver, params)

    batched = jax.vmap(conv)
    out_ref = batched(x, y, s, sender, receiver)

    mesh = jax.sharding.Mesh(jax.devices(), ("dp",))
    sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))
    out_sharded = jax.jit(batched, in_shardings=(sh,) * 5, out_shardings=sh)(
        x, y, s, sender, receiver
    )

    # Each graph is convolved by one kernel launch on its own shard, so the
    # match is bit-exact rather than merely close.
    testing.assert_array_equal(jax.device_get(out_sharded), jax.device_get(out_ref))


def test_vmap_grad_convolution_multi_devices_numerical():
    """Sharded gradients equal single-device vmap gradients over distinct graphs."""
    if jax.device_count() < 2:
        pytest.skip("requires >= 2 devices")
    coef, x, y, s, sender, receiver, params = _batched_conv_data(
        batch=2 * jax.device_count(), key=random.key(1)
    )

    def loss(x, y, s, sender, receiver):
        z = jax.vmap(lambda *a: convolution(coef, *a, params))(
            x, y, s, sender, receiver
        )
        return np.sum(z**2)

    grad_fn = jax.grad(loss, argnums=(0, 1, 2))
    g_ref = grad_fn(x, y, s, sender, receiver)

    mesh = jax.sharding.Mesh(jax.devices(), ("dp",))
    sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("dp"))
    g_sharded = jax.jit(grad_fn, in_shardings=(sh,) * 5, out_shardings=(sh, sh, sh))(
        x, y, s, sender, receiver
    )

    for ref, shard in zip(g_ref, g_sharded):
        testing.assert_array_equal(jax.device_get(shard), jax.device_get(ref))


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


def test_hvp_convolution_all_inputs():
    """Hessian-vector products match reference across x, y and s.

    mlip trains on Hessian labels, so the double backward must be correct for
    every differentiable input -- not just x. This exercises the cross-input
    second-order terms in `convolution_bwd._bwd` (Dx, Dy, Ds mixing) via a
    reverse-over-reverse HVP, which `custom_vjp` supports (unlike the
    forward-over-reverse `jax.hessian`, see `test_hessian_forward_mode_error`).
    """
    coef, x, y, s, sender, receiver, params, idx, val, s_index = generate_conv_data()

    def f_op(inputs):
        x, y, s = inputs
        z = convolution(coef, x, y, s, sender, receiver, params)
        return np.sum(z**2)

    def f_ref(inputs):
        x, y, s = inputs
        z = convolution_reference(
            idx, val, x, y, s, s_index, sender, receiver, params.num_out
        )
        return np.sum(z**2)

    primals = (x, y, s)
    kx, ky, ks = random.split(random.key(7), 3)
    tangents = (
        random.normal(kx, x.shape),
        random.normal(ky, y.shape),
        random.normal(ks, s.shape),
    )

    def hvp(f):
        def directional(p):
            grads = jax.grad(f)(p)
            dots = jax.tree_util.tree_map(lambda g, t: np.vdot(g, t), grads, tangents)
            return jax.tree_util.tree_reduce(lambda a, b: a + b, dots)

        return jax.grad(directional)(primals)

    Hv_op = jax.tree_util.tree_leaves(hvp(f_op))
    Hv_ref = jax.tree_util.tree_leaves(hvp(f_ref))
    for a, b in zip(Hv_op, Hv_ref):
        testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)


def test_third_order_convolution_all_inputs():
    """Third derivative (reverse^3) matches reference and leaks no tracers.

    The messages are trilinear in (x, y, s), so `sum(z**2)` has non-vanishing
    mixed third derivatives. Composing three `jax.grad` passes nests three
    reverse-mode traces through the convolution `custom_vjp`; each level
    re-threads `sender`/`receiver` through its residuals rather than closing
    over them. This guards against tracer leaks one order beyond the double
    backward exercised by `test_hvp_convolution_all_inputs`.
    """
    coef, x, y, s, sender, receiver, params, idx, val, s_index = generate_conv_data()

    def f_op(inputs):
        x, y, s = inputs
        z = convolution(coef, x, y, s, sender, receiver, params)
        return np.sum(z**2)

    def f_ref(inputs):
        x, y, s = inputs
        z = convolution_reference(
            idx, val, x, y, s, s_index, sender, receiver, params.num_out
        )
        return np.sum(z**2)

    primals = (x, y, s)
    k1, k2, k3 = random.split(random.key(11), 3)

    def tangents_like(key):
        kx, ky, ks = random.split(key, 3)
        return (
            random.normal(kx, x.shape),
            random.normal(ky, y.shape),
            random.normal(ks, s.shape),
        )

    t1, t2, t3 = (tangents_like(k) for k in (k1, k2, k3))

    def directional(f, tangents):
        """Contract `grad(f)` with `tangents` into a scalar."""

        def d(p):
            grads = jax.grad(f)(p)
            dots = jax.tree_util.tree_map(lambda g, t: np.vdot(g, t), grads, tangents)
            return jax.tree_util.tree_reduce(lambda a, b: a + b, dots)

        return d

    # Third-order directional derivative: reverse-over-reverse-over-reverse.
    def d3(f):
        return directional(directional(directional(f, t1), t2), t3)

    out_op = d3(f_op)(primals)
    out_ref = d3(f_ref)(primals)

    testing.assert_allclose(out_op, out_ref, atol=1e-3, rtol=1e-3)


def test_hessian_forward_mode_error():
    """`jax.hessian` (forward-over-reverse) is unsupported; use jacrev(grad).

    The convolution is a `custom_vjp`, which defines reverse mode only. The
    default `jax.hessian = jacfwd(jacrev)` needs forward-mode through it and
    must fail cleanly, documenting that Hessian labels go through
    `jax.jacrev(jax.grad(...))` instead.
    """
    coef, x, y, s, sender, receiver, params, *_ = generate_conv_data()

    def f_op(x):
        z = convolution(coef, x, y, s, sender, receiver, params)
        return np.sum(z**2)

    # eval_shape triggers the trace-time forward-mode guard deterministically,
    # independent of platform / FFI kernel availability.
    with pytest.raises(TypeError, match="forward-mode autodiff"):
        jax.eval_shape(jax.hessian(f_op), x)


def test_double_backward_batched_graph():
    """Double backward over a stacked (vmapped) graph matches reference.

    Regression for a tracer leak in `convolution_bwd._bwd`: mlip trains on
    forces, so parameter gradients differentiate through `grad(energy)` -- the
    convolution double backward. With a batch axis over the graph, the double
    backward used to close over `sender`/`receiver` and leak them across the
    backward trace. This stacks distinct graphs and vmaps a force-style
    `grad(grad(...))`, exercising that path end to end against the reference.
    """
    coef, x, y, s, sender, receiver, params, idx, val, s_index = generate_conv_data()

    batch = 3
    keys = random.split(random.key(0), batch)
    graphs = [make_graph(NUM_NODES, NUM_EDGES, k) for k in keys]
    senders = np.stack([g[0] for g in graphs])
    receivers = np.stack([g[1] for g in graphs])
    xs = random.normal(random.key(1), (batch, *x.shape))
    ys = random.normal(random.key(2), (batch, *y.shape))
    ss = random.normal(random.key(3), (batch, *s.shape))

    def force_loss(conv_fn):
        def scalar(x, y, s, sender, receiver):
            # "force" = grad of a scalar energy wrt node features x
            def energy(x):
                return np.sum(conv_fn(x, y, s, sender, receiver) ** 2)

            return np.sum(jax.grad(energy)(x) ** 2)

        return scalar

    def op_conv(x, y, s, sender, receiver):
        return convolution(coef, x, y, s, sender, receiver, params)

    def ref_conv(x, y, s, sender, receiver):
        return convolution_reference(
            idx, val, x, y, s, s_index, sender, receiver, params.num_out
        )

    grad_op = jax.vmap(jax.grad(force_loss(op_conv), argnums=(0, 1, 2)))
    grad_ref = jax.vmap(jax.grad(force_loss(ref_conv), argnums=(0, 1, 2)))

    # The stacked graph trips the (expected) batching warning on the op path.
    with pytest.warns(ConvolutionBatchingWarning):
        g_op = grad_op(xs, ys, ss, senders, receivers)
    g_ref = grad_ref(xs, ys, ss, senders, receivers)

    for a, b in zip(g_op, g_ref):
        # Looser tolerance: grad-of-grad accumulates float32 error over large
        # message magnitudes, leaving a handful of elements just past 1e-4.
        testing.assert_allclose(a, b, atol=2e-4, rtol=3e-4)


def test_hessian_under_vmap():
    """`vmap(jacrev(grad(...)))` matches per-element reference Hessians.

    jacrev uses vmap internally, so this nests two vmap levels around the
    convolution. The `custom_vmap` rule peels one axis per level and defers to
    the unbatched base case (which shards via custom_partitioning), so nested
    vmap composes without a custom_partitioning batching rule.
    """
    coef, x, y, s, sender, receiver, params, idx, val, s_index = generate_conv_data()

    batch = 3
    xs = random.normal(random.key(3), (batch, *x.shape))

    def f_op(x):
        return np.sum(convolution(coef, x, y, s, sender, receiver, params) ** 2)

    def f_ref(x):
        z = convolution_reference(
            idx, val, x, y, s, s_index, sender, receiver, params.num_out
        )
        return np.sum(z**2)

    H_op = jax.vmap(jax.jacrev(jax.grad(f_op)))(xs)
    H_ref = np.stack([jax.jacrev(jax.grad(f_ref))(xb) for xb in xs])

    testing.assert_allclose(H_op, H_ref, atol=1e-4, rtol=1e-4)


def test_vmap_grad_hessian_all_inputs():
    """`vmap(grad(hessian(...)))` matches reference, differentiating x, y and s.

    Contracting the Hessian with two tangents gives a scalar; one more grad is
    a third-order reverse pass that reaches every input. vmap batches it over
    distinct graphs, testing the recursive vmap rule with the double backward.
    """
    coef, x, y, s, sender, receiver, params, idx, val, s_index = generate_conv_data()

    batch = 3
    kx, ky, ks = random.split(random.key(5), 3)
    xs = random.normal(kx, (batch, *x.shape))
    ys = random.normal(ky, (batch, *y.shape))
    ss = random.normal(ks, (batch, *s.shape))

    def tangents(key):
        kx, ky, ks = random.split(key, 3)
        return (
            random.normal(kx, x.shape),
            random.normal(ky, y.shape),
            random.normal(ks, s.shape),
        )

    t1, t2 = tangents(random.key(8)), tangents(random.key(9))

    def directional(f, tangents):
        """Contract `grad(f)` with `tangents` into a scalar."""

        def d(p):
            grads = jax.grad(f)(p)
            dots = jax.tree_util.tree_map(lambda g, t: np.vdot(g, t), grads, tangents)
            return jax.tree_util.tree_reduce(lambda a, b: a + b, dots)

        return d

    def grad_hessian(conv_fn):
        def f(p):
            x, y, s = p
            return np.sum(conv_fn(x, y, s) ** 2)

        # grad of the twice-contracted Hessian: gradients for x, y and s.
        return jax.grad(directional(directional(f, t1), t2))

    def op_conv(x, y, s):
        return convolution(coef, x, y, s, sender, receiver, params)

    def ref_conv(x, y, s):
        return convolution_reference(
            idx, val, x, y, s, s_index, sender, receiver, params.num_out
        )

    g_op = jax.vmap(grad_hessian(op_conv))((xs, ys, ss))
    g_ref = jax.vmap(grad_hessian(ref_conv))((xs, ys, ss))

    for a, b in zip(jax.tree_util.tree_leaves(g_op), jax.tree_util.tree_leaves(g_ref)):
        testing.assert_allclose(a, b, atol=1e-3, rtol=1e-3)
