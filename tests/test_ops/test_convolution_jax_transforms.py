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


@pytest.mark.skipif(jax.device_count() < 2, reason="requires >= 2 devices")
def test_vmap_fwd_convolution_multi_devices_numerical():
    """Sharded forward equals single-device vmap over distinct graphs."""
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


@pytest.mark.skipif(jax.device_count() < 2, reason="requires >= 2 devices")
def test_vmap_grad_convolution_multi_devices_numerical():
    """Sharded gradients equal single-device vmap gradients over distinct graphs."""
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
