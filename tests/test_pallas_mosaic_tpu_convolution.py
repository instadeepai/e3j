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

"""The fused Mosaic-TPU convolution must match the unfused reference.

The forward relies on an edge swap: it reinterprets each edge as its mirror (swap
senders/receivers, fold `Y(-r) = (-1)^l Y(r)` into the CG coefficients), which
reproduces the convolution over a sender-sorted edge list when it is symmetric
with `edge_features` (sph) odd and `edge_scalars` even under edge reversal. The
backward runs the natural edges directly. The inputs below satisfy that contract.
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from e3j.core.convolution import Convolution
from e3j.data.graph import GraphCSR
from e3j.pallas_ops.convolution.mosaic_tpu import (
    PallasMosaicTPUMessagePassingConvolutionBwdConfig,
    PallasMosaicTPUMessagePassingConvolutionFwdConfig,
    PallasMosaicTPUMessagePassingConvolutionParams,
    convolution_mosaic_tpu,
)
from e3j.spaces import O3Space
from e3j.utils import config, options
from e3j.utils.options import Layout

pytestmark = pytest.mark.mosaic_tpu

RTOL = 1e-3
ATOL = 1e-3

SLOW = pytest.mark.slow


def _case(x_ir, y_ir, o_ir, channels, *marks):
    return pytest.param(
        x_ir, y_ir, o_ir, channels, marks=marks, id=f"{x_ir}__channels{channels}"
    )


CASES = [
    # dense and packed
    _case("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", 128),
    _case("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", 64),
    # output irreps with multiplicity > 1 (dense + packed)
    _case("0e + 2x1o", "0e + 1o", "0e + 1o + 2e", 128, SLOW),
    _case("0e + 2x1o", "0e + 1o", "0e + 1o + 2e", 32, SLOW),
    # l_max=3 (dense + packed)
    _case("0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", 128, SLOW),
    _case("0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", 32, SLOW),
    _case("0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", 256, SLOW),
]

# The per-edge gradient relationships are checked on fewer cases for speed.
BWD_CASES = [
    _case("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", 128),
    _case("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", 32),
]


@pytest.fixture(autouse=True)
def _require_tpu():
    """Skip (rather than error) when not running on a TPU backend."""
    if jax.default_backend() != "tpu":
        pytest.skip("mosaic_tpu tests require a TPU backend")


@pytest.fixture(autouse=True)
def _plain_jax_reference():
    with config.use(tensor_product=options.TensorProduct.UNFUSED):
        yield


def _y_parity_signs(y_space: O3Space) -> np.ndarray:
    """Parity `ir.p` per coordinate of the edge space (matches `params.swapped`)."""
    return np.concatenate(
        [
            np.full(channels * (2 * ir.l + 1), float(ir.p), np.float32)
            for channels, ir in y_space
        ]
    )


def _contract_valid_inputs(
    conv: Convolution, n_nodes: int, channels: int, seed: int = 0
):
    """Symmetric, sender-sorted edges with sph odd / edge_scalars even under
    reversal - the contract the fused forward's swap trick assumes."""
    rng = np.random.default_rng(seed)
    x_dim = conv._otimes.source[0].dim
    y_dim = conv._otimes.source[1].dim
    n_irreps = conv._mix.num_irreps
    y_signs = _y_parity_signs(conv._otimes.source[1])

    # Complete directed graph (every (i, j) has its mirror (j, i)), sender-sorted.
    edges = sorted((i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j)
    n_edges = len(edges)

    sph = np.zeros((n_edges, y_dim), np.float32)
    es = np.zeros((n_edges, n_irreps, channels), np.float32)
    base_sph: dict = {}
    base_es: dict = {}
    for e, (i, j) in enumerate(edges):
        key = (min(i, j), max(i, j))
        if key not in base_sph:
            base_sph[key] = rng.standard_normal(y_dim).astype(np.float32)
            base_es[key] = rng.standard_normal((n_irreps, channels)).astype(np.float32)
        # forward edge keeps the base; the mirror gets the (-1)^l parity flip.
        sph[e] = base_sph[key] if i < j else y_signs * base_sph[key]
        es[e] = base_es[key]  # radial scalars are even under reversal

    node = rng.standard_normal((n_nodes, x_dim, channels)).astype(np.float32)
    senders = np.array([s for s, _ in edges], np.int32)
    receivers = np.array([r for _, r in edges], np.int32)
    return (
        jnp.asarray(node),
        jnp.asarray(sph),
        jnp.asarray(es),
        jnp.asarray(senders),
        jnp.asarray(receivers),
    )


def _sum_sq_grads(conv_fn):
    """Gradients of `sum(conv_fn(...) ** 2)` w.r.t. node feats, sph and edge scalars."""
    return jax.grad(lambda n, s, e, *g: jnp.sum(conv_fn(n, s, e, *g) ** 2), (0, 1, 2))


def _second_grads(conv_fn):
    """Force-training style second-order gradients through `conv_fn`."""

    def loss(n, s, e, *g):
        d_node = jax.grad(lambda nn: jnp.sum(conv_fn(nn, s, e, *g) ** 2))(n)
        return jnp.sum(d_node**2)

    return jax.grad(loss, (0, 1, 2))


def _vjp(conv_fn):
    """Cotangents of `conv_fn` for an explicit output cotangent `ct`."""

    def vjp(n, s, e, senders, receivers, ct):
        return jax.vjp(lambda *a: conv_fn(*a, senders, receivers), n, s, e)[1](ct)

    return vjp


class _Kernels(NamedTuple):
    """A `Convolution` with its fused and unfused entry points, jitted."""

    conv: Convolution
    fused: Callable
    unfused: Callable
    fused_grads: Callable
    unfused_grads: Callable
    fused_second_grads: Callable
    unfused_second_grads: Callable
    fused_vjp: Callable
    unfused_vjp: Callable


@functools.cache
def _kernels(x_ir: str, y_ir: str, o_ir: str) -> _Kernels:
    """One `Convolution` and its jitted entry points per irreps triple, shared by
    every test so each shape compiles once per session. The graph and the optional
    node mask are traced arguments, so every graph of the same size reuses them."""
    conv = Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    fused, unfused = conv._fused_mosaic_tpu_eval, conv._unfused_eval
    return _Kernels(
        conv=conv,
        fused=jax.jit(fused),
        unfused=jax.jit(unfused),
        fused_grads=jax.jit(_sum_sq_grads(fused)),
        unfused_grads=jax.jit(_sum_sq_grads(unfused)),
        fused_second_grads=jax.jit(_second_grads(fused)),
        unfused_second_grads=jax.jit(_second_grads(unfused)),
        fused_vjp=jax.jit(_vjp(fused)),
        unfused_vjp=jax.jit(_vjp(unfused)),
    )


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def _symmetric_edges(n_nodes: int):
    """Complete directed graph (every (i, j) has its mirror (j, i)), sender-sorted."""
    edges = sorted((i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j)
    return (
        jnp.array([s for s, _ in edges], jnp.int32),
        jnp.array([r for _, r in edges], jnp.int32),
    )


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES)
def test_fused_matches_unfused(x_ir, y_ir, o_ir, channels):
    k = _kernels(x_ir, y_ir, o_ir)
    inputs = _contract_valid_inputs(k.conv, n_nodes=16, channels=channels)

    expected = k.unfused(*inputs)
    got = k.fused(*inputs)

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES)
def test_backward_cotangents_match(x_ir, y_ir, o_ir, channels):
    k = _kernels(x_ir, y_ir, o_ir)
    inputs = _contract_valid_inputs(k.conv, n_nodes=8, channels=channels)

    out = k.unfused(*inputs)
    # all three gradients, for an arbitrary output cotangent uncorrelated with the output
    ct = jnp.asarray(np.random.default_rng(1).standard_normal(out.shape), jnp.float32)
    df = k.fused_vjp(*inputs, ct)
    du = k.unfused_vjp(*inputs, ct)
    for name, a, b in zip(("d_node_feats", "d_sph", "d_edge_scalars"), df, du):
        assert _rel(a, b) < RTOL, f"{name} mismatch"


@SLOW
@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_force_training_double_backward(x_ir, y_ir, o_ir, channels):
    k = _kernels(x_ir, y_ir, o_ir)
    inputs = _contract_valid_inputs(k.conv, n_nodes=8, channels=channels)

    d2f = k.fused_second_grads(*inputs)
    d2u = k.unfused_second_grads(*inputs)
    for name, a, b in zip(("dd_node", "dd_sph", "dd_edge_scalars"), d2f, d2u):
        assert _rel(a, b) < RTOL, f"{name} mismatch"


N_REAL = 8
N_PAD = 3


def _padded_inputs(conv, channels, seed: int = 0):
    """Contract-valid graph plus padding nodes and tail dummy edges of junk data."""
    node, sph, es, senders, receivers = _contract_valid_inputs(
        conv, N_REAL, channels, seed
    )
    rng = np.random.default_rng(seed + 1)

    def junk(shape):
        return jnp.asarray(rng.standard_normal(shape), jnp.float32)

    pad_nodes = range(N_REAL, N_REAL + N_PAD)
    dummy = sorted((i, j) for i in pad_nodes for j in pad_nodes)
    padded = (
        jnp.concatenate([node, junk((N_PAD, *node.shape[1:]))]),
        jnp.concatenate([sph, junk((len(dummy), *sph.shape[1:]))]),
        jnp.concatenate([es, junk((len(dummy), *es.shape[1:]))]),
        jnp.concatenate([senders, jnp.asarray([s for s, _ in dummy], jnp.int32)]),
        jnp.concatenate([receivers, jnp.asarray([r for _, r in dummy], jnp.int32)]),
    )
    node_mask = jnp.arange(N_REAL + N_PAD) < N_REAL
    return (node, sph, es, senders, receivers), padded, node_mask


def _marked(padded, node_mask):
    """Padded operands with dummy endpoints replaced by the `DUMMY_INDEX` sentinel."""
    node, sph, es, senders, receivers = padded
    return (node, sph, es, *GraphCSR.mask_edges(senders, receivers, node_mask))


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES)
def test_dummy_edges_forward_leaves_real_nodes_untouched(x_ir, y_ir, o_ir, channels):
    """`DUMMY_INDEX`-marked edges contribute nothing: the real messages match the
    unpadded graph and the padding nodes stay at exactly zero."""
    k = _kernels(x_ir, y_ir, o_ir)
    real, padded, node_mask = _padded_inputs(k.conv, channels)

    got = k.fused(*_marked(padded, node_mask), node_mask)
    expected = k.fused(*real)

    assert got.shape[0] == N_REAL + N_PAD
    np.testing.assert_allclose(
        np.asarray(got[:N_REAL]), np.asarray(expected), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_array_equal(np.asarray(got[N_REAL:]), 0.0)


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES)
def test_dummy_edges_cotangents_vanish(x_ir, y_ir, o_ir, channels):
    """Dummy edges get zero cotangents, and the real ones match the unpadded graph."""
    k = _kernels(x_ir, y_ir, o_ir)
    real, padded, node_mask = _padded_inputs(k.conv, channels)
    n_real_edges = real[1].shape[0]

    got = k.fused_grads(*_marked(padded, node_mask), node_mask)
    expected = k.unfused_grads(*real)

    for name, a, b, n_kept in zip(
        ("d_node_feats", "d_sph", "d_edge_scalars"),
        got,
        expected,
        (N_REAL, n_real_edges, n_real_edges),
    ):
        assert _rel(a[:n_kept], b) < RTOL, f"{name} mismatch on real rows"
        np.testing.assert_array_equal(np.asarray(a[n_kept:]), 0.0, err_msg=name)


@SLOW
@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_dummy_edges_double_backward(x_ir, y_ir, o_ir, channels):
    """Force training over a padded graph matches the unpadded one; the second-order
    cotangents of dummy edges vanish too."""
    k = _kernels(x_ir, y_ir, o_ir)
    real, padded, node_mask = _padded_inputs(k.conv, channels)
    n_real_edges = real[1].shape[0]

    got = k.fused_second_grads(*_marked(padded, node_mask), node_mask)
    expected = k.unfused_second_grads(*real)

    for name, a, b, n_kept in zip(
        ("dd_node", "dd_sph", "dd_edge_scalars"),
        got,
        expected,
        (N_REAL, n_real_edges, n_real_edges),
    ):
        assert _rel(a[:n_kept], b) < RTOL, f"{name} mismatch on real rows"
        np.testing.assert_array_equal(np.asarray(a[n_kept:]), 0.0, err_msg=name)


SPLIT_SENDER = 3
GROUP_START = SPLIT_SENDER * (N_REAL - 1)
PLACEMENTS = [GROUP_START + 3, GROUP_START + (N_REAL - 1)]
PLACEMENT_IDS = ["inside_sender_group", "sender_group_end"]


def _graph_with_one_padding_edge(conv, channels, position, seed: int = 0):
    """Real graph plus one (real sender -> padding node) edge spliced at `position`."""
    node, sph, es, senders, receivers = _contract_valid_inputs(
        conv, N_REAL, channels, seed
    )
    rng = np.random.default_rng(seed + 1)

    def junk(shape):
        return jnp.asarray(rng.standard_normal(shape), jnp.float32)

    def splice(array, row):
        return jnp.concatenate([array[:position], row[None], array[position:]])

    padded = (
        jnp.concatenate([node, junk((N_PAD, *node.shape[1:]))]),
        splice(sph, junk(sph.shape[1:])),
        splice(es, junk(es.shape[1:])),
        splice(senders, jnp.asarray(SPLIT_SENDER, jnp.int32)),
        splice(receivers, jnp.asarray(N_REAL, jnp.int32)),
    )
    assert bool((jnp.diff(padded[3]) >= 0).all()), "sender ordering broken by splice"
    node_mask = jnp.arange(N_REAL + N_PAD) < N_REAL
    return (node, sph, es, senders, receivers), padded, node_mask


@pytest.mark.parametrize("position", PLACEMENTS, ids=PLACEMENT_IDS)
@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_dummy_edge_splitting_a_sender_group(x_ir, y_ir, o_ir, channels, position):
    """A dummy edge anywhere inside a real sender's group must not fragment it: the
    group flush overwrites its node row, so a reopened group loses its first
    partial and one node comes out silently wrong."""
    k = _kernels(x_ir, y_ir, o_ir)
    real, padded, node_mask = _graph_with_one_padding_edge(k.conv, channels, position)

    got = k.fused(*_marked(padded, node_mask), node_mask)
    expected = k.fused(*real)

    assert _rel(got[:N_REAL], expected) < RTOL
    np.testing.assert_array_equal(np.asarray(got[N_REAL:]), 0.0)


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_dummy_edges_under_vmap(x_ir, y_ir, o_ir, channels):
    """Batching merges each graph's dummy tail into the middle of the edge list, so
    the runtime edge bound is dropped there; results must still match per graph."""
    k = _kernels(x_ir, y_ir, o_ir)
    per_graph = [_padded_inputs(k.conv, channels, seed=g) for g in range(2)]
    marked = [_marked(padded, mask) for _, padded, mask in per_graph]
    batched = tuple(jnp.stack([m[i] for m in marked]) for i in range(5))
    node_mask = per_graph[0][2]  # same padding layout for every graph

    def fused(node, sph, es, senders, receivers):
        return k.conv._fused_mosaic_tpu_eval(
            node, sph, es, senders, receivers, node_mask
        )

    got = jax.jit(jax.vmap(fused))(*batched)
    for g, single in enumerate(marked):
        expected = k.fused(*single, node_mask)
        assert _rel(got[g], expected) < RTOL, f"graph {g} mismatch"


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_node_mask_dispatch_matches_unfused(x_ir, y_ir, o_ir, channels):
    """`Convolution.__call__(node_mask=...)` marks the dummy edges for the kernel."""
    kwargs = dict(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    with config.use(convolution=options.Convolution.FUSED_MOSAIC_TPU):
        fused = Convolution(**kwargs)
    with config.use(convolution=options.Convolution.UNFUSED):
        unfused = Convolution(**kwargs)

    _, padded, node_mask = _padded_inputs(fused, channels)

    got = jax.jit(fused.__call__)(*padded, node_mask)
    expected = unfused(*padded, node_mask)
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_all_real_mask_leaves_no_tail(x_ir, y_ir, o_ir, channels):
    """An all-real mask stops the pipeline at the last block, leaving nothing to zero."""
    k = _kernels(x_ir, y_ir, o_ir)
    real = _contract_valid_inputs(k.conv, N_REAL, channels)
    node_mask = jnp.ones(N_REAL, bool)

    masked = k.fused_grads(
        *real[:3], *GraphCSR.mask_edges(*real[3:], node_mask), node_mask
    )
    for name, got, expected in zip(
        ("d_node_feats", "d_sph", "d_edge_scalars"), masked, k.unfused_grads(*real)
    ):
        assert _rel(got, expected) < RTOL, f"{name} mismatch"


def _batch_valid_inputs(conv, n_nodes, channels, n_graphs):
    """Stack `n_graphs` independent contract-valid graphs along a leading axis."""
    per = [
        _contract_valid_inputs(conv, n_nodes, channels, seed=g) for g in range(n_graphs)
    ]
    return tuple(jnp.stack([g[i] for g in per]) for i in range(5)), per


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_vmap_forward_matches_per_graph(x_ir, y_ir, o_ir, channels):
    """vmap over a batch of graphs must shift senders/receivers into each graph's
    own node block; otherwise every graph gathers from graph 0's nodes."""
    k = _kernels(x_ir, y_ir, o_ir)
    batched, per = _batch_valid_inputs(k.conv, n_nodes=8, channels=channels, n_graphs=3)

    got = jax.jit(jax.vmap(k.conv._fused_mosaic_tpu_eval))(*batched)
    expected = jnp.stack([k.fused(*g) for g in per])
    assert _rel(got, expected) < RTOL


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_vmap_backward_matches_per_graph(x_ir, y_ir, o_ir, channels):
    """Gradients under vmap must also see per-graph-shifted edge indices."""
    k = _kernels(x_ir, y_ir, o_ir)
    batched, per = _batch_valid_inputs(k.conv, n_nodes=8, channels=channels, n_graphs=3)

    got = jax.jit(jax.vmap(_sum_sq_grads(k.conv._fused_mosaic_tpu_eval)))(*batched)
    per_graph = [k.unfused_grads(*g) for g in per]
    expected = tuple(jnp.stack([g[i] for g in per_graph]) for i in range(3))
    for name, a, b in zip(("d_node_feats", "d_sph", "d_edge_scalars"), got, expected):
        assert _rel(a, b) < RTOL, f"{name} mismatch"


def _device_mesh(axis: str = "batch") -> Mesh:
    """Real mesh over every available device, or skip if fewer than two.

    The dp-vmap path only shards (non-trivial shard_map) when a mesh with axis
    names is active, and only a >=2 device mesh actually splits the merged batch
    across shards - so a single-device run can't exercise the cross-shard logic.
    """
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(f"multidevice tests need >= 2 devices, have {len(devices)}")
    return Mesh(np.asarray(devices), (axis,))


@pytest.mark.multi_devices
@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_vmap_multidevice_forward_matches_per_graph(x_ir, y_ir, o_ir, channels):
    """vmap under a device mesh (shard_map data-parallel path) must give the same
    forward as the per-graph loop. Each shard sees only its slice of the merged
    node block, so per-graph edge offsets must be shard-local, not global."""
    k = _kernels(x_ir, y_ir, o_ir)
    mesh = _device_mesh()
    # one graph per shard: the merged (V*N) node axis splits graph-aligned.
    batched, per = _batch_valid_inputs(
        k.conv, n_nodes=8, channels=channels, n_graphs=mesh.size
    )

    expected = jnp.stack([k.fused(*g) for g in per])
    with jax.sharding.set_mesh(mesh):
        got = jax.jit(jax.vmap(k.conv._fused_mosaic_tpu_eval))(*batched)
    assert _rel(got, expected) < RTOL


@pytest.mark.multi_devices
@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_vmap_multidevice_backward_matches_per_graph(x_ir, y_ir, o_ir, channels):
    """Gradients through the sharded dp-vmap path must match the per-graph loop."""
    k = _kernels(x_ir, y_ir, o_ir)
    mesh = _device_mesh()
    batched, per = _batch_valid_inputs(
        k.conv, n_nodes=8, channels=channels, n_graphs=mesh.size
    )

    per_graph = [k.unfused_grads(*g) for g in per]
    expected = tuple(jnp.stack([g[i] for g in per_graph]) for i in range(3))
    with jax.sharding.set_mesh(mesh):
        got = jax.jit(jax.vmap(_sum_sq_grads(k.conv._fused_mosaic_tpu_eval)))(*batched)
    for name, a, b in zip(("d_node_feats", "d_sph", "d_edge_scalars"), got, expected):
        assert _rel(a, b) < RTOL, f"{name} mismatch"


@pytest.mark.multi_devices
@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES)
def test_vmap_multidevice_masked_matches_per_graph(x_ir, y_ir, o_ir, channels):
    """A padded batch under a device mesh: every shard builds its own block order,
    so the walked blocks must still cover exactly its graphs' real edges."""
    k = _kernels(x_ir, y_ir, o_ir)
    mesh = _device_mesh()
    per_graph = [_padded_inputs(k.conv, channels, seed=g)[1:] for g in range(mesh.size)]
    marked = [_marked(padded, node_mask) for padded, node_mask in per_graph]
    batched = tuple(jnp.stack([m[i] for m in marked]) for i in range(5))
    node_mask = per_graph[0][1]  # same padding layout for every graph

    def fused(node, sph, es, senders, receivers):
        return k.conv._fused_mosaic_tpu_eval(
            node, sph, es, senders, receivers, node_mask
        )

    expected = jnp.stack([k.fused(*single, node_mask) for single in marked])
    with jax.sharding.set_mesh(mesh):
        got = jax.jit(jax.vmap(fused))(*batched)
    assert _rel(got, expected) < RTOL


def _assertion_conv() -> Convolution:
    return Convolution(
        source=(O3Space("0e + 1o + 2e"), O3Space("0e + 1o + 2e")),
        target=O3Space("0e + 1o + 2e"),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )


@pytest.mark.multi_devices
def test_vmap_multidevice_indivisible_batch_raises():
    conv = _assertion_conv()
    fused = conv._fused_mosaic_tpu_eval
    mesh = _device_mesh()
    # mesh.size + 1 is never divisible by mesh.size (>= 2): a shard can't hold
    # a whole number of graphs.
    batched, _ = _batch_valid_inputs(
        conv, n_nodes=8, channels=128, n_graphs=mesh.size + 1
    )

    with jax.sharding.set_mesh(mesh):
        with pytest.raises(AssertionError, match="not divisible"):
            jax.jit(jax.vmap(fused))(*batched)


@SLOW
@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES[:1])
def test_end_to_end_position_gradient_matches(x_ir, y_ir, o_ir, channels):
    """The gradient w.r.t. atom positions (i.e. forces) matches the unfused"""
    k = _kernels(x_ir, y_ir, o_ir)
    conv = k.conv
    n_nodes = 8
    senders, receivers = _symmetric_edges(n_nodes)
    par = _y_parity_signs(conv._otimes.source[1])
    lm_sph = conv._otimes.source[1].dim
    num_irreps = conv._mix.num_irreps
    rng = np.random.default_rng(0)
    even = jnp.asarray(np.where(par > 0)[0])
    odd = jnp.asarray(np.where(par < 0)[0])

    A_even = jnp.asarray(rng.standard_normal((even.shape[0], 3)), jnp.float32)
    W_odd = jnp.asarray(rng.standard_normal((odd.shape[0], 3)), jnp.float32)
    Wr1 = jnp.asarray(rng.standard_normal((1, 8)) * 0.5, jnp.float32)
    Wr2 = jnp.asarray(
        rng.standard_normal((8, num_irreps * channels)) * 0.3, jnp.float32
    )
    positions = jnp.asarray(rng.standard_normal((n_nodes, 3)), jnp.float32)
    node = jnp.asarray(
        rng.standard_normal((n_nodes, conv._otimes.source[0].dim, channels)),
        jnp.float32,
    )

    def Y(p):  # odd-graded: even-l comps even in r, odd-l comps linear in r
        r = p[receivers] - p[senders]
        sph = jnp.zeros((senders.shape[0], lm_sph), jnp.float32)
        sph = sph.at[:, even].set((r * r) @ A_even.T)
        return sph.at[:, odd].set(r @ W_odd.T)

    def radial(p):  # even: depends only on |r|
        d = jnp.linalg.norm(p[receivers] - p[senders], axis=1, keepdims=True)
        return (jnp.tanh(d @ Wr1) @ Wr2).reshape(senders.shape[0], num_irreps, channels)

    sph, vjp_Y = jax.vjp(Y, positions)
    es, vjp_radial = jax.vjp(radial, positions)

    def d_pos_from_cotangents(conv, conv_vjp):
        ct = 2 * conv(node, sph, es, senders, receivers)  # d sum(out**2) / d out
        _, d_sph, d_es = conv_vjp(node, sph, es, senders, receivers, ct)
        return vjp_Y(d_sph)[0] + vjp_radial(d_es)[0]

    d_pos_fused = d_pos_from_cotangents(k.fused, k.fused_vjp)
    d_pos_unfused = d_pos_from_cotangents(k.unfused, k.unfused_vjp)
    assert _rel(d_pos_fused, d_pos_unfused) < 1e-3
    fused = conv._fused_mosaic_tpu_eval
    autodiff = jax.jit(
        jax.grad(
            lambda p: jnp.sum(fused(node, Y(p), radial(p), senders, receivers) ** 2)
        )
    )(positions)
    assert _rel(d_pos_fused, autodiff) < 1e-3


@SLOW
@pytest.mark.parametrize("channel_block", [256, 128])
def test_backward_channel_strides_match_unfused(channel_block):
    """The bwd may walk the edges once per channel slice; every split must agree."""
    irreps, channels = "0e + 1o + 2e", 256
    conv = Convolution(
        source=(O3Space(irreps), O3Space(irreps)),
        target=O3Space(irreps),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    node, sph, es, senders, receivers = _contract_valid_inputs(
        conv, n_nodes=8, channels=channels
    )
    coef = conv._otimes.coef
    params = PallasMosaicTPUMessagePassingConvolutionParams(
        indices=np.array(coef.indices),
        values=np.array(coef.data),
        layout=options.Layout.TRAILING_CHANNELS,
        x_space=O3Space(str(conv._otimes.source[0])),
        y_space=O3Space(str(conv._otimes.source[1])),
        z_space=O3Space(str(conv._otimes.target)),
        fwd_config=PallasMosaicTPUMessagePassingConvolutionFwdConfig(
            assume_receiver_sorted=True
        ),
        bwd_config=PallasMosaicTPUMessagePassingConvolutionBwdConfig(
            assume_sender_sorted=True, channel_block=channel_block
        ),
    )

    def strided(n, s, e):
        return convolution_mosaic_tpu(n, s, e, senders, receivers, params)

    gradients = jax.jit(
        jax.grad(lambda n, s, e: jnp.sum(strided(n, s, e) ** 2), (0, 1, 2))
    )(node, sph, es)
    expected = jax.jit(_sum_sq_grads(conv._unfused_eval))(
        node, sph, es, senders, receivers
    )
    for name, got, want in zip(
        ("d_node_feats", "d_sph", "d_edge_scalars"), gradients, expected
    ):
        assert (
            _rel(got, want) < RTOL
        ), f"{name} mismatch at channel_block={channel_block}"
