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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from e3j.core.convolution import Convolution
from e3j.spaces import O3Space
from e3j.utils import config, options
from e3j.utils.options import Layout

pytestmark = pytest.mark.mosaic_tpu

RTOL = 1e-3
ATOL = 1e-3

CASES = [
    # dense and packed
    ("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", 128),
    ("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", 64),
    # output irreps with multiplicity > 1 (dense + packed)
    ("0e + 2x1o", "0e + 1o", "0e + 1o + 2e", 128),
    ("0e + 2x1o", "0e + 1o", "0e + 1o + 2e", 32),
    # l_max=3 (dense + packed)
    ("0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", 128),
    ("0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", 32),
]
CASE_IDS = [f"{x}__channels{m}" for x, _, _, m in CASES]

# The per-edge gradient relationships are checked on fewer cases for speed.
BWD_CASES = [
    ("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", 128),
    ("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", 32),
]
BWD_CASE_IDS = [f"{x}__channels{m}" for x, _, _, m in BWD_CASES]


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


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES, ids=CASE_IDS)
def test_fused_matches_unfused(x_ir, y_ir, o_ir, channels):
    conv = Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    node, sph, es, senders, receivers = _contract_valid_inputs(
        conv, n_nodes=16, channels=channels
    )

    expected = conv._unfused_eval(node, sph, es, senders, receivers)
    got = jax.jit(conv._fused_mosaic_tpu_eval)(node, sph, es, senders, receivers)

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES, ids=CASE_IDS)
def test_backward_matches_unfused(x_ir, y_ir, o_ir, channels):
    conv = Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    node, sph, es, senders, receivers = _contract_valid_inputs(
        conv, n_nodes=8, channels=channels
    )
    fused, unfused = conv._fused_mosaic_tpu_eval, conv._unfused_eval

    # forward matches
    np.testing.assert_allclose(
        np.asarray(fused(node, sph, es, senders, receivers)),
        np.asarray(unfused(node, sph, es, senders, receivers)),
        rtol=RTOL,
        atol=ATOL,
    )
    # all three gradients match the unfused reference directly (natural frame)
    gf = jax.grad(
        lambda n, s, e: jnp.sum(fused(n, s, e, senders, receivers) ** 2), (0, 1, 2)
    )(node, sph, es)
    gu = jax.grad(
        lambda n, s, e: jnp.sum(unfused(n, s, e, senders, receivers) ** 2), (0, 1, 2)
    )(node, sph, es)
    for name, a, b in zip(("d_node_feats", "d_sph", "d_edge_scalars"), gf, gu):
        assert _rel(a, b) < RTOL, f"{name} mismatch"


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES, ids=CASE_IDS)
def test_backward_cotangents_match(x_ir, y_ir, o_ir, channels):
    conv = Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    node, sph, es, senders, receivers = _contract_valid_inputs(
        conv, n_nodes=8, channels=channels
    )
    fused, unfused = conv._fused_mosaic_tpu_eval, conv._unfused_eval

    out, vjp_unfused = jax.vjp(
        lambda n, s, e: unfused(n, s, e, senders, receivers), node, sph, es
    )
    _, vjp_fused = jax.vjp(
        lambda n, s, e: fused(n, s, e, senders, receivers), node, sph, es
    )
    # arbitrary output cotangent, uncorrelated with the output
    ct = jnp.asarray(np.random.default_rng(1).standard_normal(out.shape), jnp.float32)
    df = vjp_fused(ct)
    du = vjp_unfused(ct)
    for name, a, b in zip(("d_node_feats", "d_sph", "d_edge_scalars"), df, du):
        assert _rel(a, b) < RTOL, f"{name} mismatch"


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES, ids=BWD_CASE_IDS)
def test_force_training_double_backward(x_ir, y_ir, o_ir, channels):
    conv = Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    node, sph, es, senders, receivers = _contract_valid_inputs(
        conv, n_nodes=8, channels=channels
    )

    def make_loss(conv_fn):
        def loss(n, s, e):
            # first-order gradient (forces-like), then a scalar of it
            g = jax.grad(
                lambda nn: jnp.sum(conv_fn(nn, s, e, senders, receivers) ** 2)
            )(n)
            return jnp.sum(g**2)

        return loss

    d2f = jax.grad(make_loss(conv._fused_mosaic_tpu_eval), (0, 1, 2))(node, sph, es)
    d2u = jax.grad(make_loss(conv._unfused_eval), (0, 1, 2))(node, sph, es)
    for name, a, b in zip(("dd_node", "dd_sph", "dd_edge_scalars"), d2f, d2u):
        assert _rel(a, b) < RTOL, f"{name} mismatch"


def _batch_valid_inputs(conv, n_nodes, channels, n_graphs):
    """Stack `n_graphs` independent contract-valid graphs along a leading axis."""
    per = [
        _contract_valid_inputs(conv, n_nodes, channels, seed=g) for g in range(n_graphs)
    ]
    return tuple(jnp.stack([g[i] for g in per]) for i in range(5)), per


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES, ids=BWD_CASE_IDS)
def test_vmap_forward_matches_per_graph(x_ir, y_ir, o_ir, channels):
    """vmap over a batch of graphs must shift senders/receivers into each graph's
    own node block; otherwise every graph gathers from graph 0's nodes."""
    conv = Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    fused = conv._fused_mosaic_tpu_eval
    batched, per = _batch_valid_inputs(conv, n_nodes=8, channels=channels, n_graphs=3)

    got = jax.vmap(fused)(*batched)
    expected = jnp.stack([fused(*g) for g in per])
    assert _rel(got, expected) < RTOL


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES, ids=BWD_CASE_IDS)
def test_vmap_backward_matches_per_graph(x_ir, y_ir, o_ir, channels):
    """Gradients under vmap must also see per-graph-shifted edge indices."""
    conv = Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    fused = conv._fused_mosaic_tpu_eval
    batched, per = _batch_valid_inputs(conv, n_nodes=8, channels=channels, n_graphs=3)

    def grads(node, sph, es, senders, receivers):
        return jax.grad(
            lambda n, s, e: jnp.sum(fused(n, s, e, senders, receivers) ** 2), (0, 1, 2)
        )(node, sph, es)

    got = jax.vmap(grads)(*batched)
    expected = tuple(jnp.stack([grads(*g)[i] for g in per]) for i in range(3))
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


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES, ids=BWD_CASE_IDS)
def test_vmap_multidevice_forward_matches_per_graph(x_ir, y_ir, o_ir, channels):
    """vmap under a device mesh (shard_map data-parallel path) must give the same
    forward as the per-graph loop. Each shard sees only its slice of the merged
    node block, so per-graph edge offsets must be shard-local, not global."""
    conv = Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    fused = conv._fused_mosaic_tpu_eval
    mesh = _device_mesh()
    # one graph per shard: the merged (V*N) node axis splits graph-aligned.
    batched, per = _batch_valid_inputs(
        conv, n_nodes=8, channels=channels, n_graphs=mesh.size
    )

    expected = jnp.stack([fused(*g) for g in per])
    with jax.sharding.set_mesh(mesh):
        got = jax.jit(jax.vmap(fused))(*batched)
    assert _rel(got, expected) < RTOL


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", BWD_CASES, ids=BWD_CASE_IDS)
def test_vmap_multidevice_backward_matches_per_graph(x_ir, y_ir, o_ir, channels):
    """Gradients through the sharded dp-vmap path must match the per-graph loop."""
    conv = Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
    fused = conv._fused_mosaic_tpu_eval
    mesh = _device_mesh()
    batched, per = _batch_valid_inputs(
        conv, n_nodes=8, channels=channels, n_graphs=mesh.size
    )

    def grads(node, sph, es, senders, receivers):
        return jax.grad(
            lambda n, s, e: jnp.sum(fused(n, s, e, senders, receivers) ** 2), (0, 1, 2)
        )(node, sph, es)

    expected = tuple(jnp.stack([grads(*g)[i] for g in per]) for i in range(3))
    with jax.sharding.set_mesh(mesh):
        got = jax.jit(jax.vmap(grads))(*batched)
    for name, a, b in zip(("d_node_feats", "d_sph", "d_edge_scalars"), got, expected):
        assert _rel(a, b) < RTOL, f"{name} mismatch"


def _assertion_conv() -> Convolution:
    return Convolution(
        source=(O3Space("0e + 1o + 2e"), O3Space("0e + 1o + 2e")),
        target=O3Space("0e + 1o + 2e"),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )


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


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES, ids=CASE_IDS)
def test_end_to_end_position_gradient_matches(x_ir, y_ir, o_ir, channels):
    """The gradient w.r.t. atom positions (i.e. forces) matches the unfused"""
    conv = Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        avg_num_neighbors=None,
        normalization="SQRT_DIM_OUT",
        graph_ordering="SENDER",
    )
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

    fused, unfused = conv._fused_mosaic_tpu_eval, conv._unfused_eval
    sph, vjp_Y = jax.vjp(Y, positions)
    es, vjp_radial = jax.vjp(radial, positions)

    def d_pos_from_cotangents(conv_fn):
        d_sph, d_es = jax.grad(
            lambda s, e: jnp.sum(conv_fn(node, s, e, senders, receivers) ** 2), (0, 1)
        )(sph, es)
        return vjp_Y(d_sph)[0] + vjp_radial(d_es)[0]

    d_pos_fused = d_pos_from_cotangents(fused)
    d_pos_unfused = d_pos_from_cotangents(unfused)
    assert _rel(d_pos_fused, d_pos_unfused) < 1e-3
    autodiff = jax.grad(
        lambda p: jnp.sum(fused(node, Y(p), radial(p), senders, receivers) ** 2)
    )(positions)
    assert _rel(d_pos_fused, autodiff) < 1e-3
