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

"""The fused Mosaic GPU convolution must match the unfused reference.

The kernel maps one CTA onto one node and walks that node's edges, reading the
edge list as a CSR of whichever endpoint `graph_ordering` names -- so that
endpoint has to be sorted.

Under `RECEIVER` a CTA aggregates its node's incoming edges directly and nothing
else is assumed of the graph. Under `SENDER` it walks the node's *outgoing*
edges and folds the `y` parity into the coefficients, which recovers the same
received message only on a symmetric graph whose edge features are graded-odd
and whose edge scalars are even under edge reversal -- the Mosaic TPU contract.
Both orderings must produce the same values *and* the same per-edge cotangents
as the unfused reference; the inputs built below satisfy the `SENDER` contract.

The kernel needs a Hopper (sm_90) or newer CUDA device to run, so the whole
module skips elsewhere -- there is no interpret-mode tier, since Pallas
interpret mode has no discharge rule for Mosaic GPU's async copies.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from e3j.core.convolution import Convolution
from e3j.spaces import O3Space
from e3j.utils import config, options
from e3j.utils.options import Layout

pytestmark = pytest.mark.mosaic_gpu

RTOL = 1e-4
ATOL = 1e-4

CASES = [
    ("0e + 1o", "0e + 1o", "0e + 1o + 2e", 128),
    ("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", 128),
    # l_max=3, the size the kernel is tuned for
    ("0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", 128),
    # TODO(armand): fix layout so we can enable this test
    # # input irreps with multiplicity > 1, and a wider channel axis
    # ("0e + 2x1o", "0e + 1o", "0e + 1o + 2e", 256),
    # multiplicity > 1 on the *edge* space, so a parity sign covers several
    # copies of one irrep -- the `mul` factor of `y_parity_signs`
    ("0e + 1o", "0e + 2x1o", "0e + 1o + 2e", 128),
]
CASE_IDS = [f"{x}__{y}__{o}__channels{c}" for x, y, o, c in CASES]


@pytest.fixture(autouse=True)
def _require_hopper():
    """Skip (rather than fail) when there is no Hopper-or-newer CUDA device."""
    if jax.default_backend() != "gpu":
        pytest.skip("mosaic_gpu convolution tests require a CUDA backend")
    capability = getattr(jax.devices()[0], "compute_capability", None)
    if capability is None or tuple(int(p) for p in capability.split(".")) < (9, 0):
        pytest.skip(f"kernel targets sm_90+, device is sm_{capability}")


@pytest.fixture(autouse=True)
def _plain_jax_reference():
    """Keep the reference path on plain JAX.

    On a CUDA device the default `convolution` is `FUSED_CUDA`, whose op cannot
    take `grad(jit(f))` with traced edge indices, so a `Convolution` built
    outside a `FUSED_MOSAIC_GPU` scope must not pick it up as the reference.
    """
    with config.use(
        tensor_product=options.TensorProduct.UNFUSED,
        convolution=options.Convolution.UNFUSED,
    ):
        yield


def build_conv(
    x_ir: str,
    y_ir: str,
    o_ir: str,
    ordering: options.GraphOrdering = options.GraphOrdering.RECEIVER,
) -> Convolution:
    return Convolution(
        source=(O3Space(x_ir), O3Space(y_ir)),
        target=O3Space(o_ir),
        layout=Layout.TRAILING_CHANNELS,
        graph_ordering=ordering,
    )


def make_inputs(
    conv: Convolution,
    n_nodes: int,
    channels: int,
    degrees: np.ndarray | None = None,
    seed: int = 0,
):
    """Random operands on a receiver-sorted graph with the given in-degrees.

    `degrees` defaults to 3 incoming edges per node; pass an array to exercise
    variable degree, including nodes with none.
    """
    rng = np.random.default_rng(seed)
    if degrees is None:
        degrees = np.full(n_nodes, 3, np.int32)
    receivers = np.repeat(np.arange(n_nodes, dtype=np.int32), degrees)
    n_edges = int(receivers.size)
    senders = rng.integers(0, n_nodes, n_edges, dtype=np.int32)
    x_dim, y_dim = (src.dim for src in conv._otimes.source)
    return (
        jnp.asarray(rng.standard_normal((n_nodes, x_dim, channels), np.float32)),
        jnp.asarray(rng.standard_normal((n_edges, y_dim), np.float32)),
        jnp.asarray(
            rng.standard_normal((n_edges, conv._mix.num_irreps, channels), np.float32)
        ),
        jnp.asarray(senders),
        jnp.asarray(receivers),
    )


def y_parity_signs(y_space: O3Space) -> np.ndarray:
    """Parity `ir.p` per coordinate of the edge space.

    Deliberately a second, independent copy of the kernel's own
    `params.y_parity_signs`: importing that one would make these tests agree with
    whatever convention the kernel picked rather than check it.
    """
    return np.concatenate(
        [np.full(mul * ir.dim, float(ir.p), np.float32) for mul, ir in y_space]
    )


def symmetric_pairs(n_nodes: int, rng, density: float = 0.6) -> list[tuple[int, int]]:
    """Random unordered node pairs; a node may end up with no edge at all."""
    return [
        (i, j)
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
        if rng.random() < density
    ]


def make_symmetric_inputs(
    conv: Convolution,
    n_nodes: int,
    channels: int,
    pairs: list[tuple[int, int]] | None = None,
    order: str = "sender",
    seed: int = 0,
):
    """Random operands on a symmetric graph, in the requested edge order.

    Every unordered pair contributes both directed edges. `edge_features` pick up
    the `y` parity on the reversed direction and `edge_scalars` are shared by
    both -- the contract `SENDER` ordering rests on. `pairs` defaults to the
    complete graph; `order` is `"sender"` or `"receiver"` and only changes how
    the same edge set is sorted.
    """
    rng = np.random.default_rng(seed)
    if pairs is None:
        pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    # Keyed by the ordered pair below, so normalize first: `(j, i)` and `(i, j)`
    # must not become two independent unordered pairs.
    pairs = sorted({(min(p), max(p)) for p in pairs})
    x_dim, y_dim = (src.dim for src in conv._otimes.source)
    n_irreps = conv._mix.num_irreps
    signs = y_parity_signs(conv._otimes.source[1])

    # One base value per unordered pair, mirrored onto its two directed edges.
    base_y = {p: rng.standard_normal(y_dim).astype(np.float32) for p in pairs}
    base_s = {
        p: rng.standard_normal((n_irreps, channels)).astype(np.float32) for p in pairs
    }
    directed = [(i, j) for i, j in pairs] + [(j, i) for i, j in pairs]
    key = (lambda e: e) if order == "sender" else (lambda e: e[::-1])
    directed.sort(key=key)

    edge_features = np.stack(
        [base_y[(i, j)] if i < j else signs * base_y[(j, i)] for i, j in directed]
    )
    edge_scalars = np.stack([base_s[(min(e), max(e))] for e in directed])
    node_features = rng.standard_normal((n_nodes, x_dim, channels), np.float32)
    return (
        jnp.asarray(node_features),
        jnp.asarray(edge_features),
        jnp.asarray(edge_scalars),
        jnp.asarray(np.array([s for s, _ in directed], np.int32)),
        jnp.asarray(np.array([r for _, r in directed], np.int32)),
    )


def relative_error(got, expected) -> float:
    got, expected = np.asarray(got), np.asarray(expected)
    return float(np.abs(got - expected).max() / (np.abs(expected).max() + 1e-12))


# --------------------------------------------------------------- numerics ---


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES, ids=CASE_IDS)
def test_forward_matches_unfused(x_ir, y_ir, o_ir, channels):
    conv = build_conv(x_ir, y_ir, o_ir)
    node, edge, scalars, senders, receivers = make_inputs(conv, 16, channels)

    expected = conv._unfused_eval(node, edge, scalars, senders, receivers)
    got = jax.jit(conv._fused_mosaic_gpu_eval)(node, edge, scalars, senders, receivers)

    assert got.shape == expected.shape
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES, ids=CASE_IDS)
def test_cotangents_match_unfused(x_ir, y_ir, o_ir, channels):
    """All three cotangents, against an output cotangent uncorrelated with it."""
    conv = build_conv(x_ir, y_ir, o_ir)
    node, edge, scalars, senders, receivers = make_inputs(conv, 12, channels)

    def fused(n, e, s):
        return conv._fused_mosaic_gpu_eval(n, e, s, senders, receivers)

    def unfused(n, e, s):
        return conv._unfused_eval(n, e, s, senders, receivers)

    out, pullback_unfused = jax.vjp(unfused, node, edge, scalars)
    _, pullback_fused = jax.vjp(fused, node, edge, scalars)
    cotangent = jnp.asarray(
        np.random.default_rng(1).standard_normal(out.shape, np.float32)
    )
    got = jax.jit(pullback_fused)(cotangent)
    expected = jax.jit(pullback_unfused)(cotangent)

    for name, a, b in zip(("d_node", "d_edge", "d_scalars"), got, expected):
        assert a.shape == b.shape, f"{name} shape"
        assert relative_error(a, b) < RTOL, f"{name} mismatch"


def test_variable_degree_and_isolated_nodes():
    """Nodes with no incoming edge get zeros; degrees need not be uniform.

    The forward's accumulator is only zeroed, never seeded, precisely so that a
    receiver whose edge range is empty still produces a written, zero row.
    """
    conv = build_conv("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e")
    degrees = np.array([0, 5, 1, 0, 2, 7, 0, 3], np.int32)
    node, edge, scalars, senders, receivers = make_inputs(
        conv, degrees.size, 128, degrees=degrees, seed=3
    )

    expected = conv._unfused_eval(node, edge, scalars, senders, receivers)
    got = jax.jit(conv._fused_mosaic_gpu_eval)(node, edge, scalars, senders, receivers)

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=RTOL, atol=ATOL
    )
    assert not np.any(np.asarray(got)[degrees == 0]), "isolated nodes must be zero"


def test_forward_is_independent_of_pipeline_depth():
    """The forward's SMEM ring must give the same answer at every depth.

    `pipeline_depth` sets the slots the memory and compute warpgroups pass
    between them, so it is what the produced/consumed barrier protocol is keyed
    on: depth 1 leaves no overlap at all, a depth over the node's degree never
    reaches the refill loop, and the depths in between wrap the ring mid-walk.
    All three have to agree with the unfused reference, and a slot reused a step
    early would show up here as a wrong -- or hung -- run.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu import (
        PallasMosaicGPUMessagePassingConvolutionFwdConfig,
        PallasMosaicGPUMessagePassingConvolutionParams,
    )
    from e3j.pallas_ops.convolution.mosaic_gpu.fwd import (
        _convolution_kernel_mosaic_gpu_fwd,
    )

    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e")
    degrees = np.array([0, 1, 5, 2, 0, 9], np.int32)
    node, edge, scalars, senders, receivers = make_inputs(
        conv, degrees.size, 128, degrees=degrees, seed=7
    )
    expected = conv._unfused_eval(node, edge, scalars, senders, receivers)
    coef = conv._otimes.coef

    for depth in (1, 2, 3, 16):
        params = PallasMosaicGPUMessagePassingConvolutionParams(
            indices=np.asarray(coef.indices),
            values=np.asarray(coef.data),
            layout=Layout.TRAILING_CHANNELS,
            x_space=conv._otimes.source[0],
            y_space=conv._otimes.source[1],
            z_space=conv._otimes.target,
            graph_ordering=conv.graph_ordering,
            fwd_config=PallasMosaicGPUMessagePassingConvolutionFwdConfig(
                pipeline_depth=depth
            ),
        )
        got = jax.jit(lambda *a, p=params: _convolution_kernel_mosaic_gpu_fwd(*a, p))(
            node, edge, scalars, senders, receivers
        )
        np.testing.assert_allclose(
            np.asarray(got),
            np.asarray(expected),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{depth=}",
        )


def test_repeated_senders_accumulate_in_backward():
    """`dx`'s scatter is a cross-CTA accumulation, so a hot sender must sum.

    Every edge here has the same sender, which makes `dx[0]` the sum of all the
    per-edge contributions the CTAs reduce into it.
    """
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e")
    node, edge, scalars, _, receivers = make_inputs(conv, 8, 128, seed=4)
    senders = jnp.zeros_like(receivers)

    def fused(n, e, s):
        return conv._fused_mosaic_gpu_eval(n, e, s, senders, receivers)

    def unfused(n, e, s):
        return conv._unfused_eval(n, e, s, senders, receivers)

    cotangent = jnp.asarray(
        np.random.default_rng(5).standard_normal((8, conv.target.dim, 128), np.float32)
    )
    got = jax.jit(lambda *a: jax.vjp(fused, *a)[1](cotangent))(node, edge, scalars)
    expected = jax.jit(lambda *a: jax.vjp(unfused, *a)[1](cotangent))(
        node, edge, scalars
    )

    assert relative_error(got[0], expected[0]) < RTOL
    assert not np.any(np.asarray(got[0])[1:]), "only node 0 sends"


# --------------------------------------------------------- sender-sorted ---


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES, ids=CASE_IDS)
def test_sender_forward_matches_unfused(x_ir, y_ir, o_ir, channels):
    """Walking a node's outgoing edges must reproduce its incoming messages.

    The kernel never sees the reversed edge: it reads the forward edge's `y` and
    relies on the parity-signed coefficients to turn it into the reverse's.
    """
    conv = build_conv(x_ir, y_ir, o_ir, options.GraphOrdering.SENDER)
    node, edge, scalars, senders, receivers = make_symmetric_inputs(conv, 8, channels)

    expected = conv._unfused_eval(node, edge, scalars, senders, receivers)
    got = jax.jit(conv._fused_mosaic_gpu_eval)(node, edge, scalars, senders, receivers)

    assert got.shape == expected.shape
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES, ids=CASE_IDS)
def test_sender_cotangents_match_unfused(x_ir, y_ir, o_ir, channels):
    """The backward is the plain VJP: per-edge cotangents in the natural frame.

    Only the forward trades on the graph symmetry. The sender-grouped backward
    keeps the coefficients unsigned and gathers the cotangent by receiver, so
    `dy[e]` and `ds[e]` belong to edge `e` itself, not to its reverse.
    """
    conv = build_conv(x_ir, y_ir, o_ir, options.GraphOrdering.SENDER)
    node, edge, scalars, senders, receivers = make_symmetric_inputs(conv, 8, channels)

    def fused(n, e, s):
        return conv._fused_mosaic_gpu_eval(n, e, s, senders, receivers)

    def unfused(n, e, s):
        return conv._unfused_eval(n, e, s, senders, receivers)

    out, pullback_unfused = jax.vjp(unfused, node, edge, scalars)
    _, pullback_fused = jax.vjp(fused, node, edge, scalars)
    cotangent = jnp.asarray(
        np.random.default_rng(1).standard_normal(out.shape, np.float32)
    )
    got = jax.jit(pullback_fused)(cotangent)
    expected = jax.jit(pullback_unfused)(cotangent)

    for name, a, b in zip(("d_node", "d_edge", "d_scalars"), got, expected):
        assert a.shape == b.shape, f"{name} shape"
        assert relative_error(a, b) < RTOL, f"{name} mismatch"


def test_sender_variable_degree_and_isolated_nodes():
    """Degrees need not be uniform, and a node with no edge gets zeros.

    `dx` is CTA-local under this ordering: a node that sends nothing runs no
    loop iteration, so its row comes straight from the zeroed accumulator -- and
    it has to be written, the buffer not being zero-initialized.
    """
    conv = build_conv(
        "0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", options.GraphOrdering.SENDER
    )
    rng = np.random.default_rng(3)
    pairs = symmetric_pairs(8, rng, density=0.35)
    isolated = sorted(set(range(8)) - {n for pair in pairs for n in pair})
    assert isolated, "the fixture is meant to leave at least one node unconnected"

    node, edge, scalars, senders, receivers = make_symmetric_inputs(
        conv, 8, 128, pairs=pairs, seed=3
    )
    expected = conv._unfused_eval(node, edge, scalars, senders, receivers)
    got = jax.jit(conv._fused_mosaic_gpu_eval)(node, edge, scalars, senders, receivers)

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=RTOL, atol=ATOL
    )
    assert not np.any(np.asarray(got)[isolated]), "isolated nodes must be zero"

    cotangent = jnp.asarray(
        np.random.default_rng(7).standard_normal(expected.shape, np.float32)
    )
    d_node = jax.jit(
        lambda *a: jax.vjp(
            lambda n, e, s: conv._fused_mosaic_gpu_eval(n, e, s, senders, receivers),
            *a,
        )[1](cotangent)
    )(node, edge, scalars)[0]
    assert not np.any(np.asarray(d_node)[isolated]), "isolated nodes get no dx"


def test_sender_hub_node_accumulates_dx_in_registers():
    """A high out-degree node must sum every edge into its carried accumulator.

    The mirror of `test_repeated_senders_accumulate_in_backward`: there `dx` is a
    cross-CTA reduction, here it is one CTA's `fori_loop` carry, so the long
    group is what exercises it. A star graph gives node 0 every edge.
    """
    conv = build_conv(
        "0e + 1o", "0e + 1o", "0e + 1o + 2e", options.GraphOrdering.SENDER
    )
    n_nodes = 12
    pairs = [(0, j) for j in range(1, n_nodes)]
    node, edge, scalars, senders, receivers = make_symmetric_inputs(
        conv, n_nodes, 128, pairs=pairs, seed=9
    )
    assert int(np.sum(np.asarray(senders) == 0)) == n_nodes - 1

    def fused(n, e, s):
        return conv._fused_mosaic_gpu_eval(n, e, s, senders, receivers)

    def unfused(n, e, s):
        return conv._unfused_eval(n, e, s, senders, receivers)

    out, pullback_unfused = jax.vjp(unfused, node, edge, scalars)
    _, pullback_fused = jax.vjp(fused, node, edge, scalars)
    cotangent = jnp.asarray(
        np.random.default_rng(10).standard_normal(out.shape, np.float32)
    )
    got = jax.jit(pullback_fused)(cotangent)
    expected = jax.jit(pullback_unfused)(cotangent)

    for name, a, b in zip(("d_node", "d_edge", "d_scalars"), got, expected):
        assert relative_error(a, b) < RTOL, f"{name} mismatch"


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES[:2], ids=CASE_IDS[:2])
def test_orderings_agree_on_the_same_graph(x_ir, y_ir, o_ir, channels):
    """One symmetric graph, sorted two ways, must give the same answer.

    This is the sharpest check on the `SENDER` path: it pins the parity signing
    and the endpoint swap against the `RECEIVER` kernel rather than against the
    reference, and it also pins the *edge order* of the per-edge cotangents,
    which a permuted `dy` / `ds` would break while still matching the reference
    node-wise.
    """
    sender_conv = build_conv(x_ir, y_ir, o_ir, options.GraphOrdering.SENDER)
    receiver_conv = build_conv(x_ir, y_ir, o_ir, options.GraphOrdering.RECEIVER)
    pairs = symmetric_pairs(8, np.random.default_rng(11))
    by_sender = make_symmetric_inputs(
        sender_conv, 8, channels, pairs=pairs, order="sender", seed=11
    )
    by_receiver = make_symmetric_inputs(
        receiver_conv, 8, channels, pairs=pairs, order="receiver", seed=11
    )

    # The two orderings hold the same edges, so one is a permutation of the other.
    def endpoints(operands):
        return np.stack([np.asarray(operands[3]), np.asarray(operands[4])], axis=1)

    def canonical(edges):  # sort an edge list by (sender, receiver)
        return np.lexsort((edges[:, 1], edges[:, 0]))

    edges, other = endpoints(by_sender), endpoints(by_receiver)
    perm = canonical(other)[np.argsort(canonical(edges))]
    np.testing.assert_array_equal(other[perm], edges)

    def cotangents(conv, operands):
        node, edge, scalars, senders, receivers = operands

        def fused(n, e, s):
            return conv._fused_mosaic_gpu_eval(n, e, s, senders, receivers)

        out, pullback = jax.vjp(fused, node, edge, scalars)
        cotangent = jnp.asarray(
            np.random.default_rng(12).standard_normal(out.shape, np.float32)
        )
        return out, jax.jit(pullback)(cotangent)

    out_s, (dx_s, dy_s, ds_s) = cotangents(sender_conv, by_sender)
    out_r, (dx_r, dy_r, ds_r) = cotangents(receiver_conv, by_receiver)

    assert relative_error(out_s, out_r) < RTOL, "forward"
    assert relative_error(dx_s, dx_r) < RTOL, "d_node"
    # `perm` maps the receiver-sorted edge list onto the sender-sorted one.
    assert relative_error(dy_s, np.asarray(dy_r)[perm]) < RTOL, "d_edge"
    assert relative_error(ds_s, np.asarray(ds_r)[perm]) < RTOL, "d_scalars"


@pytest.mark.parametrize(
    "ordering", [options.GraphOrdering.RECEIVER, options.GraphOrdering.SENDER]
)
@pytest.mark.parametrize(
    "x_ir, y_ir, o_ir",
    [("0e", "0e + 1o", "0e"), ("0e + 1o", "0e", "1o")],
    ids=["unused_y_coords", "unused_x_coords"],
)
def test_input_coordinates_with_no_path(ordering, x_ir, y_ir, o_ir):
    """An input coordinate no Clebsch-Gordan path reaches gets a zero cotangent.

    Filtering the target prunes output irreps nothing reaches, but leaves the
    input spaces alone: `y = "0e + 1o"` onto `z = "0e"` keeps three `1o`
    coordinates that contribute to nothing. The backward has to fill those rows
    with zeros, and the zero row it fills them with has to be built *inside* the
    kernel body -- a body may capture scalars and refs, not array constants, so
    hoisting it makes exactly these shapes untraceable.
    """
    conv = build_conv(x_ir, y_ir, o_ir, ordering)
    operands = (
        make_symmetric_inputs(conv, 6, 128)
        if ordering == options.GraphOrdering.SENDER
        else make_inputs(conv, 6, 128)
    )
    node, edge, scalars, senders, receivers = operands

    def fused(n, e, s):
        return conv._fused_mosaic_gpu_eval(n, e, s, senders, receivers)

    def unfused(n, e, s):
        return conv._unfused_eval(n, e, s, senders, receivers)

    out, pullback_unfused = jax.vjp(unfused, node, edge, scalars)
    _, pullback_fused = jax.vjp(fused, node, edge, scalars)
    cotangent = jnp.asarray(
        np.random.default_rng(13).standard_normal(out.shape, np.float32)
    )
    got = jax.jit(pullback_fused)(cotangent)
    expected = jax.jit(pullback_unfused)(cotangent)

    for name, a, b in zip(("d_node", "d_edge", "d_scalars"), got, expected):
        assert a.shape == b.shape, f"{name} shape"
        assert relative_error(a, b) < RTOL, f"{name} mismatch"


# ------------------------------------------------------- backward grouping ---


def bwd_kernel_params(conv, ordering, grouping):
    """Kernel params for `conv`, with the backward's grouping pinned.

    `Convolution` builds its own params with default configs, so the grouping
    knob is exercised one level down, on the backward kernel itself.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu import (
        PallasMosaicGPUMessagePassingConvolutionBwdConfig,
        PallasMosaicGPUMessagePassingConvolutionParams,
    )

    coef = conv._otimes.coef
    return PallasMosaicGPUMessagePassingConvolutionParams(
        indices=np.asarray(coef.indices),
        values=np.asarray(coef.data),
        layout=Layout.TRAILING_CHANNELS,
        x_space=conv._otimes.source[0],
        y_space=conv._otimes.source[1],
        z_space=conv._otimes.target,
        graph_ordering=ordering,
        bwd_config=PallasMosaicGPUMessagePassingConvolutionBwdConfig(grouping=grouping),
    )


def sender_sorted_inputs(conv, n_nodes, channels, degrees=None, seed=0):
    """Random operands on a sender-sorted graph that is *not* symmetric.

    `make_symmetric_inputs` is what the `SENDER` *forward* needs; the backward
    asks only for the sort, so this is the sharper fixture for it: `degrees` are
    out-degrees here, and the receivers are unordered.
    """
    node, edge, scalars, senders, receivers = make_inputs(
        conv, n_nodes, channels, degrees=degrees, seed=seed
    )
    return node, edge, scalars, receivers, senders  # the sorted endpoint sends


@pytest.mark.parametrize(
    "ordering, grouping",
    [
        (options.GraphOrdering.RECEIVER, options.GraphOrdering.SENDER),
        (options.GraphOrdering.SENDER, options.GraphOrdering.RECEIVER),
    ],
    ids=["receiver_sorted__sender_grouped", "sender_sorted__receiver_grouped"],
)
@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES[:2], ids=CASE_IDS[:2])
def test_backward_grouping_is_free_of_the_edge_order(
    ordering, grouping, x_ir, y_ir, o_ir, channels
):
    """Either backward body runs on an edge list sorted by either endpoint.

    Grouping by the endpoint the list is *not* sorted by walks the CSR through a
    permutation, which renames edges without reversing them: the three
    cotangents, per-edge order included, must match the reference exactly as the
    unpermuted walk's do -- and on a graph with no symmetry to lean on, since
    only the forward's `SENDER` traversal needs that.

    Variable degrees come along for the ride: the permuted walk builds its CSR
    offsets from an unsorted key, so an empty range is worth pinning here too.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu.bwd import (
        _convolution_kernel_mosaic_gpu_bwd as bwd_kernel,
    )

    conv = build_conv(x_ir, y_ir, o_ir, ordering)
    degrees = np.array([0, 5, 1, 0, 2, 7, 0, 3], np.int32)
    make = (
        sender_sorted_inputs
        if ordering == options.GraphOrdering.SENDER
        else make_inputs
    )
    node, edge, scalars, senders, receivers = make(
        conv, degrees.size, channels, degrees=degrees, seed=17
    )

    out, pullback = jax.vjp(
        lambda n, e, s: conv._unfused_eval(n, e, s, senders, receivers),
        node,
        edge,
        scalars,
    )
    cotangent = jnp.asarray(
        np.random.default_rng(18).standard_normal(out.shape, np.float32)
    )
    expected = jax.jit(pullback)(cotangent)

    params = bwd_kernel_params(conv, ordering, grouping)
    got = jax.jit(lambda *a: bwd_kernel(*a, params))(
        node, edge, scalars, senders, receivers, cotangent
    )

    for name, a, b in zip(("d_node", "d_edge", "d_scalars"), got, expected):
        assert a.shape == b.shape, f"{name} shape"
        assert relative_error(a, b) < RTOL, f"{name} mismatch"


# -------------------------------------------------------- forward grouping ---


def fwd_kernel_params(conv, ordering, grouping):
    """Kernel params for `conv`, with the forward's grouping pinned.

    As `bwd_kernel_params`, one level below `Convolution`: the class builds its
    own params with default configs, so the knob is exercised on the kernel.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu import (
        PallasMosaicGPUMessagePassingConvolutionFwdConfig,
        PallasMosaicGPUMessagePassingConvolutionParams,
    )

    coef = conv._otimes.coef
    return PallasMosaicGPUMessagePassingConvolutionParams(
        indices=np.asarray(coef.indices),
        values=np.asarray(coef.data),
        layout=Layout.TRAILING_CHANNELS,
        x_space=conv._otimes.source[0],
        y_space=conv._otimes.source[1],
        z_space=conv._otimes.target,
        graph_ordering=ordering,
        fwd_config=PallasMosaicGPUMessagePassingConvolutionFwdConfig(grouping=grouping),
    )


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES, ids=CASE_IDS)
def test_receiver_grouped_forward_on_an_asymmetric_sender_sorted_graph(
    x_ir, y_ir, o_ir, channels
):
    """`fwd_config.grouping=RECEIVER` drops every symmetry assumption.

    The graph here is sender-sorted and otherwise arbitrary: no reverse edge,
    no graded-symmetric `y`, no reversal-even scalars. The default `SENDER`
    forward would be wrong on it -- that is what the second half asserts, so the
    test cannot pass by the two paths agreeing trivially -- while the
    receiver-grouped one walks the CSR through an edge permutation and matches
    the reference, coefficients unsigned.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu.fwd import (
        _convolution_kernel_mosaic_gpu_fwd as fwd_kernel,
    )

    conv = build_conv(x_ir, y_ir, o_ir, options.GraphOrdering.SENDER)
    degrees = np.array([0, 5, 1, 0, 2, 7, 0, 3], np.int32)
    node, edge, scalars, senders, receivers = sender_sorted_inputs(
        conv, degrees.size, channels, degrees=degrees, seed=23
    )

    expected = conv._unfused_eval(node, edge, scalars, senders, receivers)
    params = fwd_kernel_params(
        conv, conv.graph_ordering, options.GraphOrdering.RECEIVER
    )
    got = jax.jit(lambda *a: fwd_kernel(*a, params))(
        node, edge, scalars, senders, receivers
    )

    assert got.shape == expected.shape
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=RTOL, atol=ATOL
    )

    # The grouping is doing real work: the symmetry-assuming default disagrees.
    default = fwd_kernel_params(conv, conv.graph_ordering, None)
    unsigned = jax.jit(lambda *a: fwd_kernel(*a, default))(
        node, edge, scalars, senders, receivers
    )
    assert relative_error(unsigned, expected) > RTOL


@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES[:2], ids=CASE_IDS[:2])
def test_sender_grouped_forward_on_a_receiver_sorted_graph(x_ir, y_ir, o_ir, channels):
    """The other direction of the same knob, parity signs included.

    A symmetric graph sorted by receiver, walked outgoing-edge-first through the
    permutation: the reversed-edge traversal and its signed coefficients have to
    survive the renaming of the edges, so this pins the permuted walk against
    the `y` parity rather than around it.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu.fwd import (
        _convolution_kernel_mosaic_gpu_fwd as fwd_kernel,
    )

    conv = build_conv(x_ir, y_ir, o_ir, options.GraphOrdering.RECEIVER)
    pairs = symmetric_pairs(10, np.random.default_rng(24))
    node, edge, scalars, senders, receivers = make_symmetric_inputs(
        conv, 10, channels, pairs=pairs, order="receiver", seed=24
    )

    expected = conv._unfused_eval(node, edge, scalars, senders, receivers)
    params = fwd_kernel_params(conv, conv.graph_ordering, options.GraphOrdering.SENDER)
    got = jax.jit(lambda *a: fwd_kernel(*a, params))(
        node, edge, scalars, senders, receivers
    )

    assert got.shape == expected.shape
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=RTOL, atol=ATOL
    )


def test_forward_grouping_leaves_the_cotangents_alone():
    """Pinning the forward's grouping must not disturb the backward.

    The two passes read the edge list independently -- the backward never leaned
    on the symmetry, so it needs no permutation to stay exact here -- and the op
    as a whole has to differentiate the same convolution it evaluates.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu import convolution_mosaic_gpu

    conv = build_conv(
        "0e + 1o", "0e + 1o", "0e + 1o + 2e", options.GraphOrdering.SENDER
    )
    degrees = np.array([0, 4, 2, 1, 0, 3], np.int32)
    node, edge, scalars, senders, receivers = sender_sorted_inputs(
        conv, degrees.size, 128, degrees=degrees, seed=25
    )
    params = fwd_kernel_params(
        conv, conv.graph_ordering, options.GraphOrdering.RECEIVER
    )

    def fused(n, e, s):
        return convolution_mosaic_gpu(n, e, s, senders, receivers, params)

    def unfused(n, e, s):
        return conv._unfused_eval(n, e, s, senders, receivers)

    out, pullback_unfused = jax.vjp(unfused, node, edge, scalars)
    _, pullback_fused = jax.vjp(fused, node, edge, scalars)
    cotangent = jnp.asarray(
        np.random.default_rng(26).standard_normal(out.shape, np.float32)
    )
    got = jax.jit(pullback_fused)(cotangent)
    expected = jax.jit(pullback_unfused)(cotangent)

    for name, a, b in zip(("d_node", "d_edge", "d_scalars"), got, expected):
        assert a.shape == b.shape, f"{name} shape"
        assert relative_error(a, b) < RTOL, f"{name} mismatch"


# ---------------------------------------------------------------- guards ---


def test_rejects_unsupported_channel_count():
    """Channels must be a multiple of the 128-lane vector granularity."""
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e")
    node, edge, scalars, senders, receivers = make_inputs(conv, 8, 64)

    with pytest.raises(NotImplementedError, match="channels"):
        conv._fused_mosaic_gpu_eval(node, edge, scalars, senders, receivers)


def test_rejects_zero_channels():
    """An empty channel axis is no multiple of the vector granularity either.

    It used to pass validation, and the backward then crashed storing `dy`.
    """
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e")
    node, edge, scalars, senders, receivers = make_inputs(conv, 8, 0)

    with pytest.raises(NotImplementedError, match="positive multiple"):
        conv._fused_mosaic_gpu_eval(node, edge, scalars, senders, receivers)


def test_rejects_an_output_space_wider_than_an_async_copy():
    """The backward stages whole cotangent rows, so `z_dim` is bounded.

    The forward never stages the output, but refuses the shape too: it is one
    the op could not differentiate.
    """
    conv = Convolution(
        source=(O3Space("2x0e + 2x1o + 2x2e + 2x3o"), O3Space("0e + 1o + 2e + 3o")),
        target=None,  # inferred: every path, 512 coordinates
        layout=Layout.TRAILING_CHANNELS,
        graph_ordering=options.GraphOrdering.RECEIVER,
    )
    assert conv._otimes.target.dim > 256
    node, edge, scalars, senders, receivers = make_inputs(conv, 8, 128)

    with pytest.raises(NotImplementedError, match="z_space"):
        conv._fused_mosaic_gpu_eval(node, edge, scalars, senders, receivers)


def test_default_channel_block_narrows_to_fit_the_backward():
    """The wide default must narrow where its backward would not fit SMEM.

    At `l_max=2` the auto-selected block is 256 channels, and with every parity
    in the spaces the receiver-grouped backward then stages more than a CTA's
    shared memory. The forward ran and the gradient failed to compile, with no
    knob reachable from :class:`e3j.core.Convolution` to fix it.
    """
    space = "0e + 0o + 1o + 1e + 2e + 2o"
    conv = build_conv(space, "0e + 1o + 2e", space)
    node, edge, scalars, senders, receivers = make_inputs(conv, 8, 256)

    def loss(evaluate):
        def energy(n, e, s):
            return jnp.sum(evaluate(n, e, s, senders, receivers) ** 2)

        return jax.jit(jax.value_and_grad(energy, argnums=(0, 1, 2)))

    got = loss(conv._fused_mosaic_gpu_eval)(node, edge, scalars)
    expected = loss(conv._unfused_eval)(node, edge, scalars)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        assert relative_error(a, b) < RTOL


def test_rejects_unordered_graph():
    """The kernel reads the edge list as a CSR, so one endpoint must be sorted."""
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e", options.GraphOrdering.NONE)
    node, edge, scalars, senders, receivers = make_inputs(conv, 8, 128)

    with pytest.raises(NotImplementedError, match="SENDER and RECEIVER"):
        conv._fused_mosaic_gpu_eval(node, edge, scalars, senders, receivers)


@pytest.mark.parametrize("ordering", [None, options.GraphOrdering.NONE])
def test_params_reject_a_graph_ordering_naming_no_endpoint(ordering):
    """`graph_ordering` is what the configs' `None` groupings fall back to.

    So it has to name an endpoint itself: a `None` resolved both passes'
    groupings to `None`, which read any edge list as already receiver-sorted
    and silently returned a wrong forward on a sender-sorted one.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu.params import (
        PallasMosaicGPUMessagePassingConvolutionParams,
    )

    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e")
    coef = conv._otimes.coef
    with pytest.raises(NotImplementedError, match="graph_ordering"):
        PallasMosaicGPUMessagePassingConvolutionParams(
            indices=np.asarray(coef.indices),
            values=np.asarray(coef.data),
            layout=Layout.TRAILING_CHANNELS,
            x_space=conv._otimes.source[0],
            y_space=conv._otimes.source[1],
            z_space=conv._otimes.target,
            graph_ordering=ordering,
        )


def _fwd_with_channel_block(
    conv, channel_block, node, edge, scalars, senders, receivers
):
    """Run the forward kernel directly with one `channel_block` setting."""
    from e3j.pallas_ops.convolution.mosaic_gpu import (
        PallasMosaicGPUMessagePassingConvolutionFwdConfig,
        PallasMosaicGPUMessagePassingConvolutionParams,
    )
    from e3j.pallas_ops.convolution.mosaic_gpu.fwd import (
        _convolution_kernel_mosaic_gpu_fwd,
    )

    coef = conv._otimes.coef
    params = PallasMosaicGPUMessagePassingConvolutionParams(
        indices=np.asarray(coef.indices),
        values=np.asarray(coef.data),
        layout=Layout.TRAILING_CHANNELS,
        x_space=conv._otimes.source[0],
        y_space=conv._otimes.source[1],
        z_space=conv._otimes.target,
        graph_ordering=conv.graph_ordering,
        fwd_config=PallasMosaicGPUMessagePassingConvolutionFwdConfig(
            channel_block=channel_block
        ),
    )
    return jax.jit(lambda *a: _convolution_kernel_mosaic_gpu_fwd(*a, params))(
        node, edge, scalars, senders, receivers
    )


@pytest.mark.parametrize("channels, channel_block", [(128, 128), (256, 128)])
def test_channel_block_matches_one_full_width_pass(channels, channel_block):
    """Splitting the channel axis into passes must not change the result.

    Each pass sweeps every node over its own slice of the channel axis, so the
    slices partition the output and a blocked run has to reproduce the
    full-width one exactly -- the arithmetic per channel is untouched.
    """
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e")
    inputs = make_inputs(conv, 16, channels)

    expected = _fwd_with_channel_block(conv, None, *inputs)
    got = _fwd_with_channel_block(conv, channel_block, *inputs)

    assert got.shape == expected.shape
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize(
    "channels, channel_block, match",
    [(128, 64, "multiple of"), (256, 192, "must divide"), (512, 512, "async copy")],
)
def test_rejects_unsupported_channel_block(channels, channel_block, match):
    """A block must divide `channels` and stay a whole number of warpgroup rows.

    Mosaic GPU's strided warpgroup layout only represents an element count that
    is a multiple of 128, so 64 is not a legal pass width however well it
    divides -- which is why `channels=128` has no legal split at all.
    """
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e")
    inputs = make_inputs(conv, 8, channels)

    with pytest.raises(ValueError, match=match):
        _fwd_with_channel_block(conv, channel_block, *inputs)


@pytest.mark.parametrize(
    "channels, l_max, expected",
    [
        (128, 1, 128),  # 256 preferred at low l_max, but doesn't divide 128
        (256, 1, 256),  # low l_max: the wide default, unblocked
        (256, 2, 256),  # l_max=2 is still "low"
        (256, 3, 128),  # l_max=3 narrows to the tuned default
        (384, 1, 128),  # 256 doesn't divide 384 either -- same fallback
        (384, 3, 128),
    ],
)
def test_resolve_channel_block_defaults_by_l_max(channels, l_max, expected):
    """`channel_block=None` auto-selects 128 from `l_max=3` up, else 256.

    Pure logic, no launch: the fallback to 128 when the wider default does not
    divide `channels` is what makes the auto-selection always legal, whatever
    `l_max` says.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu.common import resolve_channel_block

    assert resolve_channel_block(channels, None, l_max) == expected


def test_params_l_max_is_the_max_of_the_three_spaces():
    """`params.l_max` -- what the auto-selected `channel_block` keys off of.

    Deliberately picks three spaces with three different maxima, so the
    property has to take the actual max rather than e.g. just `z_space`'s.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu import (
        PallasMosaicGPUMessagePassingConvolutionParams,
    )

    conv = build_conv("0e + 1o", "0e + 1o + 2e", "0e + 1o + 2e + 3o")
    coef = conv._otimes.coef
    params = PallasMosaicGPUMessagePassingConvolutionParams(
        indices=np.asarray(coef.indices),
        values=np.asarray(coef.data),
        layout=Layout.TRAILING_CHANNELS,
        x_space=conv._otimes.source[0],
        y_space=conv._otimes.source[1],
        z_space=conv._otimes.target,
    )
    assert params.x_space.l_max == 1
    assert params.y_space.l_max == 2
    assert params.z_space.l_max == 3
    assert params.l_max == 3


def _bwd_with_channel_block(
    conv, grouping, channel_block, node, edge, scalars, senders, receivers, cotangent
):
    """Run the backward kernel directly with one `channel_block` setting."""
    from e3j.pallas_ops.convolution.mosaic_gpu import (
        PallasMosaicGPUMessagePassingConvolutionBwdConfig,
        PallasMosaicGPUMessagePassingConvolutionParams,
    )
    from e3j.pallas_ops.convolution.mosaic_gpu.bwd import (
        _convolution_kernel_mosaic_gpu_bwd,
    )

    coef = conv._otimes.coef
    params = PallasMosaicGPUMessagePassingConvolutionParams(
        indices=np.asarray(coef.indices),
        values=np.asarray(coef.data),
        layout=Layout.TRAILING_CHANNELS,
        x_space=conv._otimes.source[0],
        y_space=conv._otimes.source[1],
        z_space=conv._otimes.target,
        graph_ordering=conv.graph_ordering,
        bwd_config=PallasMosaicGPUMessagePassingConvolutionBwdConfig(
            grouping=grouping, channel_block=channel_block
        ),
    )
    return jax.jit(
        lambda *a: _convolution_kernel_mosaic_gpu_bwd(
            *a, senders, receivers, cotangent, params
        )
    )(node, edge, scalars)


@pytest.mark.parametrize(
    "grouping",
    [options.GraphOrdering.RECEIVER, options.GraphOrdering.SENDER],
    ids=["receiver_grouped", "sender_grouped"],
)
@pytest.mark.parametrize("channels, channel_block", [(128, 128), (256, 128)])
def test_bwd_channel_block_matches_one_full_width_pass(
    channels, channel_block, grouping
):
    """Splitting the channel axis into passes must not change the cotangents.

    `dx` and `ds` are channel-indexed and partition across passes like the
    forward's output. `dy` carries no channel axis at all -- each pass reduces
    only its own slice of channels, so the passes are summed into it instead of
    concatenated -- which is the sharper case here: a wrong reduction would show
    up as a `dy`-only mismatch, `dx`/`ds` being exact slices either way. Both
    backward bodies are covered, since blocking touches their zero-init and
    accumulator shapes differently (`_receiver_grouped_body` vs
    `_sender_grouped_body`).
    """
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e")
    node, edge, scalars, senders, receivers = make_inputs(conv, 16, channels)
    cotangent = jnp.asarray(
        np.random.default_rng(31).standard_normal(
            (16, conv.target.dim, channels), np.float32
        )
    )

    expected = _bwd_with_channel_block(
        conv, grouping, None, node, edge, scalars, senders, receivers, cotangent
    )
    got = _bwd_with_channel_block(
        conv,
        grouping,
        channel_block,
        node,
        edge,
        scalars,
        senders,
        receivers,
        cotangent,
    )

    for name, a, b in zip(("d_node", "d_edge", "d_scalars"), got, expected):
        assert a.shape == b.shape, f"{name} shape"
        assert relative_error(a, b) < RTOL, f"{name} mismatch"


@pytest.mark.parametrize(
    "channels, channel_block, match",
    [(128, 64, "multiple of"), (256, 192, "must divide"), (512, 512, "async copy")],
)
def test_rejects_unsupported_bwd_channel_block(channels, channel_block, match):
    """As the forward: a block must divide `channels` and be a multiple of 128."""
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e")
    node, edge, scalars, senders, receivers = make_inputs(conv, 8, channels)
    cotangent = jnp.asarray(
        np.random.default_rng(32).standard_normal(
            (8, conv.target.dim, channels), np.float32
        )
    )

    with pytest.raises(ValueError, match=match):
        _bwd_with_channel_block(
            conv,
            options.GraphOrdering.RECEIVER,
            channel_block,
            node,
            edge,
            scalars,
            senders,
            receivers,
            cotangent,
        )


def _grad_norm(eval_fn, senders, receivers, weights):
    """Weighted squared norm of all three first-order gradients, a scalar of
    `(node, edge, scalars)` whose gradient needs every second-order term."""
    energy_grad = jax.grad(
        lambda n, e, s: jnp.sum(eval_fn(n, e, s, senders, receivers) ** 2),
        argnums=(0, 1, 2),
    )

    def loss(n, e, s):
        return sum(jnp.sum(w * g**2) for w, g in zip(weights, energy_grad(n, e, s)))

    return loss


def _random_like(arrays, seed):
    rng = np.random.default_rng(seed)
    return [jnp.asarray(rng.standard_normal(a.shape), a.dtype) for a in arrays]


@pytest.mark.parametrize(
    "ordering", [options.GraphOrdering.RECEIVER, options.GraphOrdering.SENDER]
)
@pytest.mark.parametrize("x_ir, y_ir, o_ir, channels", CASES, ids=CASE_IDS)
def test_double_backward_matches_unfused(ordering, x_ir, y_ir, o_ir, channels):
    """Second-order cotangents on all three operands match the reference.

    The cotangents reaching `dy` and `ds` are not reversal-symmetric, so under
    `SENDER` this also checks they avoid the sender-grouped forward.
    """
    conv = build_conv(x_ir, y_ir, o_ir, ordering)
    if ordering == options.GraphOrdering.SENDER:
        node, edge, scalars, senders, receivers = make_symmetric_inputs(
            conv, 8, channels
        )
    else:
        node, edge, scalars, senders, receivers = make_inputs(conv, 8, channels)
    weights = _random_like((node, edge, scalars), seed=40)

    def second_order(eval_fn):
        loss = _grad_norm(eval_fn, senders, receivers, weights)
        return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(node, edge, scalars)

    got = second_order(conv._fused_mosaic_gpu_eval)
    expected = second_order(conv._unfused_eval)
    for name, a, b in zip(("dd_node", "dd_edge", "dd_scalars"), got, expected):
        assert a.shape == b.shape, f"{name} shape"
        assert relative_error(a, b) < RTOL, f"{name} mismatch"


@pytest.mark.parametrize(
    "ordering", [options.GraphOrdering.RECEIVER, options.GraphOrdering.SENDER]
)
def test_third_order_matches_unfused(ordering):
    """The double-backward is itself differentiable."""
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e", ordering)
    if ordering == options.GraphOrdering.SENDER:
        node, edge, scalars, senders, receivers = make_symmetric_inputs(conv, 8, 128)
    else:
        node, edge, scalars, senders, receivers = make_inputs(conv, 8, 128)
    weights = _random_like((node, edge, scalars), seed=41)
    weights2 = _random_like((node, edge, scalars), seed=42)

    def third_order(eval_fn):
        loss = _grad_norm(eval_fn, senders, receivers, weights)
        dd = jax.grad(loss, argnums=(0, 1, 2))

        def loss2(n, e, s):
            return sum(jnp.sum(w * g**2) for w, g in zip(weights2, dd(n, e, s)))

        return jax.jit(jax.grad(loss2, argnums=(0, 1, 2)))(node, edge, scalars)

    got = third_order(conv._fused_mosaic_gpu_eval)
    expected = third_order(conv._unfused_eval)
    for name, a, b in zip(("ddd_node", "ddd_edge", "ddd_scalars"), got, expected):
        assert relative_error(a, b) < RTOL, f"{name} mismatch"


def test_double_backward_under_vmap():
    """Force-training style second order over a vmapped batch of graphs."""
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e")
    batch = [make_inputs(conv, 8, 128, seed=seed) for seed in range(3)]
    node, edge, scalars, senders, receivers = (jnp.stack(a) for a in zip(*batch))

    def second_order(eval_fn):
        def loss(n, e, s, snd, rcv):
            forces = jax.grad(lambda n: jnp.sum(eval_fn(n, e, s, snd, rcv) ** 2))(n)
            return jnp.sum(forces**2)

        grad = jax.grad(loss, argnums=(0, 1, 2))
        return jax.jit(jax.vmap(grad))(node, edge, scalars, senders, receivers)

    got = second_order(conv._fused_mosaic_gpu_eval)
    expected = second_order(conv._unfused_eval)
    for name, a, b in zip(("dd_node", "dd_edge", "dd_scalars"), got, expected):
        assert relative_error(a, b) < RTOL, f"{name} mismatch"


# --------------------------------------------------------------- dispatch ---


@pytest.mark.parametrize(
    "ordering", [options.GraphOrdering.RECEIVER, options.GraphOrdering.SENDER]
)
def test_class_dispatch_routes_to_kernel(ordering):
    """`Convolution` under `FUSED_MOSAIC_GPU` routes through the Mosaic kernel."""
    with config.use(convolution=options.Convolution.FUSED_MOSAIC_GPU):
        conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e", ordering)
        operands = (
            make_symmetric_inputs(conv, 8, 128)
            if ordering == options.GraphOrdering.SENDER
            else make_inputs(conv, 8, 128)
        )
        lowered = jax.jit(conv).lower(*operands)
        expected = conv._unfused_eval(*operands)
        got = jax.jit(conv)(*operands)

    assert "mosaic_gpu" in lowered.as_text()
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=RTOL, atol=ATOL
    )


def _graph_as_traced_arguments(name):
    """Ways a model differentiates the convolution with its graph traced.

    Each takes the graph as arguments of the outermost transformation, as a
    model does, so the edge indices are tracers of that trace. Each returns the
    gradient of a scalar energy with respect to the node features.
    """

    def energy(conv, node, edge, scalars, senders, receivers):
        return jnp.sum(conv(node, edge, scalars, senders, receivers) ** 2)

    def grad_of_jit(conv, *operands):  # forces from a jitted energy
        return jax.grad(jax.jit(functools.partial(energy, conv)))(*operands)

    def jit_grad_checkpoint(conv, *operands):
        loss = jax.checkpoint(functools.partial(energy, conv))
        return jax.jit(jax.grad(loss))(*operands)

    def grad_through_cond(conv, *operands):
        def loss(node, *rest):
            return jax.lax.cond(
                True,
                lambda n: energy(conv, n, *rest),
                lambda n: jnp.sum(n),
                node,
            )

        return jax.jit(jax.grad(loss))(*operands)

    def grad_through_scan(conv, node, edge, scalars, senders, receivers):
        x_dim = node.shape[1]

        def loss(node, edge, scalars, senders, receivers):
            def layer(h, _):
                out = conv(h, edge, scalars, senders, receivers)
                return h + 0.1 * out[:, :x_dim], None

            h, _ = jax.lax.scan(layer, node, None, length=2)
            return jnp.sum(h**2)

        return jax.jit(jax.grad(loss))(node, edge, scalars, senders, receivers)

    return {
        "grad_of_jit": grad_of_jit,
        "jit_grad_checkpoint": jit_grad_checkpoint,
        "grad_through_cond": grad_through_cond,
        "grad_through_scan": grad_through_scan,
    }[name]


@pytest.mark.parametrize(
    "pattern",
    ["grad_of_jit", "jit_grad_checkpoint", "grad_through_cond", "grad_through_scan"],
)
def test_differentiates_with_a_traced_graph(pattern):
    """The edge indices may be tracers of any enclosing transformation.

    The custom VJP once closed over them instead of taking them as arguments,
    so its lazily traced rules leaked them: `No constant handler for type
    DynamicJaxprTracer` or `UnexpectedTracerError`, depending on the pattern.
    """
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o")
    operands = make_inputs(conv, 8, 128)
    run = _graph_as_traced_arguments(pattern)
    with config.use(convolution=options.Convolution.FUSED_MOSAIC_GPU):
        fused = build_conv("0e + 1o", "0e + 1o", "0e + 1o")
        got = run(fused, *operands)
    expected = run(conv, *operands)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "x_ir, y_ir, o_ir, channels",
    [
        ("0e + 1o", "0e + 1o", "0e + 1o + 2e", 128),
        # Two channel-blocking passes threaded through one buffer.
        ("0e + 1o + 2e + 3o", "0e + 1o", "0e + 1o + 2e + 3o", 256),
    ],
)
def test_results_survive_lax_map_and_scan(x_ir, y_ir, o_ir, channels):
    """Results written in place must reach the caller of a loop body.

    The kernels write through buffers they alias, and inside a `lax.map` or a
    `lax.scan` XLA's dynamic-slice fusion used to fold the loop's `ys` update
    into such a call -- the kernel's writes then went nowhere, and the loop
    returned zeros, forward and backward, with no error.
    """
    conv = build_conv(x_ir, y_ir, o_ir)
    stacked = jax.tree.map(
        lambda *a: jnp.stack(a),
        *(make_inputs(conv, 16, channels, seed=seed) for seed in range(3)),
    )

    def mapped(evaluate, operands):
        return jax.lax.map(lambda a: evaluate(*a), operands)

    def scanned(evaluate, operands):
        return jax.lax.scan(lambda c, a: (c, evaluate(*a)), 0.0, operands)[1]

    def grad_mapped(evaluate, operands):
        def one(a):
            node, edge, scalars, senders, receivers = a

            def loss(n, e, s):
                return jnp.sum(evaluate(n, e, s, senders, receivers) ** 2)

            return jax.grad(loss, argnums=(0, 1, 2))(node, edge, scalars)

        return jax.lax.map(one, operands)

    for loop in (mapped, scanned, grad_mapped):
        got = jax.jit(lambda ops, f=loop: f(conv._fused_mosaic_gpu_eval, ops))(stacked)
        expected = jax.jit(lambda ops, f=loop: f(conv._unfused_eval, ops))(stacked)
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
            assert relative_error(a, b) < RTOL, loop.__name__


@pytest.mark.parametrize(
    "ordering", [options.GraphOrdering.RECEIVER, options.GraphOrdering.SENDER]
)
def test_vmap_over_stacked_graphs(ordering):
    """Stacked graphs vmap, padding edges included, forward and backward.

    The op folds the mapped graphs into one disjoint graph; each one's padding
    edges then sit mid-list, ahead of the next graph's edges. Mapping over the
    features alone shares one graph across the slices, and `jacrev` maps the
    backward over the output's coordinates.
    """
    conv = build_conv("0e + 1o", "0e + 1o", "0e + 1o + 2e", ordering)
    graphs = [
        pad_with_dummy_edges(
            (
                make_symmetric_inputs(conv, 8, 128, seed=seed)
                if ordering == options.GraphOrdering.SENDER
                else make_inputs(conv, 8, 128, seed=seed)
            ),
            3,
            seed=seed,
        )
        for seed in range(3)
    ]
    stacked = jax.tree.map(lambda *a: jnp.stack(a), *graphs)
    node, edge, scalars, senders, receivers = graphs[0]

    def mapped(evaluate):
        return jax.vmap(evaluate)(*stacked)

    def shared_graph(evaluate):
        return jax.vmap(evaluate, (0, None, None, None, None))(
            stacked[0], edge, scalars, senders, receivers
        )

    def mapped_grad(evaluate):
        def one(n, e, s, snd, rcv):
            def loss(n, e, s):
                return jnp.sum(evaluate(n, e, s, snd, rcv) ** 2)

            return jax.grad(loss, argnums=(0, 1, 2))(n, e, s)

        return jax.vmap(one)(*stacked)

    def jacobian(evaluate):
        def per_node(s):
            return evaluate(node, edge, s, senders, receivers)[:2].sum(axis=(1, 2))

        return jax.jacrev(per_node)(scalars)

    for transform in (mapped, shared_graph, mapped_grad, jacobian):
        got = jax.jit(lambda f=transform: f(conv._fused_mosaic_gpu_eval))()
        expected = jax.jit(lambda f=transform: f(conv._unfused_eval))()
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
            assert relative_error(a, b) < RTOL, transform.__name__


# -------------------------------------------------------- padding edges ---


def bwd_params(conv: Convolution, grouping: options.GraphOrdering):
    """Kernel params pinning which of the two backward bodies runs."""
    from e3j.pallas_ops.convolution.mosaic_gpu import (
        PallasMosaicGPUMessagePassingConvolutionBwdConfig,
        PallasMosaicGPUMessagePassingConvolutionParams,
    )

    coef = conv._otimes.coef
    return PallasMosaicGPUMessagePassingConvolutionParams(
        indices=np.asarray(coef.indices),
        values=np.asarray(coef.data),
        layout=Layout.TRAILING_CHANNELS,
        x_space=conv._otimes.source[0],
        y_space=conv._otimes.source[1],
        z_space=conv._otimes.target,
        graph_ordering=conv.graph_ordering,
        bwd_config=PallasMosaicGPUMessagePassingConvolutionBwdConfig(grouping=grouping),
    )


def pad_with_dummy_edges(operands, num_padding: int, seed: int = 21):
    """Append `num_padding` `DUMMY_INDEX`-marked edges carrying junk features."""
    from e3j.data.graph import DUMMY_INDEX

    node, edge, scalars, senders, receivers = operands
    rng = np.random.default_rng(seed)

    def junk(reference):
        shape = (num_padding, *reference.shape[1:])
        return jnp.asarray(rng.standard_normal(shape, np.float32))

    sentinel = jnp.full((num_padding,), DUMMY_INDEX, jnp.int32)
    return (
        node,
        jnp.concatenate([edge, junk(edge)]),
        jnp.concatenate([scalars, junk(scalars)]),
        jnp.concatenate([senders, sentinel]),
        jnp.concatenate([receivers, sentinel]),
    )


@pytest.mark.parametrize(
    "grouping",
    [options.GraphOrdering.RECEIVER, options.GraphOrdering.SENDER],
    ids=["receiver_grouped", "sender_grouped"],
)
# Under the node count, so some CTAs zero one row and the rest none, then over
# it, so `zero_dropped_edges`' strided walk wraps.
@pytest.mark.parametrize("num_padding", [5, 21])
def test_padding_edges_get_zero_cotangents(grouping, num_padding):
    """A padding edge is walked by no CTA, so `zero_dropped_edges` owes it zeros.

    Without that pass its rows are whatever `pl.empty` handed back.
    """
    from e3j.pallas_ops.convolution.mosaic_gpu.bwd import (
        _convolution_kernel_mosaic_gpu_bwd,
    )

    conv = build_conv("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e")
    params = bwd_params(conv, grouping)
    degrees = np.array([0, 5, 1, 0, 2, 7, 0, 3], np.int32)
    real = make_inputs(conv, degrees.size, 128, degrees=degrees, seed=13)
    n_real = int(real[1].shape[0])
    padded = pad_with_dummy_edges(real, num_padding)
    cotangent = jnp.asarray(
        np.random.default_rng(14).standard_normal(
            (degrees.size, conv.target.dim, 128), np.float32
        )
    )

    def backward(operands):
        node, edge, scalars, senders, receivers = operands
        return jax.jit(
            lambda *a: _convolution_kernel_mosaic_gpu_bwd(
                *a, senders, receivers, cotangent, params
            )
        )(node, edge, scalars)

    # Real edges first, so the padded run inherits a dirty buffer if anything
    # is left unwritten.
    dx, dy, ds = backward(real)
    dx_padded, dy_padded, ds_padded = backward(padded)

    assert not np.any(np.asarray(dy_padded)[n_real:]), "padding dy must be zero"
    assert not np.any(np.asarray(ds_padded)[n_real:]), "padding ds must be zero"
    assert relative_error(dx_padded, dx) < RTOL, "d_node"
    assert relative_error(np.asarray(dy_padded)[:n_real], dy) < RTOL, "d_edge"
    assert relative_error(np.asarray(ds_padded)[:n_real], ds) < RTOL, "d_scalars"


@pytest.mark.parametrize(
    "ordering",
    [options.GraphOrdering.RECEIVER, options.GraphOrdering.SENDER],
    ids=["receiver", "sender"],
)
def test_empty_edge_range_ignores_non_finite_padding(ordering):
    """A node with no edge is zero even when the padding edges hold NaNs.

    Padding edges (`DUMMY_INDEX`) sort past every node, so the position an
    empty node's range starts at can be one of them -- and their features are
    often non-finite, e.g. a radial basis evaluated at zero length. Cancelling
    such a step against a zero `y` row would still give `0 * NaN = NaN`.
    """
    conv = build_conv("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", ordering)
    if ordering == options.GraphOrdering.RECEIVER:
        degrees = np.array([0, 5, 1, 0, 2, 7, 0, 3, 0, 0], np.int32)
        real = make_inputs(conv, degrees.size, 128, degrees=degrees, seed=31)
    else:
        # The `SENDER` forward needs a symmetric graph; nodes 0, 3 and the
        # last two have no edge.
        pairs = [(1, 2), (1, 4), (2, 5), (4, 5), (5, 6), (6, 7), (1, 7)]
        real = make_symmetric_inputs(conv, 10, 128, pairs=pairs, seed=31)
    node, edge, scalars, senders, receivers = pad_with_dummy_edges(real, 4)
    n_real = int(real[1].shape[0])
    edge = edge.at[n_real:].set(jnp.nan)
    scalars = scalars.at[n_real:].set(jnp.inf)

    expected = conv._unfused_eval(*real)
    got = np.asarray(
        jax.jit(conv._fused_mosaic_gpu_eval)(node, edge, scalars, senders, receivers)
    )

    assert np.all(np.isfinite(got)), "padding edges leaked into the output"
    np.testing.assert_allclose(got, np.asarray(expected), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "ordering",
    [options.GraphOrdering.RECEIVER, options.GraphOrdering.SENDER],
    ids=["receiver", "sender"],
)
def test_graph_with_no_edges(ordering):
    """An edgeless graph aggregates nothing: zero output, zero `dx`, empty rest."""
    conv = build_conv("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", ordering)
    node, edge, scalars, senders, receivers = make_inputs(
        conv, 6, 128, degrees=np.zeros(6, np.int32), seed=41
    )
    assert edge.shape[0] == 0

    def loss(node, edge, scalars):
        return conv._fused_mosaic_gpu_eval(node, edge, scalars, senders, receivers)

    out, vjp = jax.vjp(jax.jit(loss), node, edge, scalars)
    d_node, d_edge, d_scalars = vjp(jnp.ones_like(out))

    assert out.shape == (6, conv.target.dim, 128)
    assert not np.any(np.asarray(out)), "output must be zero"
    assert not np.any(np.asarray(d_node)), "d_node must be zero"
    assert d_edge.shape == edge.shape
    assert d_scalars.shape == scalars.shape
