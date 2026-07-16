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

import e3nn_jax as e3nn
import jax
import jax.numpy as np
import jax.random as random
import numpy
import pytest
from conftest import assert_allclose

import e3j
from e3j.core.convolution import Convolution
from e3j.ops.coef import Coef4D
from e3j.utils.options import Layout
from e3j.utils.sparse import narrow_index_dtype


class _TestConvolution:
    """Base class for Convolution tests, exercising the unfused plain-JAX path."""

    node_irreps: str
    edge_irreps: str
    out: str | None = None
    num_nodes: int = 6
    num_edges: int = 18
    num_channels: int = 4
    layout: str = "TRAILING_CHANNELS"
    _seed: int = 42

    @property
    def key(self) -> jax.Array:
        return random.key(self._seed)

    @property
    def is_trailing(self) -> bool:
        return Layout.parse(self.layout) == Layout.TRAILING_CHANNELS

    def _rotate(self, x: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Act with representation matrix `D` on the irrep axis of `x`."""
        # Irrep axis is trailing under LEADING_CHANNELS, else second-to-last.
        return x @ D if not self.is_trailing else np.einsum("nic,ij->njc", x, D)

    def _feature_shape(self, dim: int) -> tuple[int, ...]:
        """Node/edge-scalar shape `(dim, K)` or `(K, dim)` from the irrep `dim`."""
        K = self.num_channels
        return (dim, K) if self.is_trailing else (K, dim)

    def assert_zero(self, data, tol: float = 5e-6):
        norm = float(np.sqrt(np.sum(data**2))) / data.size
        assert norm < tol

    # --- Fixtures ---

    @pytest.fixture(scope="class")
    def module(self) -> Convolution:
        """Build a Convolution capturing the unfused config at construction."""
        with e3j.config.use(convolution="UNFUSED", tensor_product="UNFUSED"):
            return Convolution(
                (self.node_irreps, self.edge_irreps),
                self.out,
                layout=self.layout,
                graph_ordering="NONE",
            )

    @pytest.fixture(scope="class")
    def graph(self) -> tuple[np.ndarray, np.ndarray]:
        """Random graph with edges sorted by receiver (CSR requirement)."""
        k1, k2 = random.split(random.key(7))
        senders = random.randint(k1, (self.num_edges,), 0, self.num_nodes)
        receivers = random.randint(k2, (self.num_edges,), 0, self.num_nodes)
        order = np.argsort(receivers)
        return senders[order], receivers[order]

    @pytest.fixture
    def inputs(self, module) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        node_space, edge_space, scalar_space = module.source
        k1, k2, k3 = random.split(random.key(123), 3)
        node_features = random.normal(
            k1, (self.num_nodes, *self._feature_shape(node_space.dim))
        )
        edge_features = random.normal(k2, (self.num_edges, edge_space.dim))
        edge_scalars = random.normal(
            k3, (self.num_edges, *self._feature_shape(scalar_space.dim))
        )
        return node_features, edge_features, edge_scalars

    @pytest.fixture(scope="class")
    def rotations(self, module) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Output, node and edge rotation matrices for a shared SO3 element."""
        rotation = e3nn.rand_matrix(self.key)
        node_space, edge_space, _ = module.source
        return (
            module.target.action(rotation),
            node_space.action(rotation),
            edge_space.action(rotation),
        )

    # --- Test functions ---

    def test_output_shape(self, module, inputs, graph):
        senders, receivers = graph
        z = module(*inputs, senders, receivers)
        assert z.shape == (self.num_nodes, *self._feature_shape(module.target.dim))

    def test_coef(self, module):
        """`coef` is a rank-4 Coef4D with self-consistent index dtype.

        Regression guard: the scalar-index column is taken from `mix_indices`,
        whose default integer type is wider than the narrowed CG index dtype.
        Without an explicit cast it promotes the stacked array, leaving
        `coef.idx` and `coef.idx_dtype` inconsistent (and `pack_jax` unsound).
        """
        coef = module.coef
        assert isinstance(coef, Coef4D)
        assert coef.rank == 4
        assert coef.idx.shape == (coef.val.shape[0], 4)

        # The index array dtype must match the declared idx_dtype, and be the
        # smallest integer type fitting the four (output, x, y, scalar) bounds.
        bounds = (
            module.target.dim,
            module.source[0].dim,
            module.source[1].dim,
            module.source[2].dim,
        )
        assert numpy.dtype(coef.idx.dtype) == numpy.dtype(coef.idx_dtype)
        assert numpy.dtype(coef.idx_dtype) == numpy.dtype(narrow_index_dtype(bounds))
        assert numpy.dtype(coef.val.dtype) == numpy.dtype(coef.val_dtype)

        # Indices stay in bounds and the scalar column routes via mix_indices.
        assert int(coef.idx.min()) >= 0
        assert bool((coef.idx.max(axis=0) < numpy.array(bounds)).all())
        numpy.testing.assert_array_equal(
            coef.idx[:, 3], module._mix.mix_indices[coef.idx[:, 0]]
        )

        # Packing preserves the index dtype through a round-trip.
        unpacked = Coef4D.unpack(coef.pack_jax(), val_dtype=coef.val_dtype)
        assert numpy.dtype(unpacked.idx.dtype) == numpy.dtype(coef.idx_dtype)

    def test_equivariance(self, module, inputs, graph, rotations):
        """Convolution commutes with SO3 acting on node and edge features."""
        senders, receivers = graph
        node_features, edge_features, edge_scalars = inputs
        rotation_out, rotation_node, rotation_edge = rotations

        gfx = self._rotate(module(*inputs, senders, receivers), rotation_out)
        fgx = module(
            self._rotate(node_features, rotation_node),
            edge_features @ rotation_edge,
            edge_scalars,
            senders,
            receivers,
        )
        self.assert_zero(gfx - fgx)

    def test_avg_num_neighbors(self, module, inputs, graph):
        """`avg_num_neighbors` divides the summed messages by a constant."""
        senders, receivers = graph
        avg = 3.0
        with e3j.config.use(convolution="UNFUSED", tensor_product="UNFUSED"):
            scaled = Convolution(
                (self.node_irreps, self.edge_irreps),
                self.out,
                layout=self.layout,
                avg_num_neighbors=avg,
                graph_ordering="NONE",
            )
        assert_allclose(
            module(*inputs, senders, receivers) / avg,
            scaled(*inputs, senders, receivers),
        )

    def test_jittable(self, module, inputs, graph):
        """Module can be compiled (CG/CSR setup excluded from the trace)."""
        senders, receivers = graph
        f = jax.jit(lambda a, b, c: module(a, b, c, senders, receivers))
        z = f(*inputs)
        feature_axis = -2 if self.is_trailing else -1
        assert z.shape[feature_axis] == module.target.dim


class TestConvolution_full(_TestConvolution):
    node_irreps = "0e + 1o + 2e + 3o"
    edge_irreps = "0e + 1o + 2e + 3o"
    out = None


class TestConvolution_filtered(_TestConvolution):
    node_irreps = "2x0e + 2x1o"
    edge_irreps = "0e + 1o + 2e"
    out = "0e + 1o + 2e"


class TestConvolution_pseudotensors(_TestConvolution):
    """Check parity-flipped irreps under improper rotations."""

    node_irreps = "0o + 1e + 2o + 3e"
    edge_irreps = "0o + 1e + 2o + 3e"
    out = None

    @pytest.fixture(scope="class")
    def rotations(self, module) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Output, node and edge action matrices for an improper rotation.

        Negating a proper rotation `R` (det +1) yields `-R`, which stays
        orthogonal and has det -1 in 3D, hence represents a rotation composed
        with the inversion that distinguishes pseudo-tensors from tensors.
        """
        rotation = -e3nn.rand_matrix(self.key)
        node_space, edge_space, _ = module.source
        return (
            module.target.action(rotation),
            node_space.action(rotation),
            edge_space.action(rotation),
        )


class TestConvolutionLeading(_TestConvolution):
    node_irreps = "2x0e + 1o"
    edge_irreps = "0e + 1o + 2e"
    out = None
    layout = "LEADING_CHANNELS"


@pytest.mark.e3j_ops
class TestConvolutionFused(_TestConvolution):
    """Fused CUDA kernel must agree with the unfused plain-JAX reference."""

    node_irreps = "0e + 1o + 2e + 3o"
    edge_irreps = "0e + 1o + 2e + 3o"
    out = "0e + 1o + 2e + 3o"
    # The convolution kernel currently requires LHS channels in multiples of 32.
    num_channels = 32

    def assert_zero(self, data, tol: float = 5e-5):
        # Looser bound: float32 accumulation in the fused kernel.
        super().assert_zero(data, tol)

    @pytest.fixture(scope="class")
    def module(self) -> Convolution:
        with e3j.config.use(convolution="FUSED_CUDA", tensor_product="FUSED_CUDA"):
            return Convolution(
                (self.node_irreps, self.edge_irreps),
                self.out,
                layout=self.layout,
                graph_ordering="RECEIVER",
            )

    @pytest.fixture(scope="class")
    def reference(self) -> Convolution:
        with e3j.config.use(convolution="UNFUSED", tensor_product="UNFUSED"):
            return Convolution(
                (self.node_irreps, self.edge_irreps),
                self.out,
                layout=self.layout,
                graph_ordering="NONE",
            )

    def test_fused_matches_unfused(self, module, reference, inputs, graph):
        """`fused_eval` reproduces the gather/TP/mix/scatter reference."""
        senders, receivers = graph
        result = module(*inputs, senders, receivers)
        expect = reference(*inputs, senders, receivers)
        assert_allclose(expect, result, rtol=2e-3, atol=2e-3)

    def test_jit_pack_jax(self, inputs, reference, graph):
        """`coef.pack_jax()` must succeed at trace time inside fused_eval.

        Building the module within the traced function forces the packed
        Coef4D to be materialized during compilation, which only works if the
        coefficient setup is excluded from the trace via
        `jax.ensure_compile_time_eval()`.
        """
        senders, receivers = graph
        source = (self.node_irreps, self.edge_irreps)

        with e3j.config.use(convolution="FUSED_CUDA", tensor_product="FUSED_CUDA"):
            f = jax.jit(
                lambda a, b, c: Convolution(
                    source, self.out, layout=self.layout, graph_ordering="RECEIVER"
                )(a, b, c, senders, receivers)
            )
            z = f(*inputs)

        expect = reference(*inputs, senders, receivers)
        assert_allclose(expect, z, rtol=2e-3, atol=2e-3)

    @staticmethod
    def _loss(conv):
        """Quadratic loss closing over `conv`; indices are plain arguments."""

        def loss(node, edge, scalar, snd, rcv):
            return np.sum(conv(node, edge, scalar, snd, rcv) ** 2)

        return loss

    def test_grad_traced_indices(self, module, reference, inputs, graph):
        """Backward with senders/receivers passed as traced arguments.

        Every other fused test closes over the adjacency as compile-time
        constants. MLIP instead feeds the graph in as a runtime argument, so
        the fused op closes over index *tracers* rather than constants. This
        guards that path (regression for a tracer routed through the
        `custom_vjp`/`custom_partitioning` index closures) against the unfused
        reference gradient. `snd`/`rcv` are arguments (hence tracers) but not
        differentiated.
        """
        senders, receivers = graph
        argnums = (0, 1, 2)
        g_fused = jax.grad(self._loss(module), argnums=argnums)(
            *inputs, senders, receivers
        )
        g_ref = jax.grad(self._loss(reference), argnums=argnums)(
            *inputs, senders, receivers
        )
        for gf, gr in zip(g_fused, g_ref):
            assert_allclose(gf, gr, rtol=2e-3, atol=2e-3)

    @pytest.mark.xfail(
        reason="grad(jit(f)) with traced indices is unsupported. Use jit(grad(f)).",
        strict=True,
        raises=TypeError,
    )
    def test_grad_jit_traced_indices(self, module, reference, inputs, graph):
        """grad of a jitted call with traced indices."""
        senders, receivers = graph
        argnums = (0, 1, 2)
        g_fused = jax.grad(jax.jit(self._loss(module)), argnums=argnums)(
            *inputs, senders, receivers
        )
        g_ref = jax.grad(self._loss(reference), argnums=argnums)(
            *inputs, senders, receivers
        )
        for gf, gr in zip(g_fused, g_ref):
            assert_allclose(gf, gr, rtol=2e-3, atol=2e-3)


@pytest.mark.e3j_ops
class TestConvolutionFusedSender:
    """Sender-sorted fused convolution matches the receiver-sorted result.

    Builds a symmetric graph whose edge features honour the graded symmetry the
    sender trick assumes (`y_ba = p * y_ab` per slice, `s_ba = s_ab`), then
    checks the `GraphOrdering.SENDER` path (signed coefficients, no transpose,
    null `edge_perm`) against both the receiver-sorted fused path and the
    ordering-agnostic unfused reference, for the forward and its gradient.
    """

    node_irreps = "0e + 1o + 2e + 3o"
    edge_irreps = "0e + 1o + 2e + 3o"
    out = "0e + 1o + 2e + 3o"
    num_nodes = 6
    num_pairs = 12
    num_channels = 32
    layout = "TRAILING_CHANNELS"

    @pytest.fixture(scope="class")
    def modules(self) -> tuple[Convolution, Convolution, Convolution]:
        source = (self.node_irreps, self.edge_irreps)
        with e3j.config.use(convolution="FUSED_CUDA", tensor_product="FUSED_CUDA"):
            receiver = Convolution(
                source, self.out, layout=self.layout, graph_ordering="RECEIVER"
            )
            sender = Convolution(
                source, self.out, layout=self.layout, graph_ordering="SENDER"
            )
        with e3j.config.use(convolution="UNFUSED", tensor_product="UNFUSED"):
            reference = Convolution(
                source, self.out, layout=self.layout, graph_ordering="NONE"
            )
        return receiver, sender, reference

    @pytest.fixture(scope="class")
    def data(self, modules):
        """Symmetric graph in receiver- and sender-sorted orderings, plus `x`.

        The reverse of each edge picks up the per-slice parity sign on `y` and
        leaves `s` unchanged, so the two orderings describe the same messages.
        """
        receiver, _, _ = modules
        node_space, edge_space, scalar_space = receiver.source
        K = self.num_channels

        k = random.split(random.key(7), 5)
        a = random.randint(k[0], (self.num_pairs,), 0, self.num_nodes)
        off = random.randint(k[1], (self.num_pairs,), 1, self.num_nodes)
        b = (a + off) % self.num_nodes  # off in [1, num_nodes) avoids self-loops

        # Directed edges: forward pairs followed by their reverses.
        snd = np.concatenate([a, b])
        rcv = np.concatenate([b, a])

        # Graded-symmetric y and reversal-symmetric s across edge reversal.
        y_signs = np.concatenate(
            [np.full(mul * ir.dim, float(ir.p)) for mul, ir in edge_space]
        )
        y_fwd = random.normal(k[2], (self.num_pairs, edge_space.dim))
        y = np.concatenate([y_fwd, y_fwd * y_signs[None, :]])
        s_fwd = random.normal(k[3], (self.num_pairs, scalar_space.dim, K))
        s = np.concatenate([s_fwd, s_fwd])

        x = random.normal(k[4], (self.num_nodes, node_space.dim, K))

        def reorder(order):
            return snd[order], rcv[order], y[order], s[order]

        return x, reorder(np.argsort(rcv)), reorder(np.argsort(snd))

    @staticmethod
    def _loss(conv):
        def loss(x, y, s, snd, rcv):
            return np.sum(conv(x, y, s, snd, rcv) ** 2)

        return loss

    def test_forward_matches(self, modules, data):
        """SENDER path == RECEIVER path == unfused reference on the same graph."""
        receiver, sender, reference = modules
        x, recv_sorted, send_sorted = data
        snd_r, rcv_r, y_r, s_r = recv_sorted
        snd_s, rcv_s, y_s, s_s = send_sorted

        o_recv = receiver(x, y_r, s_r, snd_r, rcv_r)
        o_send = sender(x, y_s, s_s, snd_s, rcv_s)
        o_ref = reference(x, y_s, s_s, snd_s, rcv_s)
        assert_allclose(o_send, o_ref, rtol=2e-3, atol=2e-3)
        assert_allclose(o_send, o_recv, rtol=2e-3, atol=2e-3)

    def test_grad_matches(self, modules, data):
        """Gradients through the SENDER backward (no transpose, null edge_perm,
        sign propagated to coef_dx/dy/ds) match the unfused reference on the
        same sender-sorted inputs, exercising the empty `edge_perm` handler."""
        _, sender, reference = modules
        x, _, send_sorted = data
        snd_s, rcv_s, y_s, s_s = send_sorted
        argnums = (0, 1, 2)
        g_send = jax.grad(self._loss(sender), argnums=argnums)(
            x, y_s, s_s, snd_s, rcv_s
        )
        g_ref = jax.grad(self._loss(reference), argnums=argnums)(
            x, y_s, s_s, snd_s, rcv_s
        )
        for gf, gr in zip(g_send, g_ref):
            assert_allclose(gf, gr, rtol=2e-3, atol=2e-3)
