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
import pytest
from conftest import assert_allclose

import e3j
from e3j.core.convolution import Convolution
from e3j.utils.options import Layout


class _TestConvolution:
    """Base class for Convolution tests, exercising the unfused plain-JAX path."""

    in_node: str
    in_edge: str
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
        with e3j.config.use(convolution="UNFUSED", tensor_product="SPARSE"):
            return Convolution(
                (self.in_node, self.in_edge), self.out, layout=self.layout
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
        with e3j.config.use(convolution="UNFUSED", tensor_product="SPARSE"):
            scaled = Convolution(
                (self.in_node, self.in_edge),
                self.out,
                layout=self.layout,
                avg_num_neighbors=avg,
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
    in_node = "2x0e + 1o"
    in_edge = "0e + 1o + 2e"
    out = None


class TestConvolution_filtered(_TestConvolution):
    in_node = "2x0e + 2x1o"
    in_edge = "0e + 1o + 2e"
    out = "0e + 1o + 2e"


class TestConvolutionLeading(_TestConvolution):
    in_node = "2x0e + 1o"
    in_edge = "0e + 1o + 2e"
    out = None
    layout = "LEADING_CHANNELS"
