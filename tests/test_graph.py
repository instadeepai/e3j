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

import jax.numpy as np
import jax.random as random
import numpy
import pytest
from conftest import assert_allclose

import e3j
from e3j.core.convolution import Convolution
from e3j.data.graph import GraphCSR
from e3j.ops.convolution import DUMMY_INDEX, _wrap_global_index

# A small receiver-sorted graph over `num_nodes` nodes, plus `num_dummy`
# padding edges. All dummy endpoints carry `DUMMY_INDEX` so they should be
# excluded from every CSR adjacency.
num_nodes = 4
sender_real = np.array([1, 2, 0, 1, 3, 2], dtype=np.int32)
receiver_real = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)
num_real = sender_real.shape[0]
num_dummy = 3


def _padded_graph():
    """Real edges followed by `num_dummy` `DUMMY_INDEX`-marked padding edges."""
    dummy = np.full(num_dummy, DUMMY_INDEX, dtype=np.int32)
    sender = np.concatenate([sender_real, dummy])
    receiver = np.concatenate([receiver_real, dummy])
    return sender, receiver


class TestWrapGlobalIndex:
    """`_wrap_global_index` reduces indices modulo `num_nodes` but preserves the
    padding sentinel so it stays out of range for `bincount`."""

    def test_preserves_sentinel(self):
        index = np.array([0, 3, DUMMY_INDEX], dtype=np.int32)
        wrapped = _wrap_global_index(index, num_nodes)
        assert int(wrapped[-1]) == int(DUMMY_INDEX)

    def test_wraps_in_range(self):
        # Global indices from a folded second graph wrap back to local nodes.
        index = np.array([num_nodes, num_nodes + 1], dtype=np.int32)
        wrapped = _wrap_global_index(index, num_nodes)
        numpy.testing.assert_array_equal(wrapped, np.array([0, 1]))


class TestDummyEdgesCSR:
    """Padding edges marked `DUMMY_INDEX` drop out of the CSR adjacency, leaving
    the real-edge ranges unchanged in both forward and transposed directions."""

    def _reference_ptr(self, receiver):
        return np.append(0, np.cumsum(np.bincount(receiver, length=num_nodes)))

    def test_forward_drops_dummies(self):
        sender, receiver = _padded_graph()
        sender_local = _wrap_global_index(sender, num_nodes)
        receiver_local = _wrap_global_index(receiver, num_nodes)
        ptr = GraphCSR(num_nodes, sender_local, receiver_local).receiver_ptr

        # Ranges match the unpadded graph; the final pointer excludes dummies.
        numpy.testing.assert_array_equal(ptr, self._reference_ptr(receiver_real))
        assert int(ptr[-1]) == num_real

    def test_transpose_drops_dummies(self):
        sender, receiver = _padded_graph()
        sender_local = _wrap_global_index(sender, num_nodes)
        receiver_local = _wrap_global_index(receiver, num_nodes)
        _, graph_t, _ = GraphCSR(num_nodes, sender_local, receiver_local).transpose()

        # The backward CSR buckets by sender; dummies drop there too.
        numpy.testing.assert_array_equal(
            graph_t.receiver_ptr, self._reference_ptr(sender_real)
        )
        assert int(graph_t.receiver_ptr[-1]) == num_real


class TestUnfusedConvolutionDummyEdges:
    """The plain-JAX (unfused) path treats `DUMMY_INDEX` edges as no-ops: the
    out-of-bounds gather clamps and the scatter-add drops, so padded output
    matches the unpadded graph regardless of the dummy edge data."""

    @pytest.fixture(scope="class")
    def module(self):
        with e3j.config.use(convolution="UNFUSED", tensor_product="UNFUSED"):
            return Convolution(
                ("2x0e + 1x1o", "1x0e + 1x1o"),
                None,
                layout="TRAILING_CHANNELS",
                graph_ordering="NONE",
            )

    def test_padded_matches_unpadded(self, module):
        num_channels = 4
        node_space, edge_space, scalar_space = module.source
        k = random.split(random.key(0), 5)
        node_features = random.normal(k[0], (num_nodes, node_space.dim, num_channels))
        edge_features = random.normal(k[1], (num_real, edge_space.dim))
        edge_scalars = random.normal(k[2], (num_real, scalar_space.dim, num_channels))
        reference = module(
            node_features, edge_features, edge_scalars, sender_real, receiver_real
        )

        # Pad with dummy edges carrying arbitrary (non-zero) feature data.
        sender, receiver = _padded_graph()
        edge_features_pad = np.concatenate(
            [edge_features, random.normal(k[3], (num_dummy, edge_space.dim))]
        )
        edge_scalars_pad = np.concatenate(
            [
                edge_scalars,
                random.normal(k[4], (num_dummy, scalar_space.dim, num_channels)),
            ]
        )
        padded = module(
            node_features, edge_features_pad, edge_scalars_pad, sender, receiver
        )
        assert_allclose(reference, padded)
