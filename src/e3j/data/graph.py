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


import jax.numpy as jnp
import numpy
from jax import Array
from numpy import int32

DUMMY_INDEX = int32(numpy.iinfo(int32).max)

#: Index dtype the CUDA FFI handlers declare. Pinned because `jax_enable_x64`,
#: which float64 values require, would default derived buffers to int64.
INDEX_DTYPE = jnp.int32


def is_dummy_index(index: Array) -> Array:
    """Return where a single endpoint carries the `DUMMY_INDEX` sentinel."""
    return index == DUMMY_INDEX


def is_dummy_edge(sender: Array, receiver: Array) -> Array:
    """Return where either endpoint carries the `DUMMY_INDEX` sentinel."""
    return is_dummy_index(sender) | is_dummy_index(receiver)


class GraphCSR:
    """Compressed Sparse Row (CSR) directed-graph adjacency.

    Edges are grouped by receiver so the convolution kernel can reduce
    messages per receiver without atomics.

    Note:
        Derived buffers (`receiver_ptr` and the permutations returned by `sort`
        and `transpose`) are always :data:`INDEX_DTYPE`, under x32 and x64 alike.
    """

    def __init__(self, num_nodes: int, sender: Array, receiver: Array):
        self.num_nodes = num_nodes
        self.sender = sender
        self.receiver = receiver
        self.num_neighbors = jnp.bincount(receiver, length=num_nodes)
        self.receiver_ptr = jnp.append(0, jnp.cumsum(self.num_neighbors)).astype(
            INDEX_DTYPE
        )

    @classmethod
    def sort(
        cls, num_nodes: int, sender: Array, receiver: Array
    ) -> tuple[Array, "GraphCSR", Array]:
        """Return (sigma, graph', sigma_1) sorting edges by receivers."""
        perm = jnp.argsort(receiver).astype(INDEX_DTYPE)
        graph_sorted = cls(num_nodes, sender[perm], receiver[perm])
        return perm, graph_sorted, jnp.argsort(perm).astype(INDEX_DTYPE)

    def transpose(self) -> tuple[Array, "GraphCSR", Array]:
        """Return (sigma, graph_t, sigma_1) sorting edges by senders instead."""
        return GraphCSR.sort(
            self.num_nodes,
            self.receiver,
            self.sender,
        )

    @staticmethod
    def mask_edges(
        sender: Array, receiver: Array, node_mask: Array
    ) -> tuple[Array, Array]:
        """Mark edges touching a padding node with `DUMMY_INDEX`.

        Padding edges are assumed to only connect trailing nodes and must lie
        at the end of the graph. They are assigned out-of-bounds edges to ensure
        they are skipped by the message aggregation. Trailing padding makes the
        real nodes those below `count_nonzero(node_mask)`.
        """
        num_real_nodes = jnp.count_nonzero(node_mask)
        edge_mask = (sender < num_real_nodes) & (receiver < num_real_nodes)
        return (
            jnp.where(edge_mask, sender, DUMMY_INDEX),
            jnp.where(edge_mask, receiver, DUMMY_INDEX),
        )
