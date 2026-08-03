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

# Sentinel index marking a padding ("dummy") edge. Edges whose sender or
# receiver equals this value fall outside [0, num_nodes) and are dropped from
# the CSR adjacency by `jnp.bincount` (which ignores out-of-range indices), so
# the kernel does no work on them. Callers batching padded graphs mark dummy
# edges with `where(edge_mask, index, DUMMY_INDEX)`.
#
# NOTE: valid on a single pre-batched (disjoint) graph. Under vmap folding a
#       node offset is added before the local wrap, overflowing the sentinel;
#       apply the marker after folding in that case.
DUMMY_INDEX = int32(numpy.iinfo(int32).max)


class GraphCSR:
    """Compressed Sparse Row (CSR) directed-graph adjacency.

    Edges are grouped by receiver so the convolution kernel can reduce
    messages per receiver without atomics.

    Beyond construction and transposition, this class owns index arithmetic
    required by `vmap` rules to fold a batch axis into a single disjoint
    (block-diagonal) graph, both when a single graph is replicated across
    devices or when distinct homogeneous graphs are stacked.
    """

    def __init__(self, num_nodes: int, sender: Array, receiver: Array):
        self.num_nodes = num_nodes
        self.sender = sender
        self.receiver = receiver
        self.num_neighbors = jnp.bincount(receiver, length=num_nodes)
        self.receiver_ptr = jnp.append(0, jnp.cumsum(self.num_neighbors))

    @classmethod
    def sort(
        cls, num_nodes: int, sender: Array, receiver: Array
    ) -> tuple[Array, "GraphCSR", Array]:
        """Return (sigma, graph', sigma_1) sorting edges by receivers."""
        perm = jnp.argsort(receiver)
        graph_sorted = cls(num_nodes, sender[perm], receiver[perm])
        return perm, graph_sorted, jnp.argsort(perm)

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

        Edges with a masked (padding) endpoint are set to the out-of-range
        sentinel so every path skips them: the CUDA kernel drops them from the
        CSR adjacency, the plain-JAX gather/scatter clamps and discards.
        """
        edge_mask = node_mask[sender] & node_mask[receiver]
        return (
            jnp.where(edge_mask, sender, DUMMY_INDEX),
            jnp.where(edge_mask, receiver, DUMMY_INDEX),
        )

    @staticmethod
    def fold_adjacency(
        axis_size: int,
        num_nodes: int,
        num_edges: int,
        sender: Array,
        receiver_ptr: Array,
        batched: bool,
    ) -> tuple[Array, Array]:
        """Fold a batch axis of `(sender, receiver_ptr)` into one disjoint graph.

        Args:
            axis_size: Size of the leading batch axis.
            num_nodes: Number of nodes in one graph.
            num_edges: Number of edges in one graph.
            sender: Sender node indices, relative to each graph.
            receiver_ptr: Receiver CSR pointers, relative to each graph.
            batched: When false, replicates a single graph for SPMD. If
                true, distinct graphs are batched and `axis_size` should
                match the leading dimensions of `sender` and `receiver_ptr`.

        Returns:
            A `(sender, receiver_ptr)` pair representing a graph with
            `axis_size * num_nodes` nodes and `axis_size * num_edges` edges.
        """
        if not batched:
            sender = jnp.broadcast_to(sender, (axis_size,) + sender.shape)
            receiver_ptr = jnp.broadcast_to(
                receiver_ptr, (axis_size,) + receiver_ptr.shape
            )
        node_offsets = jnp.arange(axis_size, dtype=sender.dtype)[:, None] * num_nodes
        edge_offsets = (
            jnp.arange(axis_size, dtype=receiver_ptr.dtype)[:, None] * num_edges
        )
        sender_folded = (sender + node_offsets).reshape(-1)
        ptr_body = (receiver_ptr[:, :num_nodes] + edge_offsets).reshape(-1)
        receiver_ptr_folded = jnp.append(ptr_body, axis_size * num_edges)
        return sender_folded, receiver_ptr_folded

    @staticmethod
    def fold_permutation(
        axis_size: int, num_edges: int, perm: Array, batched: bool
    ) -> Array:
        """Fold a batch axis of an edge permutation into one disjoint graph.

        Offsets each element by `b * num_edges` so it indexes the correct
        segment of the concatenated edge array.
        """
        if not batched:
            perm = jnp.broadcast_to(perm, (axis_size,) + perm.shape)
        edge_offsets = jnp.arange(axis_size, dtype=perm.dtype)[:, None] * num_edges
        return (perm + edge_offsets).reshape(-1)
