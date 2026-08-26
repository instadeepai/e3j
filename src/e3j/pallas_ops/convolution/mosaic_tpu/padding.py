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

"""Routing of `DUMMY_INDEX`-marked padding edges for the Mosaic TPU kernels.

The CUDA path drops padding edges out of the CSR adjacency (see
`GraphCSR.mask_edges`). These kernels walk a blocked, sorted edge list instead,
so they route padding onto an appended zero node and bound the edge pipeline by
the last real edge.
"""

import jax.numpy as jnp
from jax import Array, lax

from e3j.data.graph import is_dummy_edge


def block_order(edge_is_real: Array, n_blocks: int, batch_block_size: int) -> tuple:
    """Return the edge blocks with those holding a real edge first, and how many."""
    is_real_per_slot = jnp.pad(
        edge_is_real, (0, n_blocks * batch_block_size - edge_is_real.size)
    )
    block_holds_real_edge = is_real_per_slot.reshape(n_blocks, batch_block_size).any(
        axis=1
    )
    return (
        jnp.argsort(~block_holds_real_edge, stable=True).astype(jnp.int32),
        jnp.count_nonzero(block_holds_real_edge),
    )


def route_dummy_edges(
    node_features: Array, group_key: Array, gather_index: Array
) -> tuple[Array, Array, Array, Array]:
    """Hold dummy edges in their sorted key group, gathering from an appended zero node.

    Returns:
        * `node_features`: the input with one zero row appended at `num_nodes`,
        * `group_key`: each dummy edge reuses the previous key, never splitting a group,
        * `gather_index`: dummy edges point at the zero row, so their message is zero,
        * `edge_is_real`: per-edge mask, folding with the edge axis unlike a bound.
    """
    num_nodes = node_features.shape[0]
    dummy = is_dummy_edge(group_key, gather_index)
    pad_width = ((0, 1),) + ((0, 0),) * (node_features.ndim - 1)
    return (
        jnp.pad(node_features, pad_width),
        lax.cummax(jnp.where(dummy, 0, group_key)),
        jnp.where(dummy, num_nodes, gather_index),
        ~dummy,
    )
