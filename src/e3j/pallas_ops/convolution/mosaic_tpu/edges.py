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

"""Edge-sized operands of the fused MP-conv fwd/bwd kernels."""

import jax.numpy as jnp
from jax import Array, lax
from jax.experimental import pallas as pl


def edge_scalars_edge_major(num_blocks: int, channels: int) -> bool:
    """Whether XLA's TPU layout keeps `(n_edges, num_blocks, channels)` edge-major."""
    return channels % 128 == 0 and (num_blocks in (1, 2, 4) or num_blocks % 8 == 0)


def pad_to_one_block(array: Array, axis: int, edge_block_size: int) -> Array:
    """Zero-pad the edge `axis` of a graph smaller than one edge block.

    Mosaic cannot slice a block out of an operand of a few edges.
    """
    missing = edge_block_size - array.shape[axis]
    if missing <= 0:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, missing)
    return jnp.pad(array, pad_width)


def edge_scalars_spec(
    edge_major: bool,
    edge_block_size: int,
    num_blocks: int,
    channel_block: int,
    walked_block,
    stride,
) -> pl.BlockSpec:
    """Pipeline block of the edge scalars (and of their cotangent)."""
    if edge_major:
        return pl.BlockSpec(
            (edge_block_size, num_blocks, channel_block),
            lambda i, s=stride: (walked_block(i), 0, s),
        )
    return pl.BlockSpec(
        (num_blocks, edge_block_size, channel_block),
        lambda i, s=stride: (0, walked_block(i), s),
    )


def load_edge_scalars(edge_major: bool, ref, block: int) -> Array:
    """The `(edges, channels)` tile of irrep block `block`."""
    return ref[:, block, :] if edge_major else ref[block, :, :]


def store_edge_scalars(edge_major: bool, ref, block: int, tile: Array) -> None:
    if edge_major:
        ref[:, block, :] = tile
    else:
        ref[block, :, :] = tile


def walked_edge_block(
    core_id: Array,
    step: Array,
    n_walked_blocks: Array,
    num_cores: int,
    block_order_smem,
) -> Array:
    """Edge block of a core's `step`-th grid step under `emit_pipeline(core_axis=0)`."""
    base, remainder = jnp.divmod(n_walked_blocks, num_cores)
    first_step = jnp.where(
        core_id < remainder,
        core_id * (base + 1),
        core_id * base + remainder,
    )
    grid_step = first_step + step
    return grid_step if block_order_smem is None else block_order_smem[grid_step]


def mask_ragged_rows(tile: Array, block: Array, n_edges: int) -> Array:
    """Zero the rows of an edge-major tile past `n_edges` (stale VMEM in the last block)."""
    edge_block_size = tile.shape[0]
    if n_edges % edge_block_size == 0:
        return tile
    row = lax.broadcasted_iota(jnp.int32, tile.shape, 0)
    return jnp.where(row < n_edges - block * edge_block_size, tile, 0)
