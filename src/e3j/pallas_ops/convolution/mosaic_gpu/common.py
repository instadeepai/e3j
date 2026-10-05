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

"""Helpers shared by the forward and backward Mosaic GPU convolution kernels.

This covers shape checks, choosing the channel block size, and building the
CSR view of the edge list.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

from e3j.pallas_ops.convolution.mosaic_gpu.params import (
    PallasMosaicGPUMessagePassingConvolutionParams,
)

# These live in a shared utils module because the tensor product kernels use
# them too (`used_input_coordinates`, for instance, only looks at the
# Clebsch-Gordan indices). They are re-exported here so the kernels in this
# package can import everything from one place.
from e3j.pallas_ops.utils.mosaic_gpu import (  # noqa: F401
    MAX_COPY_ELEMS_PER_AXIS,
    UNTILED_COPY_MIN_BYTES,
    WARPGROUP_LANES,
    shared_memory_budget,
    used_input_coordinates,
)
from e3j.utils import options

#: Name of the grid axis. The kernels launch one CTA per node.
NODE_AXIS = "node"


class Shapes(NamedTuple):
    """Sizes of one convolution, known at trace time."""

    n_nodes: int
    n_edges: int
    x_dim: int
    y_dim: int
    y_dim_padded: int
    z_dim: int
    num_scalars: int
    channels: int


def validate_operands(
    node_features: jax.Array,
    edge_features: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    params: PallasMosaicGPUMessagePassingConvolutionParams,
) -> Shapes:
    """Check that the operands are consistent and that the kernels support them.

    The coordinate axes copied into shared memory are limited to
    `MAX_COPY_ELEMS_PER_AXIS`. The check includes `z_dim`, which only the
    backward copies, so any shape that passes here can also be differentiated.

    Raises:
        ValueError: If the shapes don't match each other or `params`.
        NotImplementedError: If a dtype is not `float32`, the channel count is
            not a multiple of `WARPGROUP_LANES`, or a coordinate axis is too
            wide to copy in one go.
    """
    if node_features.ndim != 3:
        raise ValueError(
            f"node_features must be (n_nodes, x_dim, channels), "
            f"got {node_features.shape}"
        )
    if edge_features.ndim != 2:
        raise ValueError(
            f"edge_features must be (n_edges, y_dim), got {edge_features.shape}"
        )
    if edge_scalars.ndim != 3:
        raise ValueError(
            f"edge_scalars must be (n_edges, num_scalars, channels), "
            f"got {edge_scalars.shape}"
        )
    n_nodes, x_dim, channels = node_features.shape
    n_edges, y_dim = edge_features.shape
    if edge_scalars.shape[0] != n_edges:
        raise ValueError(
            f"edge_features and edge_scalars disagree on the edge count: "
            f"{n_edges} vs {edge_scalars.shape[0]}"
        )
    if edge_scalars.shape[2] != channels:
        raise ValueError(
            f"node_features and edge_scalars disagree on the channel count: "
            f"{channels} vs {edge_scalars.shape[2]}"
        )
    if senders.shape != (n_edges,) or receivers.shape != (n_edges,):
        raise ValueError(
            f"senders and receivers must both be ({n_edges},), got "
            f"{senders.shape} and {receivers.shape}"
        )
    if x_dim != params.x_space.dim:
        raise ValueError(
            f"node_features has x_dim={x_dim}, but x_space is {params.x_space} "
            f"of dim {params.x_space.dim}"
        )
    if y_dim != params.y_space.dim:
        raise ValueError(
            f"edge_features has y_dim={y_dim}, but y_space is {params.y_space} "
            f"of dim {params.y_space.dim}"
        )
    if edge_scalars.shape[1] != params.num_scalars:
        raise ValueError(
            f"edge_scalars carries {edge_scalars.shape[1]} scalars, but output "
            f"space {params.z_space} has {params.num_scalars} irreps"
        )
    for name, operand in (
        ("node_features", node_features),
        ("edge_features", edge_features),
        ("edge_scalars", edge_scalars),
    ):
        if operand.dtype != jnp.float32:
            raise NotImplementedError(
                f"the Mosaic GPU convolution is float32-only, {name} is "
                f"{operand.dtype}"
            )
    if channels < WARPGROUP_LANES or channels % WARPGROUP_LANES:
        raise NotImplementedError(
            f"channels={channels} must be a positive multiple of "
            f"{WARPGROUP_LANES}: the kernel loads (channels,) rows from shared "
            f"memory, and Mosaic GPU can only do that in multiples of "
            f"{WARPGROUP_LANES}"
        )

    elems = UNTILED_COPY_MIN_BYTES // edge_features.dtype.itemsize
    y_dim_padded = -(-y_dim // elems) * elems
    for name, dim, staged in (
        ("x_space", x_dim, x_dim),
        ("y_space", y_dim, y_dim_padded),
        ("the output space's irreps (edge scalars)", params.num_scalars, None),
        ("z_space", params.z_space.dim, None),
    ):
        _check_staged_axis(name, dim, staged)
    return Shapes(
        n_nodes=n_nodes,
        n_edges=n_edges,
        x_dim=x_dim,
        y_dim=y_dim,
        y_dim_padded=y_dim_padded,
        z_dim=params.z_space.dim,
        num_scalars=params.num_scalars,
        channels=channels,
    )


def _check_staged_axis(name: str, dim: int, staged: int | None) -> None:
    width = dim if staged is None else staged
    if width <= MAX_COPY_ELEMS_PER_AXIS:
        return
    padded = "" if staged in (None, dim) else f" (staged as {staged})"
    raise NotImplementedError(
        f"{name} has {dim} coordinates{padded}, but an async copy can move at "
        f"most {MAX_COPY_ELEMS_PER_AXIS} elements along one dimension. The "
        f"kernels copy a whole row of this space at once; supporting wider "
        f"spaces would need splitting the coordinate axis into blocks, which "
        f"is not implemented"
    )


#: Largest `l_max` that still gets the wide automatic `channel_block`. Above it
#: the block drops to `WARPGROUP_LANES`, because the unrolled Clebsch-Gordan
#: code, and the shared memory and registers it needs per edge, grow quickly
#: with `l_max`.
_WIDE_CHANNEL_BLOCK_MAX_LMAX = 2

#: The automatic `channel_block` for the cheaper `l_max <= 2` convolutions.
_WIDE_CHANNEL_BLOCK = 256


def _default_channel_block(channels: int, l_max: int) -> int:
    """Pick a block size from `l_max`, falling back to `WARPGROUP_LANES`.

    The fallback is used when the preferred size doesn't divide `channels`.
    """
    preferred = (
        WARPGROUP_LANES if l_max > _WIDE_CHANNEL_BLOCK_MAX_LMAX else _WIDE_CHANNEL_BLOCK
    )
    return preferred if channels % preferred == 0 else WARPGROUP_LANES


def resolve_channel_block(channels: int, channel_block: int | None, l_max: int) -> int:
    """Return the `channel_block` to use, choosing one if it is `None`.

    Raises:
        ValueError: If `channel_block` does not divide `channels`, is not a
            multiple of `WARPGROUP_LANES`, or exceeds `MAX_COPY_ELEMS_PER_AXIS`.
    """
    if channel_block is None:
        channel_block = _default_channel_block(channels, l_max)
    if channels % channel_block:
        raise ValueError(
            f"channel_block={channel_block} must divide channels={channels}"
        )
    if channel_block % WARPGROUP_LANES:
        raise ValueError(
            f"channel_block={channel_block} must be a multiple of "
            f"{WARPGROUP_LANES}, the smallest row Mosaic GPU can vectorize"
        )
    if channel_block > MAX_COPY_ELEMS_PER_AXIS:
        raise ValueError(
            f"channel_block={channel_block} is larger than "
            f"{MAX_COPY_ELEMS_PER_AXIS}, the most an async copy can move along "
            f"one axis; use a smaller block and the channels will be processed "
            f"in several passes"
        )
    return channel_block


#: The number of rows in the backward's `dy` staging tile is rounded up to a
#: multiple of this (see `bwd._BackwardPlan.dy_tile_rows`).
DY_TILE_ROW_GRANULARITY = 8


def staged_smem_bytes(
    shapes: Shapes,
    params: PallasMosaicGPUMessagePassingConvolutionParams,
    channel_block: int,
    *,
    backward: bool,
    itemsize: int = 4,
) -> int:
    """Estimate the shared memory one CTA needs at the given `channel_block`.

    This is a lower bound: it leaves out barriers and alignment padding. That
    way it never rejects a configuration that would actually fit.
    """
    cb = channel_block
    edge_slot = shapes.x_dim * cb + shapes.y_dim_padded + shapes.num_scalars * cb
    if not backward:
        return params.fwd_config.pipeline_depth * edge_slot * itemsize
    dy_tile_rows = -(-shapes.y_dim // DY_TILE_ROW_GRANULARITY) * DY_TILE_ROW_GRANULARITY
    node_rows = (shapes.z_dim + dy_tile_rows) * cb
    if params.bwd_grouping == options.GraphOrdering.RECEIVER:
        depth = params.bwd_config.pipeline_depth
        staged = node_rows + shapes.x_dim * cb + depth * edge_slot
    else:
        staged = node_rows + edge_slot
    return staged * itemsize


def plan_channel_block(
    shapes: Shapes,
    params: PallasMosaicGPUMessagePassingConvolutionParams,
    *,
    backward: bool,
) -> int:
    """Choose the forward or backward `channel_block` and check it fits.

    If the block was chosen automatically and the wide one doesn't fit in
    shared memory, it falls back to `WARPGROUP_LANES`. A block set explicitly
    in the config is never changed.

    Raises:
        ValueError: If `channel_block` is invalid, or the data the kernel
            needs to hold in shared memory doesn't fit.
    """
    config = params.bwd_config if backward else params.fwd_config
    channel_block = resolve_channel_block(
        shapes.channels, config.channel_block, params.l_max
    )
    budget = shared_memory_budget()

    def staged(cb: int) -> int:
        return staged_smem_bytes(shapes, params, cb, backward=backward)

    if (
        config.channel_block is None
        and channel_block > WARPGROUP_LANES
        and staged(channel_block) > budget
    ):
        channel_block = WARPGROUP_LANES
    if staged(channel_block) > budget:
        pass_name = "backward" if backward else "forward"
        config_name = "bwd_config" if backward else "fwd_config"
        raise ValueError(
            f"the {pass_name} pass needs at least {staged(channel_block)} bytes "
            f"of shared memory per CTA at channel_block={channel_block}, but "
            f"only {budget} are available; lower {config_name}.pipeline_depth "
            f"or {config_name}.channel_block"
        )
    return channel_block


def pad_edge_features(edge_features: jax.Array, y_dim_padded: int) -> jax.Array:
    """Pad `edge_features` with zero columns up to `y_dim_padded`.

    TMA copies need rows of a minimum size, so short rows are padded.
    """
    columns = y_dim_padded - edge_features.shape[-1]
    if not columns:
        return edge_features
    return jnp.pad(edge_features, ((0, 0), (0, columns)))


def csr_endpoints(
    senders: jax.Array,
    receivers: jax.Array,
    group_by: options.GraphOrdering,
) -> tuple[jax.Array, jax.Array]:
    """Split senders and receivers by role for a walk grouped by `group_by`.

    Returns:
        `(group_index, gather_index)`: the endpoint edges are grouped by, and
        the other endpoint, whose features each edge reads.
    """
    if group_by == options.GraphOrdering.SENDER:
        return senders, receivers
    return receivers, senders


def csr_offsets(group_index: jax.Array, n_nodes: int) -> jax.Array:
    """Compute CSR offsets, of shape `(n_nodes + 1,)`.

    Node `b`'s edges are `offsets[b]:offsets[b + 1]`. These ranges only hold
    the right edges if `group_index` is sorted. We count edges per node and
    take a cumulative sum, which is about 3x faster than `jnp.searchsorted`.
    """
    degrees = (
        jnp.zeros((n_nodes + 1,), jnp.int32).at[group_index + 1].add(1, mode="drop")
    )
    return jnp.cumsum(degrees, dtype=jnp.int32)


def csr_view(
    senders: jax.Array,
    receivers: jax.Array,
    n_nodes: int,
    group_by: options.GraphOrdering,
    sorted_by: options.GraphOrdering | None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Build a CSR view of the edge list, grouped by `group_by`.

    This works whatever order the edges are in. Node `n` walks the positions
    `k` in `offsets[n]:offsets[n + 1]`; at position `k` it processes edge
    `edge_perm[k]` and reads the features of node `gather_index[k]`.

    Returns:
        `(offsets, gather_index, edge_perm)`. If the edges are already sorted
        by `group_by`, no reordering is needed and `edge_perm` is `None`.
    """
    group_index, gather_index = csr_endpoints(senders, receivers, group_by)
    offsets = csr_offsets(group_index, n_nodes)
    if group_by == sorted_by:
        return offsets, gather_index, None
    edge_perm = jnp.argsort(group_index).astype(jnp.int32)
    return offsets, gather_index[edge_perm], edge_perm


def edge_slot(slot_gmem, position):
    """Return the edge at `position` of a CSR walk (see :func:`csr_view`)."""
    return position if slot_gmem is None else plgpu.load(slot_gmem.at[position])
