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

"""Mosaic TPU helpers shared across Pallas kernels."""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Block sizes (in batch elements) a Mosaic TPU pipeline may tile over. Kept as
# powers of two so blocks evenly partition typical batch sizes.
DEFAULT_BATCH_BLOCK_SIZE_CANDIDATES = [1, 2, 4, 8, 16, 32]


def round_up(n: int, multiple: int) -> int:
    """Smallest multiple of `multiple` that is >= `n`."""
    return pl.cdiv(n, multiple) * multiple


def tiled_bytes(
    shapes: Iterable[Sequence[int]], num_lanes: int, num_sublanes: int, itemsize: int
) -> int:
    """Bytes these buffers occupy once their last two axes are padded to a tile."""
    total = 0
    for shape in shapes:
        lanes = round_up(shape[-1], num_lanes)
        sublanes = round_up(shape[-2], num_sublanes) if len(shape) > 1 else 1
        total += math.prod(shape[:-2]) * sublanes * lanes * itemsize
    return total


VMEM_SPILL = 1.3  # slack for what the shapes miss: spills, staging, temporaries
VMEM_SPILL_WIDE = 1.8  # a wide slice keeps more of them live at once
WIDE_CHANNEL_TILES = 3  # lane tiles per slice from which the wide slack applies


def vmem_step_bytes(
    scratch: Iterable[Sequence[int]],
    operands: Iterable[Sequence[int]],
    values: Iterable[Sequence[int]],
    *,
    channel_tiles: int,
    num_lanes: int,
    num_sublanes: int,
    itemsize: int,
) -> int:
    """VMEM one pipeline step holds: scratch, double-buffered operands, live values."""
    slack = VMEM_SPILL_WIDE if channel_tiles >= WIDE_CHANNEL_TILES else VMEM_SPILL
    counted = (
        tiled_bytes(scratch, num_lanes, num_sublanes, itemsize)
        + 2 * tiled_bytes(operands, num_lanes, num_sublanes, itemsize)
        + tiled_bytes(values, num_lanes, num_sublanes, itemsize)
    )
    return int(counted * slack)


@dataclass(frozen=True)
class BlockPlan:
    """Edge and channel block a Mosaic TPU kernel tiles one pipeline step with."""

    edge_block_size: int
    channel_block: int
    channels_padded: int
    num_lanes: int

    @property
    def n_channel_strides(self) -> int:
        """Passes over the edge list this plan implies."""
        return self.channels_padded // self.channel_block

    @property
    def channel_tiles(self) -> int:
        """Lane tiles one pass carries."""
        return max(1, self.channel_block // self.num_lanes)

    def channel_slice(self, stride: int) -> pl.Slice:
        """The channels pass `stride` owns."""
        return pl.ds(stride * self.channel_block, self.channel_block)


#: Block planning ceiling: a chip can report more VMEM than the scoped limit.
MAX_PLANNED_VMEM_BYTES = 32 * 1024 * 1024


def plan_blocks(
    *,
    channels_padded: int,
    num_lanes: int,
    vmem_bytes: Callable[[int, int], int],
    edge_block_sizes: Sequence[int],
    min_edge_block_size: int,
    min_strided_edge_block_size: int,
    vmem_capacity_bytes: int | None = None,
    edge_block_size: int | None = None,
    channel_block: int | None = None,
) -> BlockPlan:
    """Widest channel slice whose largest VMEM-fitting edge tile clears the floor.

    The floor is `min_edge_block_size` while one pass still carries every channel, and
    `min_strided_edge_block_size` once they are split over several. If no width clears
    it, the narrowest slice comes back with whatever fits there.

    Args:
        channels_padded: Lane-padded channel count all the passes cover together.
        num_lanes: Lanes per vector register; channel slices are multiples of it.
        vmem_bytes: `(edge_block_size, channel_block) -> bytes` estimate of what one
            pipeline step needs. Kernel-specific, so the caller supplies it.
        edge_block_sizes: Edge tile sizes to choose from. A kernel lists only the ones
            it can run, so alignment rules live here.
        min_edge_block_size: Smallest edge tile worth keeping when one pass carries
            every channel.
        min_strided_edge_block_size: Smallest edge tile worth keeping once the channels
            are split over several passes. Measured higher than `min_edge_block_size` on
            the backward, which would rather spend the freed VMEM on a wider tile
            than walk the edge list again.
        vmem_capacity_bytes: VMEM budget per core. Defaults to this device's,
            clamped to `MAX_PLANNED_VMEM_BYTES`.
        edge_block_size: Use this edge tile as given, skipping the search and the
            VMEM check.
        channel_block: Use this channel slice as given, skipping the search.

    Returns:
        The chosen `BlockPlan`.
    """
    if vmem_capacity_bytes is None:
        vmem_capacity_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    vmem_capacity_bytes = min(vmem_capacity_bytes, MAX_PLANNED_VMEM_BYTES)
    tile = min(channels_padded, num_lanes)
    n_tiles = channels_padded // tile
    widths = (
        [channel_block]
        if channel_block is not None
        else [t * tile for t in range(n_tiles, 0, -1) if n_tiles % t == 0]
    )

    def largest_fitting(width: int) -> int | None:
        if edge_block_size is not None:
            return edge_block_size
        fits = [
            b for b in edge_block_sizes if vmem_bytes(b, width) <= vmem_capacity_bytes
        ]
        return max(fits) if fits else None

    for width in widths:
        block = largest_fitting(width)
        floor = (
            min_edge_block_size
            if width == channels_padded
            else min_strided_edge_block_size
        )
        if block is not None and block >= floor:
            return BlockPlan(block, width, channels_padded, num_lanes)
    return BlockPlan(
        largest_fitting(widths[-1]) or min(edge_block_sizes),
        widths[-1],
        channels_padded,
        num_lanes,
    )


def select_batch_block_size(
    batch_size: int,
    vmem_bytes_per_batch_element: int,
    *,
    batch_block_size_candidates: list[int] | None = None,
    override: int | None = None,
    num_kernel: int | None = None,
    vmem_capacity_bytes: int | None = None,
    allow_padding: bool = False,
) -> int:
    """Pick how many batch elements to process per Mosaic TPU pipeline block.

    Chooses the largest candidate block size that fits in VMEM and evenly
    divides `batch_size`, so the pipeline tiles the batch axis without a
    ragged remainder. The resulting number of blocks must also be divisible by
    `num_kernel` so the megacore split partitions blocks evenly across cores.

    Args:
        batch_size: Total number of batch elements the kernel processes.
        vmem_bytes_per_batch_element: VMEM footprint of a single batch element;
            used to bound the block size against `vmem_capacity_bytes`.
        batch_block_size_candidates: Allowed block sizes to choose from. Defaults
            to `DEFAULT_BATCH_BLOCK_SIZE_CANDIDATES`.
        override: If given, use this block size directly and skip the VMEM-based
            selection (divisibility assertions still apply).
        num_kernel: Number of cores the work is split across. Defaults to the
            TPU's core count.
        vmem_capacity_bytes: VMEM budget per core. Defaults to the TPU's VMEM
            capacity.
        allow_padding: If set, the caller pads `batch_size` up to a multiple of
            the block, so the largest VMEM-fitting candidate is chosen without
            requiring divisibility and the divisibility assertions are skipped.

    Returns:
        The selected batch block size (in batch elements).
    """
    if num_kernel is None or vmem_capacity_bytes is None:
        tpu_info = pltpu.get_tpu_info()
        num_kernel = num_kernel or tpu_info.num_cores
        vmem_capacity_bytes = vmem_capacity_bytes or tpu_info.vmem_capacity_bytes

    batch_block_size_candidates = (
        batch_block_size_candidates or DEFAULT_BATCH_BLOCK_SIZE_CANDIDATES
    )
    if override is not None:
        batch_block_size = override
    else:
        max_bbs = vmem_capacity_bytes // vmem_bytes_per_batch_element
        fits = [bbs for bbs in batch_block_size_candidates if bbs <= max_bbs]
        if allow_padding:
            batch_block_size = max(fits) if fits else min(batch_block_size_candidates)
        else:
            candidates = [
                bbs
                for bbs in fits
                if batch_size % bbs == 0 and (batch_size // bbs) % num_kernel == 0
            ]
            if not candidates:
                raise ValueError(
                    f"No batch block size in {batch_block_size_candidates} both fits "
                    f"VMEM (<= {max_bbs} batch elements) and divides batch_size="
                    f"{batch_size} into a block count that is a multiple of "
                    f"num_kernel={num_kernel}. Lower the per-element VMEM footprint "
                    f"or choose a batch size divisible by num_kernel."
                )
            batch_block_size = max(candidates)

    if not allow_padding:
        assert batch_size % batch_block_size == 0, (
            f"batch_size {batch_size} must be divisible by "
            f"batch_block_size={batch_block_size}"
        )
        n_batch_blocks = batch_size // batch_block_size
        assert n_batch_blocks % num_kernel == 0, (
            f"n_batch_blocks (batch_size // batch_block_size = {n_batch_blocks}) must be "
            f"divisible by num_kernel={num_kernel} so the megacore split "
            f"partitions blocks evenly across cores"
        )
    return batch_block_size


def pad_for_tpu(x: jax.Array) -> jax.Array:
    """Pad the last two axes to TPU tile boundaries.

    The last axis is padded up to a multiple of 128 (lane), the second-to-last
    up to a multiple of 8 (sublane). Leading axes are left untouched.
    """
    base = tuple((0, 0) for _ in range(len(x.shape[:-2])))
    return jnp.pad(x, (*base, (0, -(x.shape[-2] % -8)), (0, -(x.shape[-1] % -128))))
