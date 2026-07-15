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

"""Sorted reduce-and-flush shared by the fused MP-conv fwd/bwd kernels.

Both kernels reduce per-edge rows by a sorted key (fwd: receivers; bwd: senders).
This walks the block's edges in chunks and, at each group boundary,
closes the open group's segment, accumulates and flushes it.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from e3j.pallas_ops.utils.named_scope import named_scope


class WalkScratch(NamedTuple):
    transition_mask: pl.MemoryRef  # (1,) i32: remaining within-chunk transitions
    open_segment_start: pl.MemoryRef  # (1,) i32: start of the still-open segment


class FlushBuffers(NamedTuple):
    dst_hbm: pl.MemoryRef  # (num_cores, n_rows, planes, width): per-core output
    stage_vmem: pl.MemoryRef  # (1, planes, width): flush staging row
    accumulator_vmem: (
        pl.MemoryRef
    )  # (planes, rows_per_chunk, width): open-group partials
    flush_pending: pl.MemoryRef  # (1,) i32: async flush in flight?
    semaphore: pltpu.SemaphoreType  # DMA semaphore for the async flush


def make_group_flush(bufs: FlushBuffers, core_id):
    """Double-buffered per-group reduce + async DMA out, shared by fwd/bwd.

    Returns `(flush_one, drain)`. `flush_one(row)` waits any in-flight copy,
    reduces the accumulator's chunk sublanes to one row, async-DMAs it to
    `dst_hbm[core_id, row]`, and re-zeros the accumulator. `drain()` waits the
    last copy (call once after the pipeline).
    """

    def _wait_inflight():
        pltpu.make_async_copy(
            bufs.stage_vmem,
            bufs.dst_hbm.at[core_id].at[pl.ds(0, 1), :, :],
            bufs.semaphore,
        ).wait()

    def flush_one(row):
        @pl.when(bufs.flush_pending[0] == jnp.int32(1))
        def _wait_previous_flush():
            _wait_inflight()

        bufs.stage_vmem[0, :, :] = jnp.sum(bufs.accumulator_vmem[:, :, :], axis=1)
        bufs.accumulator_vmem[:, :, :] = jnp.zeros_like(bufs.accumulator_vmem[:, :, :])
        pltpu.make_async_copy(
            bufs.stage_vmem,
            bufs.dst_hbm.at[core_id].at[pl.ds(row, 1), :, :],
            bufs.semaphore,
        ).start()
        bufs.flush_pending[0] = jnp.int32(1)

    def drain():
        @pl.when(bufs.flush_pending[0] == jnp.int32(1))
        def _drain_pending_flush():
            _wait_inflight()

    return flush_one, drain


def sorted_reduce_and_flush(
    *,
    read_key: Callable[[int], jax.Array],  # flat key index -> sorted group key
    cell_position: jax.Array,  # (rows_per_chunk, width): within-chunk key position of each cell
    present_planes: list[int],  # leading-axis planes of source/accumulator to scatter
    source: jax.Array,  # (planes, n_chunks*rows_per_chunk, width): per-edge plane
    accumulator: pl.MemoryRef,  # (planes_padded, rows_per_chunk, width): open-group partials
    flush_one: Callable[[jax.Array], None],  # close + emit one group given its key
    open_key: pl.MemoryRef,  # (1,) i32 SHARED carry: open group id (caller inits -1, reads after)
    scratch: WalkScratch,  # the two private loop-carry cells
    edges_per_row: int = 1,  # keys packed per row: 1 = dense, LANES//channels = packed
) -> None:

    # rows_per_chunk: source/accumulator sublane rows per chunk (the scatter target's height).
    # n_chunks: main-loop count -- source axis 1 spans n_chunks*rows_per_chunk edges.
    # keys_per_chunk: sorted keys per chunk = transition-bitmask width.
    rows_per_chunk = accumulator.shape[1]
    assert (
        cell_position.shape[0] == rows_per_chunk
    ), "cell_position must have one row per accumulator sublane row"
    assert (
        source.shape[1] % rows_per_chunk == 0
    ), "source rows must split evenly into chunks"
    n_chunks = source.shape[1] // rows_per_chunk
    keys_per_chunk = rows_per_chunk * edges_per_row
    assert keys_per_chunk <= 32, "transition bitmask is packed into an int32"

    def accumulate(chunk, in_segment):
        base = chunk * rows_per_chunk
        for idx in present_planes:
            accumulator[idx, :, :] = accumulator[idx, :, :] + jnp.where(
                in_segment, source[idx, base : base + rows_per_chunk, :], 0.0
            )

    for chunk in range(n_chunks):
        with named_scope("checking_transitions"):
            base = chunk * keys_per_chunk
            chunk_keys = [read_key(base + j) for j in range(keys_per_chunk)]
            carried_key = open_key[0]
            # keys are sorted, so the chunk crosses a group boundary iff its
            # first/last key differ or the carried (cross-chunk) group != key 0.
            boundary = jnp.logical_or(
                chunk_keys[0] != carried_key,
                chunk_keys[0] != chunk_keys[keys_per_chunk - 1],
            )
            scratch.open_segment_start[0] = jnp.int32(0)

        with named_scope("boundary_branch"):

            @pl.when(boundary)
            def _slow_walk_chunk():
                # transition bitmask: bit j set where key j starts a new group
                # (bit 0 compares key 0 against the carried group); n_new_groups
                # = how many groups start in the chunk = flushes to do.
                walk_mask = jnp.int32(0)
                n_new_groups = jnp.int32(0)
                previous_key = carried_key
                for j in range(keys_per_chunk):
                    new_group = (chunk_keys[j] != previous_key).astype(jnp.int32)
                    walk_mask = walk_mask + (new_group << j)
                    n_new_groups = n_new_groups + new_group
                    previous_key = chunk_keys[j]
                scratch.transition_mask[0] = walk_mask

                with named_scope("loop_accumulating_and_flushing_complete_group"):

                    @pl.loop(0, n_new_groups)
                    def _walk_transition(_k):
                        with named_scope("selecting_segment"):
                            mask_cur = scratch.transition_mask[0]
                            next_transition_bit = mask_cur & (-mask_cur)

                            # decode the bit to its position `pos` and the key opening there.
                            pos = jnp.int32(0)
                            opening_key = chunk_keys[0]
                            for j in range(keys_per_chunk):
                                transition_at_j = next_transition_bit == jnp.int32(
                                    1 << j
                                )
                                pos = jnp.where(transition_at_j, jnp.int32(j), pos)
                                opening_key = jnp.where(
                                    transition_at_j, chunk_keys[j], opening_key
                                )
                            previous_position = scratch.open_segment_start[0]
                            closing_key = open_key[0]

                            # close the segment [previous_position, pos) into the open group.
                            in_segment = jnp.logical_and(
                                cell_position >= previous_position, cell_position < pos
                            )

                        with named_scope("accumulating"):
                            accumulate(chunk, in_segment)

                        with named_scope("flushing_complete_group"):

                            @pl.when(closing_key >= jnp.int32(0))
                            def _flush():
                                flush_one(closing_key)

                        # open the next group, move the segment start, drop the bit.
                        with named_scope("opening_new_group"):
                            open_key[0] = opening_key.astype(jnp.int32)
                            scratch.open_segment_start[0] = pos
                            scratch.transition_mask[0] = mask_cur & (
                                mask_cur - jnp.int32(1)
                            )

        # unconditional: the still-open group keeps the chunk's tail rows.
        with named_scope("no_boundary_accummulation"):
            tail = cell_position >= scratch.open_segment_start[0]
            accumulate(chunk, tail)
