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
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from e3j.pallas_ops.utils.named_scope import named_scope


class WalkScratch(NamedTuple):
    transition_mask: pl.MemoryRef  # (1,) i32: remaining within-chunk transitions
    open_segment_start: pl.MemoryRef  # (1,) i32: start of the still-open segment


class FlushBuffers(NamedTuple):
    dst_hbm: pl.MemoryRef  # (n_rows, planes, width): shared across cores
    stage_vmem: pl.MemoryRef  # (slots, planes, width): flush staging rows, a ring
    accumulator_vmem: (
        pl.MemoryRef
    )  # (planes, rows_per_chunk, width): open-group partials
    flush_pending: pl.MemoryRef  # (slots,) i32: async flush in flight?
    semaphore: pltpu.SemaphoreType  # (slots,) DMA semaphores for the async flush
    boundary_hbm: pl.MemoryRef  # (num_cores, 2, planes, width): straddling groups
    flush_count: pl.MemoryRef  # (1,) i32: flushes done by this core
    channel_slice: Any = slice(None)  # channel range this pass owns, full by default


def sum_rows(accumulator: pl.MemoryRef) -> jax.Array:
    """`accumulator.sum(axis=1)` as strided sublane loads instead of sublane rotates."""
    planes, rows, width = accumulator.shape
    if width != 128:  # Mosaic only lowers the strided load on 128-lane rows
        return jnp.sum(accumulator[:, :, :], axis=1)
    flat = accumulator.reshape(planes * rows, width)
    total = flat[pl.ds(0, planes, stride=rows), :]
    for row in range(1, rows):
        total = total + flat[pl.ds(row, planes, stride=rows), :]
    return total


def make_group_flush(bufs: FlushBuffers, core_id):
    """Double-buffered per-group reduce + async DMA out, shared by fwd/bwd."""

    def _dst(row):
        return bufs.dst_hbm.at[pl.ds(row, 1), :, bufs.channel_slice]

    def _boundary(slot):
        return bufs.boundary_hbm.at[core_id].at[pl.ds(slot, 1), :, bufs.channel_slice]

    slots = bufs.stage_vmem.shape[0]

    def _stage(slot):
        return bufs.stage_vmem.at[pl.ds(slot, 1)]

    def _wait_inflight(slot):
        # any destination of the same shape settles the same byte count
        pltpu.make_async_copy(_stage(slot), _dst(0), bufs.semaphore.at[slot]).wait()

    def flush_row(row, value: jax.Array, final: bool = False):
        slot = bufs.flush_count[0] % slots

        def _start(dst):
            pltpu.make_async_copy(_stage(slot), dst, bufs.semaphore.at[slot]).start()

        @pl.when(bufs.flush_pending[slot] == jnp.int32(1))
        def _wait_previous_flush():
            _wait_inflight(slot)

        bufs.stage_vmem[slot, :, :] = value

        if final:
            # the still-open group at the core's last edge: may continue into the
            # next core, so it is a partial and never written in place.
            _start(_boundary(1))
        else:
            is_first = bufs.flush_count[0] == jnp.int32(0)
            bufs.flush_count[0] = bufs.flush_count[0] + jnp.int32(1)

            @pl.when(is_first)
            def _first_group_is_partial():
                # opened before this core's first edge: may continue the previous
                # core's final group.
                _start(_boundary(0))

            @pl.when(jnp.logical_not(is_first))
            def _interior_group_owns_its_row():
                _start(_dst(row))

        bufs.flush_pending[slot] = jnp.int32(1)

    def flush_one(row, final: bool = False):
        value = sum_rows(bufs.accumulator_vmem)
        bufs.accumulator_vmem[:, :, :] = jnp.zeros_like(bufs.accumulator_vmem[:, :, :])
        flush_row(row, value, final)

    def drain():
        for slot in range(slots):

            @pl.when(bufs.flush_pending[slot] == jnp.int32(1))
            def _drain_pending_flush():
                _wait_inflight(slot)

    return flush_one, flush_row, drain


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
    fast_span: int = 1,  # chunks per single-group fast path (1 = off)
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

    def walk_chunk(chunk):
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

    assert n_chunks % fast_span == 0, "fast_span must divide the chunk count"
    if fast_span == 1:
        for chunk in range(n_chunks):
            walk_chunk(chunk)
        return
    # sorted keys: a span whose first and last keys are the open group lies wholly in it
    for first_chunk in range(0, n_chunks, fast_span):
        with named_scope("checking_span"):
            carried_key = open_key[0]
            single_group = jnp.logical_and(
                read_key(first_chunk * keys_per_chunk) == carried_key,
                read_key((first_chunk + fast_span) * keys_per_chunk - 1) == carried_key,
            )

        @pl.when(single_group)
        def _span_in_open_group():
            for idx in present_planes:
                total = accumulator[idx, :, :]
                for chunk in range(first_chunk, first_chunk + fast_span):
                    base = chunk * rows_per_chunk
                    total = total + source[idx, base : base + rows_per_chunk, :]
                accumulator[idx, :, :] = total

        @pl.when(jnp.logical_not(single_group))
        def _walk_span():
            for chunk in range(first_chunk, first_chunk + fast_span):
                walk_chunk(chunk)
