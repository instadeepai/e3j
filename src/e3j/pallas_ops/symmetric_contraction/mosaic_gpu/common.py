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

"""Validation and row pipeline of the Mosaic GPU SymmetricContraction kernels.

Like the tensor product kernels, each CTA covers one `channel_block` of
channels, one channel per lane, so their blocking helpers are reused.
"""

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu

from e3j.pallas_ops.symmetric_contraction.common import params as common_params
from e3j.pallas_ops.symmetric_contraction.mosaic_gpu.params import (
    PallasMosaicGPUSymmetricContractionParams,
)

# Re-exported for the kernels.
from e3j.pallas_ops.tensor_product.mosaic_gpu.common import (  # noqa: F401
    BLOCK_AXIS,
    block_indices,
    check_staged_axis,
    check_staged_smem,
    resolve_channel_block,
)
from e3j.pallas_ops.utils.mosaic_gpu import MAX_COPY_ELEMS_PER_AXIS, WARPGROUP_LANES


def validate_operands(
    x1: jax.Array,
    index: jax.Array,
    weights: list[jax.Array],
    params: PallasMosaicGPUSymmetricContractionParams,
) -> None:
    """Check the operands against each other and the kernels' limits.

    Raises:
        ValueError: If the operands do not match.
        NotImplementedError: If they are not float32, the channels are not
            a multiple of `WARPGROUP_LANES`, or `x1` is too wide to stage.
    """
    common_params.validate_operands(x1, index, weights, params)
    _, x1_dim, channels = x1.shape
    # The weights share x1's dtype.
    if x1.dtype != jnp.float32:
        raise NotImplementedError(
            f"the Mosaic GPU symmetric_contraction kernel is float32-only, "
            f"x1 is {x1.dtype}"
        )
    if channels < WARPGROUP_LANES or channels % WARPGROUP_LANES:
        raise NotImplementedError(
            f"channels={channels} must be a positive multiple of "
            f"{WARPGROUP_LANES}: every SMEM row the kernel materializes is a "
            f"(channels,) vector, which Mosaic GPU only vectorizes at that "
            f"granularity"
        )
    check_staged_axis("x1_space", x1_dim)


#: The thread (warpgroup) axis of the warp-specialized kernels.
THREAD_AXIS = "wg"


def warpgroup_barrier() -> None:
    """Synchronize the warpgroup, without `commit_smem`'s proxy fence.

    Enough before refilling a slot the warpgroup was reading: the fence is
    only needed before copying out what the warpgroup wrote.
    """

    @plgpu.inline_mgpu()
    def barrier(ctx):
        del ctx
        mgpu_utils.warpgroup_barrier()

    barrier()


def table_copies(rows: int, n_channel_blocks: int) -> list[tuple[int, int]]:
    """`(start, size)` row spans for copying a `(rows, channels)` table.

    One contiguous copy if a CTA covers every channel, else copies of at
    most `MAX_COPY_ELEMS_PER_AXIS` rows.
    """
    if n_channel_blocks == 1:
        return [(0, rows)]
    step = MAX_COPY_ELEMS_PER_AXIS
    return [(start, min(step, rows - start)) for start in range(0, rows, step)]


def copy_table(gmem, smem, channel_slice, barrier, spans) -> None:
    """Copy `gmem[:, channel_slice]` into `smem`, one copy per span of
    :func:`table_copies`, all arriving on `barrier`."""
    for start, size in spans:
        plgpu.copy_gmem_to_smem(
            gmem.at[pl.ds(start, size), channel_slice],
            smem.at[pl.ds(start, size)],
            barrier,
            oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
        )


#: `(gmem, smem)`: `gmem(row)` is the GMEM ref to copy for a batch row, and
#: `smem` its `(depth, ...)` SMEM slots.
RowCopy = tuple[Callable[[jax.Array], Any], Any]


def species_of(raw: jax.Array, num_species: int) -> tuple[jax.Array, jax.Array]:
    """`(species, in_range)` of a raw index: negative indices count from the
    end, out-of-range ones are clamped."""
    raw = jnp.where(raw < 0, raw + num_species, raw)
    species = jnp.clip(raw, 0, num_species - 1)
    return species, raw == species


def worker_walk(batch: int, worker: jax.Array, workers: int):
    """`(row, num_steps)` of one worker, for :func:`row_pipeline`.

    Worker `k` takes rows `k, k + workers, ...`, so the rows in flight are
    contiguous: 2-5% faster than contiguous runs per worker on MACE layers.
    """
    num_steps = jnp.maximum(batch - worker + workers - 1, 0) // workers
    return (lambda step: worker + step * workers), num_steps


#: Per-row scalars are copied in aligned blocks of this many (128 bytes):
#: the smallest copy Mosaic GPU makes, and 16-byte aligned as a copy from a
#: 1D GMEM array requires (otherwise it traps).
SCALAR_BLOCK = 32


def row_pipeline(
    body: Callable[..., Any],
    *,
    row: Callable[[jax.Array], jax.Array],
    num_steps: jax.Array,
    depth: int,
    copies: Sequence[RowCopy],
    barrier: Callable[[jax.Array], Any],
    row_scalars: tuple[Any, Any],
    read: Callable[[list[Any]], Any],
    before_loop: Callable[[], None],
) -> None:
    """Process the batch rows `row(step)` of one warpgroup, `depth` in flight.

    Each step's `copies` land in slot `step % depth`, issued `depth` steps
    ahead by the first warp. A step reads its slots into registers (`read`)
    and frees them before its math, so even `depth=1` overlaps one row's
    math with the next row's copy.

    A per-row scalar (e.g. the species) is copied with the row rather than
    loaded from GMEM, which would stall the step.

    Args:
        body: `body(step, row, values, scalar)`, the step's math on what
            `read` returned and the row's scalar.
        row: `row(step)`, the batch row of a step.
        num_steps: Scalar int32, possibly zero.
        depth: Steps in flight (number of slots).
        copies: The copies of each step (:data:`RowCopy`), all in bounds.
        barrier: `barrier(slot)`, the slot's barrier, with `len(copies) + 1`
            arrivals.
        row_scalars: `(gmem, smem)`: the 1D `(batch,)` GMEM array of scalars
            and its `(depth, SCALAR_BLOCK)` SMEM slots. Out-of-bounds reads
            are zero-filled.
        read: `read(slots) -> values`, loads what `body` needs from a step's
            slots (one per copy).
        before_loop: Called once, after the first copies are issued and
            before the first step.
    """

    def issue(step, slot) -> None:
        step_row = row(step)

        @plgpu.warp_map
        def _issue(warp):
            @pl.when(warp == 0)
            def _first_warp():
                for gmem, smem in copies:
                    # In bounds, so contiguous copies need no tensor map.
                    plgpu.copy_gmem_to_smem(
                        gmem(step_row),
                        smem.at[slot],
                        barrier(slot),
                        oob_mode=plgpu.OOBFillMode.PROMISE_IN_BOUNDS,
                    )
                gmem, smem = row_scalars
                block = step_row - jax.lax.rem(step_row, jnp.int32(SCALAR_BLOCK))
                plgpu.copy_gmem_to_smem(
                    gmem.at[pl.ds(block, SCALAR_BLOCK)],
                    smem.at[slot],
                    barrier(slot),
                    oob_mode=plgpu.OOBFillMode.ZEROS,
                )

    for step in range(depth):

        @pl.when(step < num_steps)
        def _prologue(step=step):
            issue(jnp.int32(step), step)

    before_loop()

    def loop(step, carry):
        slot = jax.lax.rem(step, jnp.int32(depth))
        step_row = row(step)
        plgpu.barrier_wait(barrier(slot))
        at = jax.lax.rem(step_row, jnp.int32(SCALAR_BLOCK))
        scalar = plgpu.load(row_scalars[1].at[slot, at])
        values = read([smem.at[slot] for _, smem in copies])
        warpgroup_barrier()

        @pl.when(step + depth < num_steps)
        def _refill():
            issue(step + depth, slot)

        body(step, step_row, values, scalar)
        return carry

    jax.lax.fori_loop(0, num_steps, loop, None)
