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

"""Forward pass of the fused symmetric contraction on Pallas Mosaic TPU.

One ``pallas_call`` evaluates the whole contraction through the reduced plan
(:mod:`e3j.pallas_ops.symmetric_contraction.common.plan`): each batch block is
read into VMEM once and only the output is written back. The folded weights
(one ``(plan.weight_rows, mul)`` table per species) stay in HBM: each batch
row DMAs its species' table into a double-buffered VMEM block alongside ``x``.

The arithmetic runs on one ``(batch, mul)`` tile per lm coordinate.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from e3j.pallas_ops.symmetric_contraction.common.plan import ReducedPlan, build_plan
from e3j.pallas_ops.symmetric_contraction.common.unroll import emit_row_fwd
from e3j.pallas_ops.symmetric_contraction.mosaic_tpu.params import (
    BATCH_BLOCK_SIZE_CANDIDATES,
    PallasMosaicTPUSymmetricContractionParams,
)

# The arithmetic runs on (SUBLANES, LANES) = (batch rows, channels) vregs.
LANES = 128
SUBLANES = 8
# Mosaic's default scoped VMEM limit, and how much of it to leave to Mosaic.
_VMEM_SCOPED = 16 * 1024 * 1024
_VMEM_HEADROOM = 2 * 1024 * 1024
# VMEM the buffers may take when the block size is picked automatically.
_VMEM_BUDGET = 6 * 1024 * 1024


def block_size(
    rows_per_batch_row: int, candidates: list[int], reserved: int = 0
) -> int:
    """Largest of ``candidates`` that fits the VMEM budget.

    Each batch row takes ``rows_per_batch_row`` f32 ``(LANES,)`` rows, on
    top of ``reserved`` fixed bytes.
    """
    budget = min(_VMEM_BUDGET, _VMEM_SCOPED - _VMEM_HEADROOM - reserved)
    row_bytes = rows_per_batch_row * LANES * 4
    fits = [b for b in candidates if b * row_bytes <= budget]
    return max(fits, default=min(candidates))


def _pad_lanes(a: jax.Array) -> jax.Array:
    pad = -a.shape[-1] % LANES
    return jnp.pad(a, ((0, 0),) * (a.ndim - 1) + ((0, pad),)) if pad else a


def launch(
    body,
    inputs: tuple[jax.Array, ...],
    w: jax.Array,
    species: jax.Array,
    out_rows: list[int],
    bbs: int,
    name: str,
    with_dw: bool = False,
) -> list[jax.Array]:
    """Run ``body`` over blocks of ``bbs`` batch rows, split across cores.

    ``body(xs, wb)`` gets ``SUBLANES`` batch rows of every input and of
    ``w[species]``, each as a list of ``(SUBLANES, LANES)`` tiles, one per
    coordinate. It returns one list of tiles per ``out_rows`` entry, plus,
    with ``with_dw``, the weight cotangent, which is summed per species into
    an extra ``(S, w_rows, mul)`` output.

    All relayouts are done by DMAs: inputs are copied one coordinate at a
    time into lm-major ``(rows, bbs, LANES)`` buffers, double-buffered, and
    outputs are copied back the same way. The batch is not padded: the last
    block is shifted back to end on the last row, and its rows already
    counted are skipped in the ``dw`` sums.

    Channels wider than ``LANES`` run as one launch per lane tile: on v4,
    XLA may place operands in CMEM, where lane-tile DMAs of wider rows
    corrupt data (libtpu 0.0.46).
    """
    mul = w.shape[-1]
    if mul > LANES:
        passes = [
            launch(
                body,
                tuple(a[..., c : c + LANES] for a in inputs),
                w[..., c : c + LANES],
                species,
                out_rows,
                bbs,
                name,
                with_dw,
            )
            for c in range(0, mul, LANES)
        ]
        return [jnp.concatenate(parts, axis=-1) for parts in zip(*passes)]
    num_cores = pltpu.get_tpu_info().num_cores
    batch = inputs[0].shape[0]
    if batch < SUBLANES:
        pad = ((0, SUBLANES - batch), (0, 0), (0, 0))
        inputs = tuple(jnp.pad(a, pad) for a in inputs)
        species = jnp.pad(species, (0, SUBLANES - batch))
    inputs = tuple(_pad_lanes(a) for a in inputs)
    # Pad weight rows to a multiple of SUBLANES: DMAs of tables with
    # 8k + r rows corrupt their neighbours.
    w_rows_unpadded = w.shape[1]
    w = jnp.pad(_pad_lanes(w), ((0, 0), (0, -w.shape[1] % SUBLANES), (0, 0)))
    num_species, w_rows, _ = w.shape
    rows_total = inputs[0].shape[0]
    in_rows = [a.shape[1] for a in inputs]
    block = min(bbs, rows_total // SUBLANES * SUBLANES)
    n_blocks = pl.cdiv(rows_total, block)
    steps = pl.cdiv(n_blocks, num_cores)

    def block_start(b):
        return jnp.minimum(b * block, rows_total - block)

    # Each block's species (the last block shifted back).
    starts = np.minimum(np.arange(n_blocks) * block, rows_total - block)
    species = species.astype(jnp.int32)[starts[:, None] + np.arange(block)]
    species = species.reshape(n_blocks, 1, block)

    def species_spec(ahead):
        return pl.BlockSpec(
            (1, 1, block),
            lambda core, step: (
                jnp.minimum(core * steps + step + ahead, n_blocks - 1),
                0,
                0,
            ),
            memory_space=pltpu.SMEM,
        )

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [
        species_spec(0),
        species_spec(1),
        hbm,
        *(hbm for _ in inputs),
    ]
    out_specs = [hbm for _ in out_rows]
    out_shape = [
        jax.ShapeDtypeStruct((rows_total, rows, LANES), w.dtype) for rows in out_rows
    ]
    if with_dw:
        # One per-species dw sum per core, kept in VMEM across its steps.
        out_specs.append(
            pl.BlockSpec((1, *w.shape), lambda core, step: (core, 0, 0, 0))
        )
        out_shape.append(jax.ShapeDtypeStruct((num_cores, *w.shape), w.dtype))

    # (slot, coordinate, batch row, lane)
    def buffer(rows):
        return pltpu.VMEM((2, rows, block, LANES), w.dtype)

    scratch = [
        *(buffer(rows) for rows in in_rows),
        buffer(w_rows),
        *(buffer(rows) for rows in out_rows),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    if with_dw:
        scratch += [
            pltpu.VMEM((w_rows, block, LANES), w.dtype),
            pltpu.VMEM((block, w_rows, LANES), w.dtype),
            pltpu.SemaphoreType.DMA(()),
        ]
    n_in, n_out = len(inputs), len(out_rows)
    # VMEM of the buffers above and the double-buffered dw sums.
    row_bytes = LANES * w.dtype.itemsize
    vmem = 2 * (sum(in_rows) + w_rows + sum(out_rows)) * block * row_bytes
    if with_dw:
        vmem += 2 * w_rows * block * row_bytes + 2 * num_species * w_rows * row_bytes
    # With many species, the dw sums alone can exceed the default limit.
    vmem_limit = None
    if vmem + _VMEM_HEADROOM > _VMEM_SCOPED:
        vmem_limit = vmem + _VMEM_HEADROOM
        capacity = pltpu.get_tpu_info().vmem_capacity_bytes
        if vmem_limit > capacity:
            raise NotImplementedError(
                f"{name} needs {vmem_limit} bytes of VMEM, the TPU has {capacity}"
            )

    def kernel(species_ref, next_species_ref, w_ref, *refs):
        in_hbm, refs = refs[:n_in], refs[n_in:]
        out_hbm, refs = refs[:n_out], refs[n_out:]
        if with_dw:
            dw_ref, *refs = refs
        in_bufs, refs = refs[:n_in], refs[n_in:]
        w_buf, *refs = refs
        out_bufs, refs = refs[:n_out], refs[n_out:]
        in_sem, out_sem, *refs = refs
        if with_dw:
            dw_buf, dw_stage, dw_sem = refs

        core, step = pl.program_id(0), pl.program_id(1)
        blk = core * steps + step
        # This core's last step with a block (-1 if none).
        last = jnp.minimum(steps, n_blocks - core * steps) - 1
        slot = step % 2

        def in_copies(b, slot, sp_ref):
            start = block_start(b)
            copies = [
                pltpu.make_async_copy(
                    src.at[pl.ds(start, block), pl.ds(k, 1)],
                    buf.at[slot, k].reshape(block, 1, LANES),
                    in_sem.at[slot],
                )
                for src, buf in zip(in_hbm, in_bufs)
                for k in range(buf.shape[1])
            ]
            copies += [
                pltpu.make_async_copy(
                    w_ref.at[sp_ref[0, 0, n]], w_buf.at[slot, :, n], in_sem.at[slot]
                )
                for n in range(block)
            ]
            return copies

        def out_copies(b, slot):
            start = block_start(b)
            return [
                pltpu.make_async_copy(
                    buf.at[slot, m].reshape(block, 1, LANES),
                    dst.at[pl.ds(start, block), pl.ds(m, 1)],
                    out_sem.at[slot],
                )
                for dst, buf in zip(out_hbm, out_bufs)
                for m in range(buf.shape[1])
            ]

        if with_dw:

            @pl.when(step == 0)
            def _():
                dw_ref[...] = jnp.zeros_like(dw_ref)

        @pl.when((step == 0) & (step <= last))
        def _():
            for copy in in_copies(blk, slot, species_ref):
                copy.start()

        @pl.when(step + 1 <= last)
        def _():
            for copy in in_copies(blk + 1, 1 - slot, next_species_ref):
                copy.start()

        @pl.when(step <= last)
        def _():
            for copy in in_copies(blk, slot, species_ref):
                copy.wait()

            # Wait for this slot's outputs from two steps ago.
            @pl.when(step >= 2)
            def _():
                for copy in out_copies(blk - 2, slot):
                    copy.wait()

            # SUBLANES batch rows at a time.
            @pl.loop(0, block // SUBLANES)
            def _(j):
                rows = pl.ds(pl.multiple_of(j * SUBLANES, SUBLANES), SUBLANES)
                outs = body(
                    [
                        [buf[slot, k, rows] for k in range(buf.shape[1])]
                        for buf in in_bufs
                    ],
                    [w_buf[slot, r, rows] for r in range(w_rows)],
                )
                if with_dw:
                    *outs, dw = outs
                    for r, tile in enumerate(dw):
                        dw_buf[r, rows] = tile
                for buf, tiles in zip(out_bufs, outs, strict=True):
                    for m, tile in enumerate(tiles):
                        buf[slot, m, rows] = tile

            for copy in out_copies(blk, slot):
                copy.start()

            if with_dw:
                copies = [
                    pltpu.make_async_copy(dw_buf.at[:, n], dw_stage.at[n], dw_sem)
                    for n in range(block)
                ]
                for copy in copies:
                    copy.start()
                for copy in copies:
                    copy.wait()
                # Skip rows of the shifted last block already summed.
                first_new = blk * block - block_start(blk)
                for n in range(block):

                    @pl.when(first_new <= n)
                    def _(n=n):
                        dw_ref[0, species_ref[0, 0, n]] += dw_stage[n]

            @pl.when(step == last)
            def _():
                for copy in out_copies(blk, slot):
                    copy.wait()

                @pl.when(step >= 1)
                def _():
                    for copy in out_copies(blk - 1, 1 - slot):
                        copy.wait()

    out = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(num_cores, steps),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                pltpu.GridDimensionSemantics.PARALLEL,
                pltpu.GridDimensionSemantics.ARBITRARY,
            ),
            vmem_limit_bytes=vmem_limit,
        ),
        name=name,
    )(species, species, w, *inputs)
    if with_dw:
        *out, dw = out
    # Trim the padding.
    out = [o[:batch, :, :mul] for o in out]
    if with_dw:
        out.append(jnp.sum(dw, axis=0)[:, :w_rows_unpadded, :mul])
    return out


def forward_rows(plan: ReducedPlan, x, w, n_out: int) -> list[jax.Array]:
    """The kernel body: ``n_out`` output tiles from the tiles ``x[k]`` (input
    coordinate ``k``) and ``w[r]`` (folded weight row ``r``)."""
    zero = jnp.zeros_like(x[0])
    out = emit_row_fwd(
        plan,
        {k: x[k] for k in plan.x_coords},
        lambda block, mo, j: w[block.weight_row + mo * block.columns + j],
        zero,
    )
    return [out.get(k, zero) for k in range(n_out)]


def check_inputs(x, w, species, params, plan) -> None:
    """Assert the shapes of ``x``, ``w`` and ``species``."""
    assert x.ndim == 3, f"x must be (batch, x_dim, mul), got {x.shape}"
    assert w.ndim == 3, f"w must be (num_species, w_rows, mul), got {w.shape}"
    batch_size, x_dim, mul = x.shape
    assert (
        x_dim == params.stage_dims[0]
    ), f"x lm axis {x_dim} != source dim {params.stage_dims[0]}"
    assert (
        w.shape[1] == plan.weight_rows and w.shape[2] == mul
    ), f"w {w.shape} incompatible with x {x.shape}, {plan.weight_rows} weight rows"
    assert species.shape == (
        batch_size,
    ), f"species {species.shape} must be ({batch_size},)"


def _symmetric_contraction_kernel_mosaic_tpu_fwd(
    x: jax.Array,
    w: jax.Array,
    species: jax.Array,
    params: PallasMosaicTPUSymmetricContractionParams,
) -> jax.Array:
    """Fused forward symmetric contraction."""
    plan = build_plan(params)
    check_inputs(x, w, species, params, plan)
    # Per batch row: double-buffered x, weight rows and output.
    bbs = params.fwd_batch_block_size or block_size(
        2 * (x.shape[1] + plan.weight_rows + params.out_dim),
        BATCH_BLOCK_SIZE_CANDIDATES,
    )

    def body(xs, wb):
        (xb,) = xs
        return [forward_rows(plan, xb, wb, params.out_dim)]

    (out,) = launch(
        body, (x,), w, species, [params.out_dim], bbs, "symmetric_contraction_mtpu_fwd"
    )
    return out


def symmetric_contraction_reference(
    x: jax.Array,
    w: jax.Array,
    species: jax.Array,
    params: PallasMosaicTPUSymmetricContractionParams,
) -> jax.Array:
    """Pure-JAX version of the kernel, on the same inputs.

    Runs the kernel body (:func:`forward_rows`) on whole arrays, so it tests
    the kernel's tiling and pipelining, and JAX can differentiate it twice.
    """
    rows = forward_rows(
        build_plan(params),
        jnp.moveaxis(x, 1, 0),
        jnp.moveaxis(w[species], 1, 0),
        params.out_dim,
    )
    return jnp.stack(rows, axis=1)
