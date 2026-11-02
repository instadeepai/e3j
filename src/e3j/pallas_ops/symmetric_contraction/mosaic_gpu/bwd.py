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

"""Backward pass of the fused SymmetricContraction on Pallas Mosaic GPU.

One kernel recomputes each row's intermediate copies and walks the reduced
plan backwards (:func:`.unroll.emit_row_bwd`), giving:

* `dx`, written once per row;
* `dW_eff`, summed per species in a shared-memory table per warpgroup. Each
  CTA writes the sum of its tables, and XLA adds the CTAs' partials.

`dW` is then the transpose of the weight fold (:func:`unfold_weights`).
With `with_dw=False`, only `dx` is computed.

The same kernel on :class:`.unroll.Dual` rows gives the double backward
(:func:`chain_double_backward`).
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from e3j.pallas_ops.symmetric_contraction.common.plan import ReducedPlan, fold_weights
from e3j.pallas_ops.symmetric_contraction.common.unroll import (
    Dual,
    emit_row_bwd,
    emit_row_fwd,
)
from e3j.pallas_ops.symmetric_contraction.mosaic_gpu.common import (
    BLOCK_AXIS,
    SCALAR_BLOCK,
    THREAD_AXIS,
    block_indices,
    check_staged_axis,
    copy_table,
    resolve_channel_block,
    row_pipeline,
    species_of,
    table_copies,
    worker_walk,
)
from e3j.pallas_ops.symmetric_contraction.mosaic_gpu.params import (
    PallasMosaicGPUSymmetricContractionParams,
)
from e3j.pallas_ops.utils.mosaic_gpu import add_term, shared_memory_budget, sm_count

#: Default warpgroups per CTA (fewer in the backward if the tables do not
#: fit). In the backward each has its own `dW_eff` table: a shared one would
#: need shared-memory float atomics, which are slow CAS loops.
DEFAULT_WARPGROUPS = 2

#: Default warpgroups per CTA for a backward without `dW_eff`: one was
#: measured 1-5% faster than two on MACE layers.
DEFAULT_WARPGROUPS_DX = 1

#: Default maximum rows in flight per warpgroup.
MAX_PIPELINE_DEPTH = 4

#: Shared memory bytes kept free for barriers, species and alignment.
_SMEM_MARGIN = 4096


class ChainLaunch(NamedTuple):
    """How a kernel lays its work out on the device.

    Attributes:
        channel_block: Channels per CTA.
        n_channel_blocks: CTAs along the channel axis.
        row_groups: CTAs along the batch axis; in the backward, each writes
            one partial `dW_eff`.
        warpgroups: Warpgroups per CTA.
        depth: Rows in flight per warpgroup.
        resident_weights: Whether the folded weights are kept in shared
            memory; if not, they are read from GMEM (L2).
    """

    channel_block: int
    n_channel_blocks: int
    row_groups: int
    warpgroups: int
    depth: int
    resident_weights: bool

    @property
    def workers(self) -> int:
        """Total warpgroups walking the batch."""
        return self.row_groups * self.warpgroups


def resolve_launch(
    params: PallasMosaicGPUSymmetricContractionParams,
    plan: ReducedPlan,
    batch: int,
    channels: int,
    with_dw: bool = True,
    staged: int | None = None,
    forward: bool = False,
) -> ChainLaunch:
    """Pick a kernel's launch: one CTA per SM, tables in shared memory.

    The forward always keeps the weights resident. The backward prefers
    resident weights, then as many warpgroups as fit (up to the configured
    or default count). Each row stages `staged` coordinates (by default
    `x1`, plus `dy` in the backward).

    Raises:
        NotImplementedError: If no configuration leaves room for the
            requested rows in flight.
    """
    config = params.config
    channel_block = resolve_channel_block(channels, config.channel_block)
    n_channel_blocks = channels // channel_block
    table = params.num_species * plan.weight_rows * channel_block * 4
    if staged is None:
        staged = params.stage_dims[0] + (0 if forward else params.out_dim)
    slot = staged * channel_block * 4
    budget = shared_memory_budget() - _SMEM_MARGIN
    with_dw = with_dw and not forward
    most = config.warpgroups or (
        DEFAULT_WARPGROUPS if with_dw or forward else DEFAULT_WARPGROUPS_DX
    )
    if forward:
        candidates = [(True, most)]
    else:
        # Resident weights matter more than warpgroups: one warpgroup with
        # them measured 4% faster than two without.
        counts = range(most, 0, -1) if config.warpgroups is None else [most]
        candidates = [(r, w) for r in (True, False) for w in counts]
    for resident, warpgroups in candidates:
        tables = warpgroups * with_dw + resident
        room = (budget - tables * table) // (warpgroups * slot)
        depth = config.pipeline_depth or min(MAX_PIPELINE_DEPTH, room)
        if 1 <= depth <= room:
            break
    else:
        raise NotImplementedError(
            f"the chain kernel's tables of {table} bytes of shared memory "
            f"leave no room for {config.pipeline_depth or 1} rows of {slot} "
            f"bytes in flight per warpgroup"
        )
    row_groups = min(
        max(1, sm_count() // n_channel_blocks), math.ceil(batch / warpgroups)
    )
    return ChainLaunch(
        channel_block, n_channel_blocks, row_groups, warpgroups, depth, resident
    )


def unfold_weights(
    dw_eff: jax.Array,
    plan: ReducedPlan,
    params: PallasMosaicGPUSymmetricContractionParams,
) -> list[jax.Array]:
    """`dW` (one `(S, C, m_out, m_in)` per block) from `dW_eff` `(S, rows, C)`:
    the transpose of :func:`.plan.fold_weights`."""
    num_species, _, channels = dw_eff.shape
    shapes = [
        jax.ShapeDtypeStruct(
            (num_species, channels, block.m_out, block.m_in), dw_eff.dtype
        )
        for block in params.blocks
    ]
    transpose = jax.linear_transpose(lambda w: fold_weights(w, plan), shapes)
    (dweights,) = transpose(dw_eff)
    return dweights


def chain_backward(
    x1: jax.Array,
    index: jax.Array,
    w_eff: jax.Array,
    dy: jax.Array,
    plan: ReducedPlan,
    params: PallasMosaicGPUSymmetricContractionParams,
    with_dw: bool = True,
    dx_addend: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """`(dx1, dW_eff)` from the folded weights `w_eff` `(S, rows, C)`.

    `dW_eff` is `None` without `with_dw`. `dx_addend`, if given, is added
    to `dx1` inside the kernel, saving a separate pass.
    """

    def row(x, inputs, weight, accumulate, store, zero_row):
        (cts,) = inputs
        store(0, emit_row_bwd(plan, x, _cotangent(cts), weight, accumulate, zero_row))

    (dx1,), dw_eff = _chain_rows(
        x1,
        index,
        w_eff,
        plan,
        params,
        inputs=[(dy, _ct_coords(plan))],
        outputs=[x1.shape[1]],
        row=row,
        with_dw=with_dw,
        kernel_name="symmetric_contraction_mgpu_bwd",
        addends={} if dx_addend is None else {0: dx_addend},
    )
    return dx1, dw_eff


def chain_double_backward(
    x1: jax.Array,
    index: jax.Array,
    w_eff: jax.Array,
    dy: jax.Array,
    u: jax.Array,
    plan: ReducedPlan,
    params: PallasMosaicGPUSymmetricContractionParams,
    with_dw: bool = True,
) -> tuple[jax.Array, jax.Array | None, jax.Array]:
    """Differentiate :func:`chain_backward`'s `dx1` along its cotangent `u`.

    With `f(x, W) = sum_j W_j F_j(x)` the forward and `dx = J_x^T dy` the
    backward:

    ```
    d(x)   = H u,              H the Hessian of dy . f in x
    d(W_j) = <dy, DF_j(x)[u]>, summed per species
    d(dy)  = J_x u
    ```

    All three are derivatives along `u`, so they come from running the
    forward and backward rows on :class:`.unroll.Dual` rows `x + eps u`.

    The forward row runs first, so the shared powers die sooner (the other
    order spilled to local memory and was a third slower).

    Args:
        x1, index, w_eff, dy: The backward's operands.
        u: `(batch, x1_dim, channels)`, the cotangent of `dx1`.
        plan, params: Static structure of the call.
        with_dw: Whether to compute `d(W)`.

    Returns:
        `(d_x1, d_w_eff, d_dy)`, shaped like `x1`, `w_eff` and `dy`;
        `d_w_eff` is `None` without `with_dw`.
    """

    def row(x, inputs, weight, accumulate, store, zero_row):
        cts, us = inputs
        x = {k: Dual(value, us[k]) for k, value in x.items()}

        def dw(block, mo, j, value):
            accumulate(block, mo, j, Dual.parts(value)[1])

        y = emit_row_fwd(plan, x, weight, zero_row)
        store(1, {k: Dual.parts(value)[1] for k, value in y.items()})
        dw = None if accumulate is None else dw
        dx = emit_row_bwd(plan, x, _cotangent(cts), weight, dw, zero_row)
        store(0, {k: Dual.parts(value)[1] for k, value in dx.items()})

    (d_x1, d_dy), d_w_eff = _chain_rows(
        x1,
        index,
        w_eff,
        plan,
        params,
        inputs=[(dy, _ct_coords(plan)), (u, plan.x_coords)],
        outputs=[x1.shape[1], params.out_dim],
        row=row,
        with_dw=with_dw,
        kernel_name="symmetric_contraction_mgpu_bwd2",
    )
    return d_x1, d_w_eff, d_dy


def _cotangent(cts: dict):
    """The `ct(block, mo, m)` callback of :func:`.unroll.emit_row_bwd`."""
    return lambda block, mo, m: cts[block.offset + mo * block.dim + m]


def _ct_coords(plan: ReducedPlan) -> list[int]:
    """The output cotangent coordinates the plan reads."""
    return sorted(
        block.offset + k
        for block in plan.blocks
        for k in range(block.m_out * block.dim)
    )


def _chain_rows(
    x1: jax.Array,
    index: jax.Array,
    w_eff: jax.Array,
    plan: ReducedPlan,
    params: PallasMosaicGPUSymmetricContractionParams,
    *,
    inputs: list[tuple[jax.Array, list[int]]],
    outputs: list[int],
    row,
    with_dw: bool,
    kernel_name: str,
    addends: dict[int, jax.Array] | None = None,
) -> tuple[list[jax.Array], jax.Array | None]:
    """The shared backward kernel: runs `row` on every batch row.

    `inputs` are `(array, coords)` pairs: a `(batch, n, channels)` array and
    the coordinates `row` reads from it. `row(x, inputs, weight, accumulate,
    store, zero_row)` reads weights with `weight(block, mo, j)`, adds weight
    cotangents with `accumulate(block, mo, j, value)` (`None` without
    `with_dw`), and calls `store(i, {coordinate: row})` once per `outputs`
    entry (missing coordinates are zero). `addends[i]`, if given, is added
    to output `i`; it is staged with the inputs (14% faster than loading it
    in the row).

    The layout is the forward's (:func:`.fwd.chain_forward`), plus one
    `dW_eff` table per warpgroup, summed per CTA at the end.

    Returns:
        `(outputs, dW_eff)`, `dW_eff` `(S, rows, C)` or `None` without
        `with_dw`.
    """
    batch, x_dim, channels = x1.shape
    dtype = x1.dtype
    num_species = params.num_species
    weight_rows = plan.weight_rows
    addends = addends or {}
    # The addends are staged as extra inputs, hidden from `row`.
    inputs = inputs + [(addends[i], list(range(outputs[i]))) for i in addends]
    staged = x_dim + sum(array.shape[1] for array, _ in inputs)
    for array, _ in inputs:
        # Input rows are staged whole, like `x1`'s.
        check_staged_axis("a backward input's coordinates", array.shape[1])
    launch = resolve_launch(params, plan, batch, channels, with_dw, staged)
    channel_block, depth, n_wg = launch.channel_block, launch.depth, launch.warpgroups
    resident = launch.resident_weights
    table_rows = num_species * weight_rows
    spans = table_copies(table_rows, launch.n_channel_blocks)
    n_in, n_out = len(inputs), len(outputs)
    n_row_in = n_in - len(addends)
    row_layout = plgpu.Layout.WG_STRIDED((channel_block,), vec_size=1)

    def kernel(x1_gmem, w_gmem, index_gmem, *refs):
        in_gmems = refs[:n_in]
        out_gmems = refs[n_in : n_in + n_out]
        partial_gmem = refs[-1]
        row_group, block_index = block_indices(launch.n_channel_blocks)
        channel_slice = pl.ds(block_index * channel_block, channel_block)
        wg = jax.lax.axis_index(THREAD_AXIS)
        walk, num_steps = worker_walk(batch, row_group * n_wg + wg, launch.workers)
        zero_row = plgpu.layout_cast(jnp.zeros((channel_block,), dtype), row_layout)

        def load(ref):
            return plgpu.load(ref, layout=row_layout)

        def scoped(w_smem, dw_smem, x_smem, *rest):
            in_smems, (index_smem, barriers) = rest[:n_in], rest[n_in:]
            table_barrier, sync_barrier, slot_barrier = barriers

            if resident:

                @pl.when(wg == 0)
                def _load_table():
                    copy_table(w_gmem, w_smem, channel_slice, table_barrier, spans)

            if with_dw:
                dw_table = dw_smem.at[wg]

                @pl.loop(0, table_rows)
                def _zero(i):
                    dw_table[i] = zero_row

            def before_loop():
                if resident:
                    plgpu.barrier_wait(table_barrier)

            def read(slots):
                x_slot, in_slots = slots[0], slots[1:]
                x = {k: load(x_slot.at[k]) for k in plan.x_coords}
                values = [
                    {k: load(slot.at[k]) for k in coords}
                    for slot, (_, coords) in zip(in_slots, inputs)
                ]
                return x, values

            def body(step, r, values, raw):
                x, in_values = values
                # Out-of-range rows add nothing to `dW_eff`.
                species, in_range = species_of(raw, num_species)

                def table_row(block, mo, j):
                    return species * weight_rows + (
                        block.weight_row + mo * block.columns + j
                    )

                def weight(block, mo, j):
                    if resident:
                        return load(w_smem.at[table_row(block, mo, j)])
                    return load(w_gmem.at[table_row(block, mo, j), channel_slice])

                def accumulate(block, mo, j, value):
                    if value is not None:
                        acc = dw_table.at[table_row(block, mo, j)]
                        acc[...] = load(acc) + jnp.where(in_range, value, zero_row)

                extra = dict(zip(addends, in_values[n_row_in:]))

                def store(i, out):
                    for k in range(outputs[i]):
                        value = out.get(k)
                        if i in extra:
                            value = add_term(value, extra[i][k])
                        out_gmems[i][r, k, channel_slice] = (
                            zero_row if value is None else value
                        )

                row(
                    x,
                    in_values[:n_row_in],
                    weight,
                    accumulate if with_dw else None,
                    store,
                    zero_row,
                )

            row_pipeline(
                body,
                row=walk,
                num_steps=num_steps,
                depth=depth,
                copies=[
                    (lambda r: x1_gmem.at[r, :, channel_slice], x_smem.at[wg]),
                    *[
                        (
                            lambda r, gmem=gmem: gmem.at[r, :, channel_slice],
                            smem.at[wg],
                        )
                        for gmem, smem in zip(in_gmems, in_smems)
                    ],
                ],
                barrier=lambda slot: slot_barrier.at[wg * depth + slot],
                row_scalars=(index_gmem, index_smem.at[wg]),
                read=read,
                before_loop=before_loop,
            )

            if not with_dw:
                return
            # Once every warpgroup is done, write the sum of the tables,
            # splitting the rows between warpgroups.
            plgpu.barrier_arrive(sync_barrier)
            plgpu.barrier_wait(sync_barrier)

            @pl.loop(wg, table_rows, step=n_wg)
            def _flush(i):
                total = None
                for t in range(n_wg):
                    total = add_term(total, load(dw_smem.at[t, i]))
                partial_gmem[row_group, i, channel_slice] = total

        pl.run_scoped(
            scoped,
            plgpu.SMEM((table_rows if resident else 1, channel_block), dtype),
            plgpu.SMEM(
                (n_wg if with_dw else 1, table_rows if with_dw else 1, channel_block),
                dtype,
            ),
            plgpu.SMEM((n_wg, depth, x_dim, channel_block), dtype),
            *[
                plgpu.SMEM((n_wg, depth, array.shape[1], channel_block), dtype)
                for array, _ in inputs
            ],
            plgpu.SMEM((n_wg, depth, SCALAR_BLOCK), jnp.int32),
            (
                plgpu.Barrier(num_arrivals=len(spans)),
                plgpu.Barrier(num_arrivals=n_wg),
                # A row's `x1`, inputs and species block.
                plgpu.Barrier(num_arrivals=2 + n_in, num_barriers=n_wg * depth),
            ),
            collective_axes=THREAD_AXIS,
        )

    # The weights are copied flat, `(species * weight_rows, C)`.
    w_table = w_eff.reshape(table_rows, channels)
    # The partial `dW_eff`s. Without `with_dw`, a dummy row is still output:
    # XLA schedules a single-output custom call after unrelated work, which
    # cost ~8 µs.
    partial_rows = (launch.row_groups, table_rows) if with_dw else (1,)
    *outs, partial = plgpu.kernel(
        kernel,
        out_type=(
            *[jax.ShapeDtypeStruct((batch, n, channels), dtype) for n in outputs],
            jax.ShapeDtypeStruct((*partial_rows, channels), dtype),
        ),
        # All stores go straight to GMEM; the explicit barriers provide
        # the ordering the copies and tables need.
        compiler_params=plgpu.CompilerParams(
            unsafe_no_auto_barriers=True,
            lowering_semantics=plgpu.LoweringSemantics.Lane,
        ),
        grid=(launch.row_groups * launch.n_channel_blocks,),
        grid_names=(BLOCK_AXIS,),
        num_threads=n_wg,
        thread_name=THREAD_AXIS,
        kernel_name=kernel_name,
    )(
        x1,
        w_table,
        index.astype(jnp.int32),
        *[array for array, _ in inputs],
    )
    if not with_dw:
        return outs, None
    return outs, partial.sum(axis=0).reshape(num_species, weight_rows, channels)
