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

"""Forward pass of the fused SymmetricContraction on Pallas Mosaic GPU."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from e3j.pallas_ops.symmetric_contraction.common.plan import ReducedPlan
from e3j.pallas_ops.symmetric_contraction.common.unroll import emit_row_fwd
from e3j.pallas_ops.symmetric_contraction.mosaic_gpu.bwd import resolve_launch
from e3j.pallas_ops.symmetric_contraction.mosaic_gpu.common import (
    BLOCK_AXIS,
    SCALAR_BLOCK,
    THREAD_AXIS,
    block_indices,
    copy_table,
    row_pipeline,
    species_of,
    table_copies,
    worker_walk,
)
from e3j.pallas_ops.symmetric_contraction.mosaic_gpu.params import (
    PallasMosaicGPUSymmetricContractionParams,
)


def chain_forward(
    x1: jax.Array,
    index: jax.Array,
    w_eff: jax.Array,
    plan: ReducedPlan,
    params: PallasMosaicGPUSymmetricContractionParams,
) -> jax.Array:
    """Power-expand `x1` and mix it index-wise, through the reduced plan.

    At `l_max = 3 -> 0e + 1o`, correlation 3, the plan takes ~300 FMAs per
    row and channel, against ~3.3k for the unreduced chain.

    One persistent CTA per SM and channel block copies every species'
    folded weights into shared memory. Each of its warpgroups then walks
    its own rows (:func:`.common.row_pipeline`) and writes the outputs
    straight to GMEM.

    Args:
        x1: `(batch, x1_dim, channels)` input features.
        index: `(batch,)` integer species of every row. Negative indices
            count from the end; out-of-range ones are clamped.
        w_eff: `(num_species, plan.weight_rows, channels)` folded weights.
        plan: The reduced plan of `params`.
        params: Static structure of the call.

    Returns:
        `(batch, params.out_dim, channels)`.
    """
    batch, x_dim, channels = x1.shape
    dtype = x1.dtype
    num_species = params.num_species
    launch = resolve_launch(params, plan, batch, channels, forward=True)
    channel_block, depth, n_wg = launch.channel_block, launch.depth, launch.warpgroups
    weight_rows = plan.weight_rows
    # The weights are copied flat, `(species * weight_rows, C)`.
    spans = table_copies(num_species * weight_rows, launch.n_channel_blocks)
    row_layout = plgpu.Layout.WG_STRIDED((channel_block,), vec_size=1)

    def kernel(x1_gmem, w_gmem, index_gmem, y_gmem):
        row_group, block_index = block_indices(launch.n_channel_blocks)
        channel_slice = pl.ds(block_index * channel_block, channel_block)
        wg = jax.lax.axis_index(THREAD_AXIS)
        row, num_steps = worker_walk(batch, row_group * n_wg + wg, launch.workers)
        zero_row = plgpu.layout_cast(jnp.zeros((channel_block,), dtype), row_layout)

        def load(ref):
            return plgpu.load(ref, layout=row_layout)

        def scoped(w_smem, x_smem, index_smem, table_barrier, barrier):
            @pl.when(wg == 0)
            def _load_table():
                copy_table(w_gmem, w_smem, channel_slice, table_barrier, spans)

            def read(slots):
                (x_slot,) = slots
                return {k: load(x_slot.at[k]) for k in plan.x_coords}

            def body(step, r, x, raw):
                s, _ = species_of(raw, num_species)

                def weight(block, mo, j):
                    j = block.weight_row + mo * block.columns + j
                    return load(w_smem.at[s * weight_rows + j])

                y = emit_row_fwd(plan, x, weight, zero_row)
                for k in range(params.out_dim):
                    y_gmem[r, k, channel_slice] = y.get(k, zero_row)

            row_pipeline(
                body,
                row=row,
                num_steps=num_steps,
                depth=depth,
                copies=[
                    (lambda r: x1_gmem.at[r, :, channel_slice], x_smem.at[wg]),
                ],
                barrier=lambda slot: barrier.at[wg * depth + slot],
                row_scalars=(index_gmem, index_smem.at[wg]),
                read=read,
                before_loop=lambda: plgpu.barrier_wait(table_barrier),
            )

        pl.run_scoped(
            scoped,
            plgpu.SMEM((num_species * weight_rows, channel_block), dtype),
            plgpu.SMEM((n_wg, depth, x_dim, channel_block), dtype),
            plgpu.SMEM((n_wg, depth, SCALAR_BLOCK), jnp.int32),
            plgpu.Barrier(num_arrivals=len(spans)),
            # A row's `x1` and its species block.
            plgpu.Barrier(num_arrivals=2, num_barriers=n_wg * depth),
            collective_axes=THREAD_AXIS,
        )

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((batch, params.out_dim, channels), dtype),
        # All stores go straight to GMEM, and the pipeline's warpgroup
        # barrier orders slot reads before refills.
        compiler_params=plgpu.CompilerParams(
            unsafe_no_auto_barriers=True,
            lowering_semantics=plgpu.LoweringSemantics.Lane,
        ),
        grid=(launch.row_groups * launch.n_channel_blocks,),
        grid_names=(BLOCK_AXIS,),
        num_threads=n_wg,
        thread_name=THREAD_AXIS,
        kernel_name="symmetric_contraction_mgpu_fwd",
    )(x1, w_eff.reshape(-1, channels), index.astype(jnp.int32))
