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

"""Forward pass of the sparse tensor product on Pallas Mosaic GPU."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from e3j.pallas_ops.tensor_product.mosaic_gpu.common import (
    BLOCK_AXIS,
    block_indices,
    check_staged_smem,
    pad_y_rows,
    resolve_channel_block,
    resolve_rows_per_block,
    used_input_coordinates,
    validate_operands,
    y_block_spec,
)
from e3j.pallas_ops.tensor_product.mosaic_gpu.params import (
    PallasMosaicGPUTensorProductParams,
)
from e3j.pallas_ops.utils.coef import group_coef_by_zi_then_xi
from e3j.pallas_ops.utils.mosaic_gpu import add_term, sum_terms


def _tensor_product_kernel_mosaic_gpu_fwd(
    x: jax.Array,
    y: jax.Array,
    params: PallasMosaicGPUTensorProductParams,
) -> jax.Array:
    """Forward kernel: contract the Clebsch-Gordan triples `(xi, yi, v)` of each `zi`.

    ```
    z[b, zi, c] = sum_(xi, yi, v) v * x[b, xi, c] * y[b, yi, c]   (MAP)
    z[b, zi, c] = sum_(xi, yi, v) v * x[b, xi, c] * y[b, yi]      (OUTER)
    ```

    Each CTA handles `rows_per_block` rows of one channel block, pipelined with
    `emit_pipeline`. The coefficients are unrolled at trace time and grouped by
    `(zi, xi)`, so each group is one FMA against `x[xi]`. Output rows are
    written straight from registers to GMEM. Staging them in SMEM was 2-5%
    faster, but would make the SMEM footprint grow with `z_dim`.
    """
    config = params.fwd_config
    shapes = validate_operands(x, y, params)
    channel_block = resolve_channel_block(shapes.channels, config.channel_block)
    rows_per_block = resolve_rows_per_block(shapes.batch, config.rows_per_block)
    n_channel_blocks = shapes.channels // channel_block
    # SMEM holds one row of `x` and one of `y`; the output is not staged.
    check_staged_smem(
        "forward",
        shapes.x_dim + (0 if shapes.outer else shapes.y_dim),
        channel_block,
        # `emit_pipeline` clamps its depth to the step count.
        min(config.pipeline_depth, rows_per_block),
        x.dtype.itemsize,
        extra_bytes=(shapes.y_dim_padded * x.dtype.itemsize if shapes.outer else 0),
    )

    coef_by_zi_then_xi = dict(group_coef_by_zi_then_xi(params.indices, params.values))
    used_xis, used_yis = used_input_coordinates(params)

    dtype = x.dtype
    y = pad_y_rows(y, shapes)
    # Layout of every value in the kernel body.
    row_layout = plgpu.Layout.WG_STRIDED((channel_block,), vec_size=1)

    def kernel(x_gmem, y_gmem, z_gmem):
        row_block, block_index = block_indices(n_channel_blocks)
        first_row = row_block * rows_per_block
        channel_slice = pl.ds(block_index * channel_block, channel_block)
        # Created here because the kernel body cannot capture array constants. A
        # bare `jnp.zeros` has a splat layout, so cast it to match the others.
        zero_row = plgpu.layout_cast(jnp.zeros((channel_block,), dtype), row_layout)

        def row_body(indices, x_smem, y_smem):
            (step,) = indices
            row = first_row + step
            # Each used coordinate is read from SMEM once, then reused.
            x_rows = {
                xi: plgpu.load(x_smem.at[xi], layout=row_layout) for xi in used_xis
            }
            if shapes.outer:
                # No channel axis: each `y[yi]` is an SMEM scalar.
                y_values = {yi: y_smem[yi] for yi in used_yis}
            else:
                y_values = {
                    yi: plgpu.load(y_smem.at[yi], layout=row_layout) for yi in used_yis
                }

            # Write every `zi`, including those no coefficient reaches, because
            # the output is not zero-initialized.
            for zi in range(shapes.z_dim):
                total = None
                for xi, paths in coef_by_zi_then_xi.get(zi, ()):
                    weight = sum_terms(value * y_values[yi] for yi, value in paths)
                    total = add_term(total, weight * x_rows[xi])
                z_gmem[row, zi, channel_slice] = zero_row if total is None else total

            # Warpgroup barrier. The pipeline's next copy overwrites the slot we
            # just read, and with auto barriers off nothing else orders the two.
            plgpu.commit_smem()

        index_map = lambda step: (first_row + step, 0, block_index)  # noqa: E731
        plgpu.emit_pipeline(
            row_body,
            grid=(rows_per_block,),
            in_specs=[
                plgpu.BlockSpec((None, shapes.x_dim, channel_block), index_map),
                y_block_spec(shapes, channel_block, first_row, block_index),
            ],
            max_concurrent_steps=config.pipeline_depth,
        )(x_gmem, y_gmem)

    return plgpu.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct(
            (shapes.batch, shapes.z_dim, shapes.channels), dtype
        ),
        # Stores go straight to GMEM, so `commit_smem` is the only barrier needed.
        compiler_params=plgpu.CompilerParams(
            unsafe_no_auto_barriers=True,
            lowering_semantics=plgpu.LoweringSemantics.Lane,
        ),
        # Flat grid; each CTA splits its index with `block_indices`.
        grid=(shapes.batch // rows_per_block * n_channel_blocks,),
        grid_names=(BLOCK_AXIS,),
        kernel_name="tensor_product_mgpu_fwd",
    )(x, y)
