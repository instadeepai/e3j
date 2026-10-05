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

"""Backward pass of the sparse tensor product on Pallas Mosaic GPU."""

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
from e3j.pallas_ops.utils.coef import group_coef_by_zi_then_xi, group_coef_by_zi_then_yi
from e3j.pallas_ops.utils.mosaic_gpu import add_term, sum_terms

#: `OUTER`'s `dy` tile is read back with a layout that tiles rows by 8, so its
#: row count is rounded up to a multiple of this.
_DY_TILE_ROW_GRANULARITY = 8


def _or_zero(row: jax.Array | None, zero_row: jax.Array) -> jax.Array:
    """Return `row`, or `zero_row` if no coefficient path reached it."""
    return zero_row if row is None else row


def _tensor_product_kernel_mosaic_gpu_bwd(
    x: jax.Array,
    y: jax.Array,
    ct_z: jax.Array,
    params: PallasMosaicGPUTensorProductParams,
) -> tuple[jax.Array, jax.Array]:
    """Backward kernel: both input cotangents in one pass over `ct_z`.

    ```
    dx[b, xi, c] = sum_(zi, yi, v) v * ct_z[b, zi, c] * y[b, yi, c]   (MAP)
    dy[b, yi, c] = sum_(zi, xi, v) v * ct_z[b, zi, c] * x[b, xi, c]   (MAP)

    dx[b, xi, c] = sum_(zi, yi, v) v * ct_z[b, zi, c] * y[b, yi]      (OUTER)
    dy[b, yi]    = sum_c sum_(zi, xi, v) v * ct_z[b, zi, c] * x[b, xi, c]  (OUTER)
    ```

    Blocking is the same as in the forward. Each `ct_z[zi]` row is read once and
    used for both `dx` and `dy`, whose accumulators stay in registers for the
    whole row. In `OUTER` mode, `dy` is reduced over the channel lanes through
    an SMEM tile (:func:`_reduce_dy`). Each channel block writes its own partial
    sum and the partial sums are added after the kernel, which avoids atomics
    and keeps the result deterministic.
    """
    config = params.bwd_config
    shapes = validate_operands(x, y, params)
    if ct_z.shape != (shapes.batch, shapes.z_dim, shapes.channels):
        raise ValueError(
            f"ct_z must be {(shapes.batch, shapes.z_dim, shapes.channels)}, "
            f"got {ct_z.shape}"
        )
    channel_block = resolve_channel_block(shapes.channels, config.channel_block)
    rows_per_block = resolve_rows_per_block(shapes.batch, config.rows_per_block)
    n_channel_blocks = shapes.channels // channel_block

    # The same triples folded both ways: by `xi` for `dx`, by `yi` for `dy`.
    xi_paths = dict(group_coef_by_zi_then_xi(params.indices, params.values))
    yi_paths = dict(group_coef_by_zi_then_yi(params.indices, params.values))
    # Both groupings have the same keys today. The union keeps this correct if
    # one of them ever starts dropping entries.
    used_zis = sorted(set(xi_paths) | set(yi_paths))
    used_xis, used_yis = used_input_coordinates(params)

    dtype = x.dtype
    if shapes.batch == 0:
        # A zero-row `dy` view fails Mosaic GPU's static slice check.
        return jnp.zeros_like(x), jnp.zeros_like(y)

    y = pad_y_rows(y, shapes)
    x_dim, y_dim, outer = shapes.x_dim, shapes.y_dim, shapes.outer
    row_layout = plgpu.Layout.WG_STRIDED((channel_block,), vec_size=1)
    dy_tile_rows = (
        -(-y_dim // _DY_TILE_ROW_GRANULARITY) * _DY_TILE_ROW_GRANULARITY if outer else 0
    )
    dy_tile_layout = (
        plgpu.Layout.SMEM_GMEM_COPY((dy_tile_rows, channel_block), dtype, 128)
        if outer
        else None
    )
    # SMEM holds one row each of `x`, `y` and `ct_z`, plus the `dy` tile in
    # `OUTER` mode.
    check_staged_smem(
        "backward",
        x_dim + shapes.z_dim + (0 if outer else y_dim),
        channel_block,
        # `emit_pipeline` clamps its depth to the step count.
        min(config.pipeline_depth, rows_per_block),
        dtype.itemsize,
        extra_bytes=(
            (dy_tile_rows * channel_block + shapes.y_dim_padded) * dtype.itemsize
            if outer
            else 0
        ),
    )

    def kernel(x_gmem, y_gmem, ct_gmem, dx_gmem, dy_gmem, *scratch):
        dy_rows_smem = scratch[0] if outer else None
        row_block, block_index = block_indices(n_channel_blocks)
        first_row = row_block * rows_per_block
        channel_slice = pl.ds(block_index * channel_block, channel_block)
        zero_row = plgpu.layout_cast(jnp.zeros((channel_block,), dtype), row_layout)

        def row_body(indices, x_smem, y_smem, ct_smem):
            (step,) = indices
            row = first_row + step
            x_rows = {
                xi: plgpu.load(x_smem.at[xi], layout=row_layout) for xi in used_xis
            }
            if outer:
                y_values = {yi: y_smem[yi] for yi in used_yis}
            else:
                y_values = {
                    yi: plgpu.load(y_smem.at[yi], layout=row_layout) for yi in used_yis
                }

            dx: list[jax.Array | None] = [None] * x_dim
            dy: list[jax.Array | None] = [None] * y_dim
            for zi in used_zis:
                ct_row = plgpu.load(ct_smem.at[zi], layout=row_layout)
                for xi, paths in xi_paths.get(zi, ()):
                    weight = sum_terms(value * y_values[yi] for yi, value in paths)
                    dx[xi] = add_term(dx[xi], weight * ct_row)
                for yi, paths in yi_paths.get(zi, ()):
                    weighted_x = sum_terms(value * x_rows[xi] for xi, value in paths)
                    dy[yi] = add_term(dy[yi], weighted_x * ct_row)

            for xi in range(x_dim):
                dx_gmem[row, xi, channel_slice] = _or_zero(dx[xi], zero_row)
            if outer:
                _reduce_dy(
                    row,
                    dy,
                    dy_rows_smem,
                    dy_gmem.at[block_index],
                    dy_tile_layout,
                    zero_row,
                )
            else:
                for yi in range(y_dim):
                    dy_gmem[row, yi, channel_slice] = _or_zero(dy[yi], zero_row)

            # The SMEM reads must finish before the pipeline refills the slot. In
            # `OUTER` mode this also keeps `_reduce_dy`'s read-back ahead of the
            # next row's write to the tile.
            plgpu.commit_smem()

        index_map = lambda step: (first_row + step, 0, block_index)  # noqa: E731
        plgpu.emit_pipeline(
            row_body,
            grid=(rows_per_block,),
            in_specs=[
                plgpu.BlockSpec((None, x_dim, channel_block), index_map),
                y_block_spec(shapes, channel_block, first_row, block_index),
                plgpu.BlockSpec((None, shapes.z_dim, channel_block), index_map),
            ],
            max_concurrent_steps=config.pipeline_depth,
        )(x_gmem, y_gmem, ct_gmem)

    # In `OUTER` mode `dy` holds one padded partial sum per channel block. They
    # are summed and trimmed after the kernel.
    dy_shape = (
        (n_channel_blocks, shapes.batch, shapes.y_dim_padded)
        if outer
        else (shapes.batch, y_dim, shapes.channels)
    )
    dx, dy = plgpu.kernel(
        kernel,
        out_type=(
            jax.ShapeDtypeStruct((shapes.batch, x_dim, shapes.channels), dtype),
            jax.ShapeDtypeStruct(dy_shape, dtype),
        ),
        scratch_types=(
            [plgpu.SMEM((dy_tile_rows, channel_block), dtype)] if outer else []
        ),
        compiler_params=plgpu.CompilerParams(
            unsafe_no_auto_barriers=True,
            lowering_semantics=plgpu.LoweringSemantics.Lane,
        ),
        # Flat grid; each CTA splits its index with `block_indices`.
        grid=(shapes.batch // rows_per_block * n_channel_blocks,),
        grid_names=(BLOCK_AXIS,),
        kernel_name="tensor_product_mgpu_bwd",
    )(x, y, ct_z)
    if outer:
        dy = dy[0] if n_channel_blocks == 1 else dy.sum(axis=0)
        dy = dy[:, :y_dim]
    return dx, dy


def _reduce_dy(
    row: jax.Array,
    dy: list[jax.Array | None],
    dy_rows_smem,
    dy_gmem,
    dy_tile_layout,
    zero_row: jax.Array,
) -> None:
    """Sum `OUTER`'s `dy` over the channel lanes and store the row.

    The rows are written to an SMEM tile and read back with a tiled layout, so a
    single `jnp.sum` reduces every coordinate at once. The store writes the
    whole tile, so rows past `y_dim` are zeroed.
    """
    rows = dy_rows_smem.shape[0]
    for yi in range(rows):
        dy_rows_smem[yi, :] = _or_zero(dy[yi] if yi < len(dy) else None, zero_row)
    plgpu.commit_smem()
    dy_gmem[row, 0:rows] = jnp.sum(
        plgpu.load(dy_rows_smem, layout=dy_tile_layout), axis=-1
    )
