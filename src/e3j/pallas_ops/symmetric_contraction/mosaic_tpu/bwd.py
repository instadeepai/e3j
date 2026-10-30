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

"""Backward pass of the fused symmetric contraction on Pallas Mosaic TPU.

From the output cotangent ``do``, computes ``dx`` and the folded weights'
cotangent ``dw`` by walking the reduced plan backwards
(:func:`~e3j.pallas_ops.symmetric_contraction.common.unroll.emit_row_bwd`).
``dw`` is summed per species in VMEM, one table per core; JAX adds the
cores' tables and unfolds the result.
"""

import jax
import jax.numpy as jnp

from e3j.pallas_ops.symmetric_contraction.common.plan import build_plan
from e3j.pallas_ops.symmetric_contraction.common.unroll import emit_row_bwd
from e3j.pallas_ops.symmetric_contraction.mosaic_tpu.fwd import (
    LANES,
    SUBLANES,
    block_size,
    check_inputs,
    launch,
)
from e3j.pallas_ops.symmetric_contraction.mosaic_tpu.params import (
    BATCH_BLOCK_SIZE_CANDIDATES,
    PallasMosaicTPUSymmetricContractionParams,
)
from e3j.pallas_ops.utils.mosaic_tpu import round_up


def _symmetric_contraction_kernel_mosaic_tpu_bwd(
    x: jax.Array,
    w: jax.Array,
    do: jax.Array,
    species: jax.Array,
    params: PallasMosaicTPUSymmetricContractionParams,
    with_dw: bool = True,
) -> tuple[jax.Array, jax.Array | None]:
    """Fused backward symmetric contraction; ``dw`` is ``None`` without
    ``with_dw``."""
    plan = build_plan(params)
    check_inputs(x, w, species, params, plan)
    x_dim = x.shape[1]
    # Per batch row: double-buffered x, do, weight rows and dx (plus two dw
    # staging buffers with `with_dw`).
    rows = 2 * (2 * x_dim + params.out_dim + plan.weight_rows)
    # Fixed: the double-buffered per-species dw sums, one lane tile wide.
    dw_bytes = 2 * params.num_species * round_up(plan.weight_rows, SUBLANES)
    dw_bytes *= LANES * 4
    bbs = params.bwd_batch_block_size or block_size(
        rows + 2 * plan.weight_rows * with_dw,
        BATCH_BLOCK_SIZE_CANDIDATES,
        reserved=dw_bytes * with_dw,
    )

    def body(xs, wb):
        xb, dob = xs
        zero = jnp.zeros_like(xb[0])
        dw_rows = [zero] * len(wb) if with_dw else None

        def emit_dw(block, mo, j, value):
            if value is not None:
                dw_rows[block.weight_row + mo * block.columns + j] = value

        dx = emit_row_bwd(
            plan,
            {k: xb[k] for k in plan.x_coords},
            lambda block, mo, m: dob[block.offset + mo * block.dim + m],
            lambda block, mo, j: wb[block.weight_row + mo * block.columns + j],
            emit_dw if with_dw else None,
            zero,
        )
        outs = [[dx.get(k, zero) for k in range(x_dim)]]
        if with_dw:
            outs.append(dw_rows)
        return outs

    dx, *dw = launch(
        body,
        (x, do),
        w,
        species,
        [x_dim],
        bbs,
        (
            "symmetric_contraction_mtpu_bwd"
            if with_dw
            else "symmetric_contraction_mtpu_bwd_dx"
        ),
        with_dw=with_dw,
    )
    return dx, dw[0] if with_dw else None
