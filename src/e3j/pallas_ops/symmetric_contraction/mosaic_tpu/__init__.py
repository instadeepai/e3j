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

"""Fused symmetric contraction on Pallas Mosaic TPU."""

import functools

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from e3j.pallas_ops.symmetric_contraction.common.batching import vmappable
from e3j.pallas_ops.symmetric_contraction.common.params import validate_operands
from e3j.pallas_ops.symmetric_contraction.common.plan import build_plan, fold_weights
from e3j.pallas_ops.symmetric_contraction.common.reference import reference_backward_fn
from e3j.pallas_ops.symmetric_contraction.mosaic_tpu.bwd import (
    _symmetric_contraction_kernel_mosaic_tpu_bwd,
)
from e3j.pallas_ops.symmetric_contraction.mosaic_tpu.fwd import (
    LANES,
    SUBLANES,
    _symmetric_contraction_kernel_mosaic_tpu_fwd,
    symmetric_contraction_reference,
)
from e3j.pallas_ops.symmetric_contraction.mosaic_tpu.params import (
    PallasMosaicTPUSymmetricContractionParams,
)
from e3j.pallas_ops.utils.mosaic_tpu import round_up

__all__ = [
    "symmetric_contraction_pallas_mosaic_tpu",
    "PallasMosaicTPUSymmetricContractionParams",
]


def symmetric_contraction_pallas_mosaic_tpu(
    x1: jax.Array,
    index: jax.Array,
    weights: list[jax.Array],
    params: PallasMosaicTPUSymmetricContractionParams,
) -> jax.Array:
    """Fused symmetric contraction ``LinearIndexwise(sort(powers(x1)), index, weights)``.

    The weights are folded outside the custom VJP, so JAX differentiates
    the fold itself.

    Args:
        x1: ``(batch, source.dim, mul)`` float32 input features.
        index: ``(batch,)`` integer species of every row.
        weights: One ``(num_species, mul, m_out, m_in)`` array per
            ``params.blocks`` (degenerate blocks left out).
        params: Fused kernel parameters.

    Returns:
        ``(batch, target.dim, mul)`` output.
    """
    weights = list(weights)
    validate_operands(x1, index, weights, params)
    # The kernels slice in float32 (8, 128) tiles.
    if x1.dtype != jnp.float32:
        raise ValueError(f"x1 must be float32, got {x1.dtype}")
    w = fold_weights(weights, build_plan(params))
    return _symmetric_contraction_folded(x1, w, index, params)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _symmetric_contraction_folded(
    x: jax.Array,
    w: jax.Array,
    species: jax.Array,
    params: PallasMosaicTPUSymmetricContractionParams,
) -> jax.Array:
    """The kernels, on folded weights ``w`` ``(num_species, plan.weight_rows,
    mul)``."""
    return _fwd_vmappable(x, w, species, params)


# VMEM the vmapped backward may give its per-species `dw` sums.
_VMAP_DW_BUDGET = 8 * 1024 * 1024


def _fwd_vmappable(x, w, species, params):
    """Forward kernel call; vmap folds into the batch (and species) axes."""
    return vmappable(
        lambda w, species, x: _symmetric_contraction_kernel_mosaic_tpu_fwd(
            x, w, species, params
        )
    )(w, species, x)


def _bwd_vmappable(x, w, ct, species, params, with_dw):
    """Backward kernel call, vmappable as :func:`_fwd_vmappable`.

    Every vmap lane needs its own `dw`, so with `with_dw` the weights are
    tiled per lane, in chunks of lanes whose per-species `dw` sums fit in
    VMEM.
    """
    species_bytes = 2 * round_up(w.shape[-2], SUBLANES) * LANES * 4
    return vmappable(
        lambda w, species, x, ct: _symmetric_contraction_kernel_mosaic_tpu_bwd(
            x, w, ct, species, params, with_dw
        ),
        tile_w=with_dw,
        fits=lambda num_species: num_species * species_bytes <= _VMAP_DW_BUDGET,
    )(w, species, x, ct)


def _symmetric_contraction_pallas_mosaic_tpu_fwd(x, w, species, params):
    # Skip `dw` when the weights are not differentiated.
    with_dw = w.perturbed
    x, w, species = x.value, w.value, species.value
    out = _symmetric_contraction_folded(x, w, species, params)
    return out, (x, w, species, with_dw)


def _symmetric_contraction_pallas_mosaic_tpu_bwd(params, res, ct):
    x, w, species, with_dw = res
    if isinstance(ct, SymbolicZero):
        return None, None, None
    # Second derivatives go through the reference.
    backward = reference_backward_fn(
        lambda x, w, ct, species: _bwd_vmappable(x, w, ct, species, params, with_dw),
        lambda x, species, w: symmetric_contraction_reference(x, w, species, params),
        with_dw=with_dw,
        kind="backward",
    )
    return (*backward(x, w, ct, species), None)


_symmetric_contraction_folded.defvjp(
    _symmetric_contraction_pallas_mosaic_tpu_fwd,
    _symmetric_contraction_pallas_mosaic_tpu_bwd,
    symbolic_zeros=True,
)
