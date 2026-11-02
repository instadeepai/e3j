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

"""Entry point of the fused Pallas Mosaic GPU SymmetricContraction.

The forward (`fwd`) and backward (`bwd`) kernels each evaluate the reduced
plan in one launch. When the forward's tables do not fit in shared memory
(many species), the plain JAX `reference` runs instead. When only the
backward's do not, the forward still runs fused and derivatives go through
`reference`, so calls never differentiated (inference) stay fused.

The backward is itself differentiable (:func:`chain_backward_fn`), so
training on forces runs fused too. Where the double backward does not fit,
second derivatives go through `reference`.

Every kernel is vmappable, the vmap axis folded into its batch and split
across the context mesh (:mod:`..common.batching`, shared with Mosaic TPU).
"""

import dataclasses
import functools

import jax
from jax.custom_derivatives import SymbolicZero

from e3j.pallas_ops.symmetric_contraction.common.batching import vmappable
from e3j.pallas_ops.symmetric_contraction.common.plan import build_plan, fold_weights
from e3j.pallas_ops.symmetric_contraction.common.reference import (
    reference_backward_fn,
    reference_vjp,
)
from e3j.pallas_ops.symmetric_contraction.mosaic_gpu import bwd, fwd
from e3j.pallas_ops.symmetric_contraction.mosaic_gpu.common import validate_operands
from e3j.pallas_ops.symmetric_contraction.mosaic_gpu.params import (
    PallasMosaicGPUSymmetricContractionParams,
)
from e3j.pallas_ops.utils.mosaic_gpu import add_term


def _fits(params: PallasMosaicGPUSymmetricContractionParams, *args, **kwargs) -> bool:
    """Whether the plan builds and `bwd.resolve_launch(*args, **kwargs)` fits."""
    try:
        bwd.resolve_launch(params, build_plan(params), *args, **kwargs)
    except RuntimeError:  # includes NotImplementedError
        return False
    return True


def chain_supported(
    params: PallasMosaicGPUSymmetricContractionParams,
    batch: int,
    channels: int,
    direction: str,
    with_dw: bool = True,
) -> bool:
    """Whether the `direction` (`fwd` or `bwd`) kernel can run this call: the
    plan must build and its tables fit in shared memory. Without `with_dw`,
    the backward computes `dx` alone, and needs no weight cotangent tables."""
    return _fits(params, batch, channels, with_dw=with_dw, forward=direction == "fwd")


def double_backward_supported(
    params: PallasMosaicGPUSymmetricContractionParams, batch: int, channels: int
) -> bool:
    """Whether :func:`.bwd.chain_double_backward` fits in shared memory (it
    stages more per row than the backward)."""
    staged = 2 * params.stage_dims[0] + params.out_dim
    return _fits(params, batch, channels, staged=staged)


def _vmappable(kernel, params, *, tile_w=False, **launch):
    """`kernel(x1, index, w_eff, *rows, plan=, params=)` (a :mod:`.fwd` or
    :mod:`.bwd` chain kernel), vmappable by
    :func:`.common.batching.vmappable` as `(w_eff, index, x1, *rows)`.

    The kernels size their tables by `params.num_species`, so a call on
    weights folded across vmap lanes gets params with as many species, in
    chunks of lanes whose tables fit in shared memory (`launch` as for
    :func:`.bwd.resolve_launch`)."""
    plan = build_plan(params)

    def with_species(num_species):
        if num_species == params.num_species:
            return params
        return dataclasses.replace(params, num_species=num_species)

    def call(w_eff, index, x1, *rows):
        # The plan does not depend on the species count.
        p = with_species(w_eff.shape[0])
        return kernel(x1, index, w_eff, *rows, plan=plan, params=p)

    def run(w_eff, index, x1, *rows):
        batch, channels = x1.shape[0], w_eff.shape[-1]

        def fits(num_species):
            try:
                bwd.resolve_launch(
                    with_species(num_species), plan, batch, channels, **launch
                )
            except RuntimeError:  # includes NotImplementedError
                return False
            return True

        return vmappable(call, tile_w=tile_w, fits=fits)(w_eff, index, x1, *rows)

    return run


def _chain_forward(x1, index, w_eff, plan, params):
    """:func:`.fwd.chain_forward`, vmappable."""
    del plan  # `_vmappable` takes it from the (cached) `build_plan`.
    return _vmappable(fwd.chain_forward, params, forward=True)(w_eff, index, x1)


def _chain_backward(x1, index, w_eff, dy, plan, params, with_dw=True, dx_addend=None):
    """:func:`.bwd.chain_backward`, vmappable."""
    del plan

    def kernel(x1, index, w_eff, dy, *addend, **static):
        return bwd.chain_backward(
            x1,
            index,
            w_eff,
            dy,
            with_dw=with_dw,
            dx_addend=addend[0] if addend else None,
            **static,
        )

    if dx_addend is None:
        rows, staged = (dy,), None
    else:
        # The addend is staged with the inputs, like `dx1`'s cotangent in
        # the double backward.
        rows, staged = (dy, dx_addend), 2 * params.stage_dims[0] + params.out_dim
    call = _vmappable(kernel, params, tile_w=with_dw, with_dw=with_dw, staged=staged)
    return call(w_eff, index, x1, *rows)


def _chain_double_backward(x1, index, w_eff, dy, u, plan, params, with_dw=True):
    """:func:`.bwd.chain_double_backward`, vmappable."""
    del plan
    kernel = functools.partial(bwd.chain_double_backward, with_dw=with_dw)
    staged = 2 * params.stage_dims[0] + params.out_dim
    call = _vmappable(kernel, params, tile_w=with_dw, with_dw=with_dw, staged=staged)
    return call(w_eff, index, x1, dy, u)


def _chain_grads(dx1, dw_eff, plan, params):
    """`(dx1, dweights)` from the backward's `(dx1, dW_eff)`."""
    if dw_eff is None:
        return dx1, None
    return dx1, bwd.unfold_weights(dw_eff, plan, params)


def chain_backward_fn(params: PallasMosaicGPUSymmetricContractionParams, with_dw: bool):
    """The backward `(x1, w_eff, dy, index) -> dx1` (and `dW_eff` with
    `with_dw`), as a differentiable function.

    The cotangent `u` of `dx1` goes through :func:`.bwd.chain_double_backward`.
    Since everything is linear in the weights, the cotangent `v` of `dW_eff`
    is just a forward (`d dy += f(x1, v)`) and a backward
    (`d x1 += J(x1, v)^T dy`) with `v` as the weights.
    """
    plan = build_plan(params)

    @jax.custom_vjp
    def backward(x1, w_eff, dy, index):
        dx1, dw_eff = _chain_backward(
            x1, index, w_eff, dy, plan, params, with_dw=with_dw
        )
        return (dx1, dw_eff) if with_dw else dx1

    def backward_fwd(x1, w_eff, dy, index):
        # `dw` is None when the weights are not differentiated.
        dw = () if w_eff.perturbed else None
        x1, w_eff, dy, index = x1.value, w_eff.value, dy.value, index.value
        return backward(x1, w_eff, dy, index), (x1, w_eff, dy, index, dw)

    def backward_bwd(residuals, cotangents):
        x1, w_eff, dy, index, dw = residuals
        u, v = cotangents if with_dw else (cotangents, None)
        d_x1 = d_w_eff = d_dy = None
        if not isinstance(u, SymbolicZero):
            d_x1, d_w_eff, d_dy = _chain_double_backward(
                x1, index, w_eff, dy, u, plan, params, with_dw=dw is not None
            )
        if v is not None and not isinstance(v, SymbolicZero):
            d_dy = add_term(d_dy, _chain_forward(x1, index, v, plan, params))
            d_x1, _ = _chain_backward(
                x1, index, v, dy, plan, params, with_dw=False, dx_addend=d_x1
            )
        return d_x1, d_w_eff, d_dy, None

    backward.defvjp(backward_fwd, backward_bwd, symbolic_zeros=True)
    return backward


def symmetric_contraction_pallas_mosaic_gpu(
    x1: jax.Array,
    index: jax.Array,
    weights: list[jax.Array],
    params: PallasMosaicGPUSymmetricContractionParams,
    reference=None,
) -> jax.Array:
    """Fused power expansion + index-wise linear on Mosaic GPU.

    No power is ever written to GMEM. Differentiable twice in `x1` and
    `weights` (see the module docstring). Where the forward does not fit
    (:func:`chain_supported`), the whole call runs `reference`; where only
    the backward does not, `reference` differentiates the fused forward.

    Args:
        x1: `(batch, x1_dim, channels)` input features.
        index: `(batch,)` integer species of every row.
        weights: One `(num_species, channels, m_out, m_in)` array per
            `params.blocks` (degenerate blocks left out).
        params: Static structure, from
            :attr:`e3j.core.SymmetricContraction.mosaic_gpu_params`.
        reference: Optional `(x1, index, weights) -> y`, the same contraction
            in plain JAX, for what the kernels cannot run. Without it, that
            raises `NotImplementedError`.

    Returns:
        `(batch, params.out_dim, channels)`, zero at
        `params.zero_coordinates`.
    """
    weights = list(weights)
    validate_operands(x1, index, weights, params)
    batch, _, channels = x1.shape
    if not chain_supported(params, batch, channels, "fwd"):
        if reference is None:
            raise NotImplementedError(
                "the fused SymmetricContraction's tables do not fit in shared "
                "memory at this size, and no `reference` runs it instead"
            )
        return reference(x1, index, weights)
    plan = functools.partial(build_plan, params)
    # Whether second derivatives run fused. The double backward stages more
    # per row than the backward, so where it fits, the backward does too.
    fused_second = double_backward_supported(params, batch, channels)

    def forward(x1, weights, index, w_eff=None):
        if w_eff is None:
            w_eff = fold_weights(weights, plan())
        return _chain_forward(x1, index, w_eff, plan(), params)

    @jax.custom_vjp
    def contraction(x1, weights, index):
        return forward(x1, weights, index)

    @jax.custom_vjp
    def with_x1(x1, weights, index):
        """`(contraction(x1, weights, index), x1)`.

        Passing `x1` through here makes its second-order cotangent arrive
        with `dy`, so one backward kernel adds both.
        """
        return forward(x1, weights, index), x1

    def with_x1_fwd(x1, weights, index):
        dw = () if any(w.perturbed for w in weights) else None
        x1, weights, index = x1.value, [w.value for w in weights], index.value
        w_eff = fold_weights(weights, plan())
        return (forward(x1, weights, index, w_eff), x1), (x1, w_eff, index, dw)

    def with_x1_bwd(residuals, cotangents):
        x1, w_eff, index, dw = residuals
        dy, d_x1 = cotangents
        d_x1 = None if isinstance(d_x1, SymbolicZero) else d_x1
        if isinstance(dy, SymbolicZero):
            return d_x1, None, None
        dx1, dw_eff = _chain_backward(
            x1, index, w_eff, dy, plan(), params, dw is not None, d_x1
        )
        return *_chain_grads(dx1, dw_eff, plan(), params), None

    def contraction_fwd(x1, weights, index):
        # `dw` is None when the weights are not differentiated.
        dw = () if any(w.perturbed for w in weights) else None
        x1, weights, index = x1.value, [w.value for w in weights], index.value
        # This may itself be differentiated, so only call custom VJPs here,
        # never a bare kernel.
        if fused_second:
            y, x1 = with_x1(x1, weights, index)
            return y, (x1, fold_weights(weights, plan()), index, dw)
        return contraction(x1, weights, index), (x1, weights, index, dw)

    def first_order(x1, weights, dy, index, with_dw):
        dx1, dw_eff = _chain_backward(
            x1, index, fold_weights(weights, plan()), dy, plan(), params, with_dw
        )
        return _chain_grads(dx1, dw_eff, plan(), params)

    def contraction_bwd(residuals, dy):
        x1, w, index, dw = residuals
        with_dw = dw is not None
        if isinstance(dy, SymbolicZero):
            return None, None, None
        if fused_second:
            # Differentiable, for second derivatives.
            out = chain_backward_fn(params, with_dw)(x1, w, dy, index)
            dx1, dw_eff = out if with_dw else (out, None)
            return *_chain_grads(dx1, dw_eff, plan(), params), None
        if not chain_supported(params, batch, channels, "bwd", with_dw):
            if reference is None:
                raise NotImplementedError(
                    "the fused SymmetricContraction's backward tables do not fit "
                    "in shared memory at this size, and no `reference` runs it "
                    "instead"
                )
            # Plain JAX, so second derivatives go through it too.
            dx1, dweights = reference_vjp(reference, x1, index, w, dy, with_dw)
            return dx1, dweights, None
        kind = (
            "chain backward at this size (its double backward does not fit in "
            "shared memory)"
        )
        dx1, dweights = reference_backward_fn(
            functools.partial(first_order, with_dw=with_dw), reference, with_dw, kind
        )(x1, w, dy, index)
        return dx1, dweights, None

    with_x1.defvjp(with_x1_fwd, with_x1_bwd, symbolic_zeros=True)
    contraction.defvjp(contraction_fwd, contraction_bwd, symbolic_zeros=True)
    return contraction(x1, weights, index)
