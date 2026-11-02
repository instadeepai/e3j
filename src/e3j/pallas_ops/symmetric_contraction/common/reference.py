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

"""Derivatives of the fused kernels, through a plain JAX reference."""

import jax


def reference_vjp(reference, x1, index, weights, dy, with_dw: bool):
    """`(dx1, dweights)`, the VJP of `reference(x1, index, weights) -> y`
    with `dy` (`dweights` is `None` without `with_dw`). Plain JAX, so
    differentiable in turn."""
    _, pullback = jax.vjp(
        lambda x1, weights: reference(x1, index, weights), x1, weights
    )
    dx1, dweights = pullback(dy)
    return dx1, dweights if with_dw else None


def reference_backward_fn(first_order, reference, with_dw: bool, kind: str):
    """Wrap a fused backward so it is differentiated through `reference`.

    `first_order(x1, weights, dy, index) -> (dx1, dweights)` is a fused
    backward with no derivative of its own (`dweights` is `None` without
    `with_dw`). `reference(x1, index, weights) -> y` is the same contraction
    in plain JAX; the returned function differentiates its VJP instead, so
    second derivatives (e.g. training on forces) run at the reference's
    speed. Without `reference`, they raise. `kind` names the backward in
    that error.
    """

    @jax.custom_vjp
    def backward(x1, weights, dy, index):
        return first_order(x1, weights, dy, index)

    def backward_fwd(x1, weights, dy, index):
        return backward(x1, weights, dy, index), (x1, weights, dy, index)

    def backward_bwd(residuals, cotangents):
        if reference is None:
            raise NotImplementedError(
                f"the fused SymmetricContraction's {kind} has no derivative "
                "of its own, and no `reference` to differentiate instead"
            )
        x1, weights, dy, index = residuals

        def first_order_reference(x1, weights, dy):
            return reference_vjp(reference, x1, index, weights, dy, with_dw)

        _, pullback = jax.vjp(first_order_reference, x1, weights, dy)
        return (*pullback(cotangents), None)

    backward.defvjp(backward_fwd, backward_bwd)
    return backward
