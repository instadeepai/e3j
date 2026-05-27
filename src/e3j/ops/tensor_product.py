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

from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import e3j_ops
import jax
import jax.experimental.custom_partitioning
import jax.numpy as jnp
import numpy as np
from jax import Array, custom_vjp
from jax.ffi import ffi_call
from numpy import int32

from e3j.ops.coef import Coef
from e3j.utils import config
from e3j.utils.exceptions import ShardingError
from e3j.utils.options import Layout, TPMode


@dataclass
class TensorProductParams:
    num_out: int
    mode: str | TPMode = TPMode.OUTER
    layout: str | Layout = Layout.LEADING_CHANNELS


# XLA-FFI Primitives


@partial(custom_vjp, nondiff_argnums=(3,))
def tensor_product(
    coef: Array,
    x: Array,
    y: Array,
    params: TensorProductParams,
) -> Array:
    """Sparse tensor product kernel using parallel scatter-reduction.

    Args:
        coef: Packed coefficient array (opaque idx_dtype vector),
            encoding (val, i, j, k) per nonzero CG entry.
        x: l.h.s. operand
        y: r.h.s. operand
        params: kernel parameters (num_out, mode, layout).

    Returns:
        The contraction of the sparse 3D coefficient array with x and y.
    """
    has_cx, has_cy = x.ndim > 2, y.ndim > 2

    # Parse layout and input channels
    layout = Layout.parse(params.layout)
    if layout in (Layout.LEADING_CHANNELS, Layout.E3NN):
        channels_x = x.shape[-2] if has_cx else 1
        channels_y = y.shape[-2] if has_cy else 1
    elif layout is Layout.TRAILING_CHANNELS:
        channels_x = x.shape[-1] if has_cx else 1
        channels_y = y.shape[-1] if has_cy else 1
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    num_out = params.num_out

    # Infer output channels
    mode = TPMode.parse(params.mode)
    if mode.name == "OUTER":
        if has_cx and has_cy:
            channels_z = [channels_x, channels_y]
        elif has_cx:
            channels_z = [channels_x]
        elif has_cy:
            channels_z = [channels_y]
        else:
            channels_z = []
    elif mode.name == "INNER":
        assert channels_x == channels_y
        channels_z = []
    elif mode.name == "MAP":
        assert channels_x == channels_y
        channels_z = [channels_x]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # We need to redefine a functions for vmap because
    # custom_vmap doesn't support static arguments.
    @jax.custom_batching.custom_vmap
    def _tensor_product_impl(coef, x, y):
        @jax.experimental.custom_partitioning.custom_partitioning
        def __tensor_product_impl(coef, x, y):
            num_rows = x.shape[0]

            # Infer output shape
            if layout is Layout.LEADING_CHANNELS:
                shape_out = (num_rows, *channels_z, num_out)
            elif layout is Layout.TRAILING_CHANNELS:
                shape_out = (num_rows, num_out, *channels_z)
            else:
                raise ValueError(f"Unsupported layout: {layout}")

            return ffi_call(
                "tensor_product",
                jax.ShapeDtypeStruct(shape_out, x.dtype),
            )(
                coef,
                x,
                y,
                num_out=int32(num_out),
                mode=int32(mode.value),
                layout=int32(layout.value),
                debug=int32(config().debug_level),
            )

        def partition(mesh, arg_shapes, result_shape):
            assert len(arg_shapes) == 3, "Expected three arguments: coef, x, y"
            assert len(arg_shapes[1].shape) in (
                2,
                3,
            ), f"Expected x rank 2 or 3, got {arg_shapes[1].shape}"
            assert len(arg_shapes[2].shape) in (
                2,
                3,
            ), f"Expected y rank 2 or 3, got {arg_shapes[2].shape}"
            coef_shape, x_shape, y_shape = arg_shapes

            result_sharding = result_shape.sharding

            return (
                mesh,
                __tensor_product_impl,
                (result_sharding),
                (coef_shape.sharding, x_shape.sharding, y_shape.sharding),
            )

        x_rule = "a b" if has_cx else "a"
        y_rule = "c d" if has_cy else "c"
        out_rule = " ".join(["e", "f", "g"][: len(channels_z) + 1])

        sharding_rule = f"u v, ... {x_rule}, ... {y_rule} -> ... {out_rule}"

        # Make sure coef are replicated
        __tensor_product_impl.def_partition(
            partition=partition,
            sharding_rule=sharding_rule,
            need_replication_factors=("u", "v"),
        )
        return __tensor_product_impl(coef, x, y)

    @_tensor_product_impl.def_vmap
    def _tensor_product_vmap_rule(axis_size, in_batched, coef, x, y):
        coef_b, x_b, y_b = in_batched
        if coef_b:
            raise ValueError("Batching over the coef argument is not supported.")
        # Typically both (or neither) of x, y are batched together, but
        # jacrev produces an asymmetric pattern where only the cotangent
        # is batched so we broadcast the unbatched operand here.
        if not x_b:
            x = jnp.broadcast_to(x[None], (axis_size,) + x.shape)
        if not y_b:
            y = jnp.broadcast_to(y[None], (axis_size,) + y.shape)
        num_rows = x.shape[1]
        x = x.reshape((axis_size * num_rows,) + x.shape[2:])
        y = y.reshape((axis_size * num_rows,) + y.shape[2:])
        out = _tensor_product_impl(coef, x, y)

        return out.reshape((axis_size, num_rows) + out.shape[1:]), True

    return _tensor_product_impl(coef, x, y)


# @partial(custom_vjp, nondiff_argnums=(5,))
def tensor_product_bwd(
    coef: Array,
    x: Array,
    y: Array,
    ct_z: Array,
    params: TensorProductParams,
) -> tuple[Array, Array]:
    """Backward tensor product kernel handler.

    This kernel loads output cotangents `ct_z` once
    to compute both input cotangents `ct_x, ct_y`.
    """

    # Parse forward mode and layout: backward modes parsed on C++ side
    mode = TPMode.parse(params.mode)
    layout = Layout.parse(params.layout)

    # Prepare transposed indices and concatenate two backward arrays
    #
    # TODO: lexsorting indices should reduce bank conflicts.
    with jax.ensure_compile_time_eval():
        c = Coef.unpack(coef, val_dtype="float32")
        val, idx = c.val, c.idx.T
        sigma_xzy = jnp.argsort(idx[1])
        val_xzy = val[sigma_xzy]
        idx_xzy = jnp.stack(
            [
                idx[1][sigma_xzy],
                idx[0][sigma_xzy],
                idx[2][sigma_xzy],
            ]
        )
        sigma_yzx = jnp.argsort(idx[2])
        val_yzx = val[sigma_yzx]
        idx_yzx = jnp.stack(
            [
                idx[2][sigma_yzx],
                idx[0][sigma_yzx],
                idx[1][sigma_yzx],
            ]
        )
        coef_bwd = jnp.stack(
            [
                Coef(val_xzy, idx_xzy.T).pack_jax(),
                Coef(val_yzx, idx_yzx.T).pack_jax(),
            ]
        )

    has_cx, has_cy = x.ndim > 2, y.ndim > 2

    @jax.custom_batching.custom_vmap
    def _tensor_product_bwd_impl(coef_bwd, x, y, ct_z):
        @jax.experimental.custom_partitioning.custom_partitioning
        def __tensor_product_bwd_impl(coef_bwd, x, y, ct_z):
            shapes_out = (
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(y.shape, y.dtype),
            )
            return ffi_call(
                "tensor_product_bwd",
                shapes_out,
            )(
                coef_bwd,
                x,
                y,
                ct_z,
                mode=int32(mode.value),
                layout=int32(layout.value),
                debug=int32(config().debug_level),
            )

        def partition(mesh, arg_shapes, result_shape):
            assert len(arg_shapes) == 4, "Expected four arguments: coef_bwd, x, y, ct_z"
            if len(arg_shapes[1].shape) not in (2, 3):
                raise ShardingError(
                    f"Expected x rank 2 or 3, got {arg_shapes[1].shape}"
                )
            if len(arg_shapes[2].shape) not in (2, 3):
                raise ShardingError(
                    f"Expected y rank 2 or 3, got {arg_shapes[2].shape}"
                )
            coef_shape, x_shape, y_shape, ct_z_shape = arg_shapes
            ct_x_shape, ct_y_shape = result_shape

            return (
                mesh,
                __tensor_product_bwd_impl,
                (ct_x_shape.sharding, ct_y_shape.sharding),
                (
                    coef_shape.sharding,
                    x_shape.sharding,
                    y_shape.sharding,
                    ct_z_shape.sharding,
                ),
            )

        x_rule = "a b" if has_cx else "a"
        y_rule = "c d" if has_cy else "c"
        z_rule = " ".join(["e", "f", "g"][: ct_z.ndim - 1])

        # coef_bwd is (2, K, numel): replicate all axes.
        sharding_rule = (
            f"o p q, ... {x_rule}, ... {y_rule}, ... {z_rule} "
            f"-> ... {x_rule}, ... {y_rule}"
        )

        __tensor_product_bwd_impl.def_partition(
            partition=partition,
            sharding_rule=sharding_rule,
            need_replication_factors=("o", "p", "q"),
        )

        return __tensor_product_bwd_impl(coef_bwd, x, y, ct_z)

    @_tensor_product_bwd_impl.def_vmap
    def _tensor_product_bwd_vmap_rule(axis_size, in_batched, coef_bwd, x, y, ct_z):
        coef_b, x_b, y_b, ct_z_b = in_batched
        if coef_b:
            raise NotImplementedError(
                "Batching over the coef_bwd argument is not supported."
            )
        if not x_b:
            x = jnp.broadcast_to(x[None], (axis_size,) + x.shape)
        if not y_b:
            y = jnp.broadcast_to(y[None], (axis_size,) + y.shape)
        if not ct_z_b:
            ct_z = jnp.broadcast_to(ct_z[None], (axis_size,) + ct_z.shape)
        num_rows = x.shape[1]
        x = x.reshape((axis_size * num_rows,) + x.shape[2:])
        y = y.reshape((axis_size * num_rows,) + y.shape[2:])
        ct_z = ct_z.reshape((axis_size * num_rows,) + ct_z.shape[2:])
        ct_x, ct_y = _tensor_product_bwd_impl(coef_bwd, x, y, ct_z)
        ct_x = ct_x.reshape((axis_size, num_rows) + ct_x.shape[1:])
        ct_y = ct_y.reshape((axis_size, num_rows) + ct_y.shape[1:])
        return (ct_x, ct_y), (True, True)

    return _tensor_product_bwd_impl(coef_bwd, x, y, ct_z)


# AD Rules


def _tensor_product_fwd(coef, x, y, params):
    z = tensor_product(coef, x, y, params)
    return z, (coef, x, y)


def _tensor_product_bwd(params, res, ct_z):
    """Backward tensor product rule.

    Backpropagate gradients following the Leibniz rule,
    with circular references to the `tensor_product` op.
    """
    coef, x, y = res
    # ct_coef: non-differentiable right now, as it would break equivariance.
    ct_coef = jnp.zeros_like(coef)

    layout = Layout.parse(params.layout)

    # Opt-in to `tensor_product_bwd` kernel handler for force inference
    if config().tensor_product_bwd and layout == Layout.TRAILING_CHANNELS:
        dx, dy = tensor_product_bwd(coef, x, y, ct_z, params)
        return (ct_coef, dx, dy)

    # TODO: move/remove once double backward rule has been implemented in
    #       terms of tensor_product and/or tensor_product_bwd.
    #
    # Fallback to 2 `tensor_product` kernel calls
    coef, x, y = res
    has_cx, has_cy = x.ndim > 2, y.ndim > 2

    mode = TPMode.parse(params.mode)
    # with jax.ensure_compile_time_eval():
    if mode.name == "OUTER" and has_cx and has_cy:
        raise NotImplementedError(
            "Cannot backpropagate through 'OUTER' tensor product "
            "with channel axes on both inputs."
        )

    if mode.name == "MAP":
        mode_x = "MAP"
        mode_y = "MAP"
    # u,u->1 yields u,1->u and 1,u->u
    if mode.name == "INNER":
        mode_x = "OUTER"
        mode_y = "OUTER"
    # u,1->u yields 1,u->u and u,u->1
    elif mode.name == "OUTER" and has_cx:
        mode_x = "OUTER"
        mode_y = "INNER"
    # 1,u->u yields u,u->1 and u,1->u
    elif mode.name == "OUTER" and has_cy:
        mode_x = "INNER"
        mode_y = "OUTER"
    # 1,1->1 yields 1,1->1 and 1,1->1
    # => for now, stick to 'OUTER' for performance
    elif mode.name == "OUTER":
        mode_x = "OUTER"
        mode_y = "OUTER"

    # Transposition of coefficients for backward pass
    #
    # NOTE: with the current OUTER mode restriction that only LHS should
    #       have channels, i.e. mode "(u, 1) -> u", permutation of operands
    #       should both put the output cotangent first. Circular permutations
    #       yield less supported mode "(1, u) -> 1".
    with jax.ensure_compile_time_eval():
        # Unpack to permute indices for transposed tensor products
        c = Coef.unpack(coef, val_dtype="float32")
        val, idx = c.val, c.idx

        # ct_x: transpose placing idx_x first, (dz, y) as operands
        sigma_x = jnp.argsort(idx[:, 1])
        idx_x = jnp.stack([idx[:, 1], idx[:, 0], idx[:, 2]], axis=-1)[sigma_x]
        val_x = val[sigma_x]

        # ct_y: transpose placing idx_y first, (dz, x) as operands
        sigma_y = jnp.argsort(idx[:, 2])
        idx_y = jnp.stack([idx[:, 2], idx[:, 0], idx[:, 1]], axis=-1)[sigma_y]
        val_y = val[sigma_y]

        # Repack transposed coefficients
        coef_x = Coef(val_x, idx_x).pack_jax()
        coef_y = Coef(val_y, idx_y).pack_jax()

    layout = Layout.parse(params.layout)
    if layout is Layout.LEADING_CHANNELS:
        num_x, num_y = x.shape[-1], y.shape[-1]
    elif layout is Layout.TRAILING_CHANNELS:
        num_x, num_y = x.shape[1], y.shape[1]

    params_x, params_y = (
        replace(params, num_out=num_x, mode=mode_x),
        replace(params, num_out=num_y, mode=mode_y),
    )

    ct_x = tensor_product(coef_x, ct_z, y, params_x)
    ct_y = tensor_product(coef_y, ct_z, x, params_y)

    return (ct_coef, ct_x, ct_y)


tensor_product.defvjp(_tensor_product_fwd, _tensor_product_bwd)


tensor_product.Params = TensorProductParams
