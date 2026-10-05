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

"""Differentiable tensor-product op for Pallas Mosaic GPU.

Both VJP rules call the wrapped ops instead of the raw kernels. A bare
`plgpu.kernel` has no differentiation rule, so going back through the wrapped
ops is what lets derivatives of any order work.
"""

import functools

import jax
import jax.numpy as jnp

from e3j.pallas_ops.tensor_product.mosaic_gpu.bwd import (
    _tensor_product_kernel_mosaic_gpu_bwd,
)
from e3j.pallas_ops.tensor_product.mosaic_gpu.fwd import (
    _tensor_product_kernel_mosaic_gpu_fwd,
)
from e3j.pallas_ops.tensor_product.mosaic_gpu.params import (
    PallasMosaicGPUTensorProductParams,
)


def _merge_leading_into_batch(a: jax.Array) -> jax.Array:
    """Fold a mapped axis into the batch: `(V, B, ...)` -> `(V*B, ...)`."""
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


def _split_batch_to_leading(a: jax.Array, v: int) -> jax.Array:
    """Inverse of :func:`_merge_leading_into_batch`."""
    return a.reshape((v, a.shape[0] // v) + a.shape[1:])


def _merged_operands(axis_size: int, in_batched, *operands) -> list[jax.Array]:
    """Broadcast the unmapped operands, then fold the mapped axis into the batch."""
    return [
        _merge_leading_into_batch(
            a if batched else jnp.broadcast_to(a[None], (axis_size,) + a.shape)
        )
        for a, batched in zip(operands, in_batched)
    ]


def _fwd_vmappable(
    x: jax.Array, y: jax.Array, params: PallasMosaicGPUTensorProductParams
) -> jax.Array:
    """Run the forward kernel, folding any mapped axis into its batch.

    `plgpu.kernel`'s own batching rule would add a second grid axis, and the
    CUDA grid caps that axis at 65535 blocks.
    """

    @jax.custom_batching.custom_vmap
    def _impl(x, y):
        return _tensor_product_kernel_mosaic_gpu_fwd(x, y, params)

    @_impl.def_vmap
    def _rule(axis_size, in_batched, x, y):
        # Call `_impl` again so that another vmap layer folds in the same way.
        z = _impl(*_merged_operands(axis_size, in_batched, x, y))
        return _split_batch_to_leading(z, axis_size), True

    return _impl(x, y)


def _bwd_vmappable(
    x: jax.Array,
    y: jax.Array,
    ct_z: jax.Array,
    params: PallasMosaicGPUTensorProductParams,
) -> tuple[jax.Array, jax.Array]:
    """Run the backward kernel, folding any mapped axis into its batch.

    See :func:`_fwd_vmappable` for why.
    """

    @jax.custom_batching.custom_vmap
    def _impl(x, y, ct_z):
        return _tensor_product_kernel_mosaic_gpu_bwd(x, y, ct_z, params)

    @_impl.def_vmap
    def _rule(axis_size, in_batched, x, y, ct_z):
        dx, dy = _impl(*_merged_operands(axis_size, in_batched, x, y, ct_z))
        return (
            _split_batch_to_leading(dx, axis_size),
            _split_batch_to_leading(dy, axis_size),
        ), (True, True)

    return _impl(x, y, ct_z)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _tensor_product_cotangents(
    x: jax.Array,
    y: jax.Array,
    ct_z: jax.Array,
    params: PallasMosaicGPUTensorProductParams,
) -> tuple[jax.Array, jax.Array]:
    """Compute both input cotangents, differentiable to any order."""
    return _bwd_vmappable(x, y, ct_z, params)


def _ct_fwd(
    x: jax.Array,
    y: jax.Array,
    ct_z: jax.Array,
    params: PallasMosaicGPUTensorProductParams,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    return _tensor_product_cotangents(x, y, ct_z, params), (x, y, ct_z)


def _ct_bwd(
    params: PallasMosaicGPUTensorProductParams,
    res: tuple[jax.Array, jax.Array, jax.Array],
    ct_dxdy: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Differentiate the backward pass with one backward and two forwards.

    `dy` depends on `x` as `dx` does on `y`, so one backward over `(ddx, ddy)`
    gives the cotangents of `x` and `y`. Both outputs are linear in `ct_z`, so
    its cotangent is `fwd(x, ddy) + fwd(ddx, y)`.
    """
    ddx, ddy = ct_dxdy
    x, y, ct_z = res
    dx, dy = _tensor_product_cotangents(ddx, ddy, ct_z, params)
    dct_z = tensor_product_pallas_mosaic_gpu(
        x, ddy, params
    ) + tensor_product_pallas_mosaic_gpu(ddx, y, params)
    return (dx, dy, dct_z)


_tensor_product_cotangents.defvjp(_ct_fwd, _ct_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def tensor_product_pallas_mosaic_gpu(
    x: jax.Array,
    y: jax.Array,
    params: PallasMosaicGPUTensorProductParams,
) -> jax.Array:
    """Evaluate the sparse tensor product with the Pallas Mosaic GPU kernels.

    Supports `jax.vmap` and reverse-mode differentiation to any order, but not
    forward mode (so no `jax.hessian`). Needs a Hopper (sm_90) or newer GPU,
    float32 operands and a multiple of 128 channels.

    Args:
        x: `(batch, x_dim, channels)` first operand.
        y: `(batch, y_dim, channels)` in `MAP` mode, `(batch, y_dim)` in
            `OUTER` mode, where it carries no channel axis.
        params: Kernel parameters carrying the sparse coefficients, the layout
            and the mixing mode.

    Returns:
        `(batch, z_dim, channels)` output.
    """
    return _fwd_vmappable(x, y, params)


def _tensor_product_fwd(
    x: jax.Array,
    y: jax.Array,
    params: PallasMosaicGPUTensorProductParams,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    # Call the wrapped op instead of the raw kernel (see the module docstring).
    return tensor_product_pallas_mosaic_gpu(x, y, params), (x, y)


def _tensor_product_bwd(
    params: PallasMosaicGPUTensorProductParams,
    res: tuple[jax.Array, jax.Array],
    ct_z: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    x, y = res
    return _tensor_product_cotangents(x, y, ct_z, params)


tensor_product_pallas_mosaic_gpu.defvjp(_tensor_product_fwd, _tensor_product_bwd)
