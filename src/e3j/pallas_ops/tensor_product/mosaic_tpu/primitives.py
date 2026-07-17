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

"""Differentiable tensor-product op for Pallas Mosaic TPU.

Wraps the raw forward/backward kernels (`fwd.py` / `bwd.py`) into the public
`tensor_product_pallas_mosaic_tpu` op and registers its JAX transformation
rules: custom VJP (incl. double-backward for force training), custom vmap, and
shard_map-based data parallelism over the device mesh.
"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from e3j.pallas_ops.tensor_product.mosaic_tpu.bwd import (
    _tensor_product_kernel_mosaic_tpu_bwd,
)
from e3j.pallas_ops.tensor_product.mosaic_tpu.fwd import (
    _tensor_product_kernel_mosaic_tpu_fwd,
)
from e3j.pallas_ops.tensor_product.mosaic_tpu.params import (
    PallasMosaicTPUTensorProductParams,
)


def _active_mesh_axes() -> tuple[object, tuple[str, ...]]:
    """Return the surrounding context mesh and its axis names, or `()` if unset."""
    mesh = jax.sharding.get_abstract_mesh()
    return mesh, tuple(getattr(mesh, "axis_names", ()) or ())


def _num_shards(mesh, axes: tuple[str, ...]) -> int:
    n = 1
    for name in axes:
        n *= mesh.shape[name]
    return n


def _batch_partition_spec(ndim: int, axes: tuple[str, ...]) -> PartitionSpec:
    """Shard the leading batch axis across the mesh."""
    spec = [None] * ndim
    spec[0] = axes if len(axes) > 1 else axes[0]
    return PartitionSpec(*spec)


def _run_kernel_data_parallel(impl, arrays, out_ndims):
    """Run the merged-batch kernel `impl(*arrays)` data-parallel.

    Shards the leading (merged batch) axis of every input across the mesh; the
    TP is independent per batch element, so any even split is correct, but
    shard_map still requires the axis to be divisible by the device count."""
    mesh, axes = _active_mesh_axes()
    if not axes:
        return impl(*arrays)
    d = _num_shards(mesh, axes)
    n = arrays[0].shape[0]
    assert n % d == 0, f"data-parallel batch of {n} not divisible by {d} devices"
    in_specs = tuple(_batch_partition_spec(a.ndim, axes) for a in arrays)
    out_specs = tuple(_batch_partition_spec(nd, axes) for nd in out_ndims)
    if len(out_specs) == 1:
        out_specs = out_specs[0]
    return jax.shard_map(
        impl, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )(*arrays)


def _merge_leading_into_batch(a: jax.Array) -> jax.Array:
    """Merge the leading vmap axis `V` into the leading batch axis:
    `(V, batch, *shape)` -> `(V * batch, *shape)`."""
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


def _split_batch_to_leading(a: jax.Array, v: int) -> jax.Array:
    """Inverse of :func:`_merge_leading_into_batch`."""
    return a.reshape((v, a.shape[0] // v) + a.shape[1:])


def _fwd_vmappable(
    x: jax.Array, y: jax.Array, params: PallasMosaicTPUTensorProductParams
) -> jax.Array:
    """Forward kernel call wrapped for vmap."""

    @jax.custom_batching.custom_vmap
    def _impl(x, y):
        return _tensor_product_kernel_mosaic_tpu_fwd(x, y, params)

    @_impl.def_vmap
    def _rule(axis_size, in_batched, x, y):
        x_b, y_b = in_batched
        if not x_b:
            x = jnp.broadcast_to(x[None], (axis_size,) + x.shape)
        if not y_b:
            y = jnp.broadcast_to(y[None], (axis_size,) + y.shape)
        mx, my = _merge_leading_into_batch(x), _merge_leading_into_batch(y)
        out = _run_kernel_data_parallel(_impl, (mx, my), (mx.ndim,))
        return _split_batch_to_leading(out, axis_size), True

    return _impl(x, y)


def _bwd_vmappable(
    x: jax.Array,
    y: jax.Array,
    ct_z: jax.Array,
    params: PallasMosaicTPUTensorProductParams,
) -> tuple[jax.Array, jax.Array]:
    """Backward kernel call wrapped for vmap (see `_fwd_vmappable`)."""

    @jax.custom_batching.custom_vmap
    def _impl(x, y, ct_z):
        return _tensor_product_kernel_mosaic_tpu_bwd(x, y, ct_z, params)

    @_impl.def_vmap
    def _rule(axis_size, in_batched, x, y, ct_z):
        x_b, y_b, ct_b = in_batched
        if not x_b:
            x = jnp.broadcast_to(x[None], (axis_size,) + x.shape)
        if not y_b:
            y = jnp.broadcast_to(y[None], (axis_size,) + y.shape)
        if not ct_b:
            ct_z = jnp.broadcast_to(ct_z[None], (axis_size,) + ct_z.shape)
        mx = _merge_leading_into_batch(x)
        my = _merge_leading_into_batch(y)
        mct = _merge_leading_into_batch(ct_z)
        dx, dy = _run_kernel_data_parallel(_impl, (mx, my, mct), (mx.ndim, my.ndim))
        return (
            _split_batch_to_leading(dx, axis_size),
            _split_batch_to_leading(dy, axis_size),
        ), (True, True)

    return _impl(x, y, ct_z)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _tp_bwd(
    x: jax.Array,
    y: jax.Array,
    ct_z: jax.Array,
    params: PallasMosaicTPUTensorProductParams,
) -> tuple[jax.Array, jax.Array]:
    return _bwd_vmappable(x, y, ct_z, params)


def _tp_bwd_fwd(
    x: jax.Array,
    y: jax.Array,
    ct_z: jax.Array,
    params: PallasMosaicTPUTensorProductParams,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    dx, dy = _bwd_vmappable(x, y, ct_z, params)
    return (dx, dy), (x, y, ct_z)


def _tp_bwd_bwd(
    params: PallasMosaicTPUTensorProductParams,
    res: tuple[jax.Array, jax.Array, jax.Array],
    ct_dxdy: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Cotangents on the backward's OUTPUTS (dx, dy) are 2nd-order variations.
    ddx, ddy = ct_dxdy
    x, y, ct_z = res
    # Cotangents on the backward's INPUTS (x, y, ct_z): 1 backward + 2 forwards.
    dx, dy = _tp_bwd(ddx, ddy, ct_z, params)
    dct_z = tensor_product_pallas_mosaic_tpu(
        x, ddy, params
    ) + tensor_product_pallas_mosaic_tpu(ddx, y, params)
    return (dx, dy, dct_z)


_tp_bwd.defvjp(_tp_bwd_fwd, _tp_bwd_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def tensor_product_pallas_mosaic_tpu(
    x: jax.Array,
    y: jax.Array,
    params: PallasMosaicTPUTensorProductParams,
) -> jax.Array:
    return _fwd_vmappable(x, y, params)


def _tensor_product_pallas_mosaic_tpu_fwd(
    x: jax.Array,
    y: jax.Array,
    params: PallasMosaicTPUTensorProductParams,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    z = tensor_product_pallas_mosaic_tpu(x, y, params)
    return z, (x, y)


def _tensor_product_pallas_mosaic_tpu_bwd(
    params: PallasMosaicTPUTensorProductParams,
    res: tuple[jax.Array, jax.Array],
    ct_z: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    x, y = res
    return _tp_bwd(x, y, ct_z, params)


tensor_product_pallas_mosaic_tpu.defvjp(
    _tensor_product_pallas_mosaic_tpu_fwd, _tensor_product_pallas_mosaic_tpu_bwd
)
