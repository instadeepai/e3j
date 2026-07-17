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

"""Differentiable message-passing convolution op for Pallas Mosaic TPU.

Wraps the raw forward/backward kernels (`fwd.py` / `bwd.py`) into the public
`convolution_mosaic_tpu` op and registers its JAX transformation
rules: custom VJP (incl. double-backward for force training), custom vmap, and
shard_map-based data parallelism over the device mesh.
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from e3j.pallas_ops.convolution.mosaic_tpu.bwd import (
    _message_passing_kernel_mosaic_tpu_bwd as _bwd_impl,
)
from e3j.pallas_ops.convolution.mosaic_tpu.fwd import (
    _message_passing_kernel_mosaic_tpu_fwd as _fwd_impl,
)
from e3j.pallas_ops.convolution.mosaic_tpu.params import (
    PallasMosaicTPUMessagePassingConvolutionParams,
)

# --------------------------------------------------------------------------- #
# shard_map data-parallel helpers                                             #
# --------------------------------------------------------------------------- #


def _active_mesh_axes() -> tuple[object, tuple[str, ...]]:
    mesh = jax.sharding.get_abstract_mesh()
    return mesh, tuple(getattr(mesh, "axis_names", ()) or ())


def _batch_partition_spec(ndim: int, axes: tuple[str, ...]) -> PartitionSpec:
    # TRAILING_CHANNELS puts the graph batch (nodes/edges) on the leading axis.
    return PartitionSpec(axes if len(axes) > 1 else axes[0], *([None] * (ndim - 1)))


def _num_shards(mesh, axes: tuple[str, ...]) -> int:
    n = 1
    for name in axes:
        n *= mesh.shape[name]
    return n


def _assert_graphs_divide_shards(axis_size: int) -> None:
    mesh, axes = _active_mesh_axes()
    d = _num_shards(mesh, axes)
    assert (
        axis_size % d == 0
    ), f"data-parallel batch of {axis_size} graphs not divisible by {d} devices"


def _shard_index(mesh, axes: tuple[str, ...]) -> jax.Array:
    """Row-major linear index of this shard along the sharded (graph batch) axis,
    matching how `_batch_partition_spec` flattens `axes` onto the leading dim."""
    idx = 0
    for name in axes:
        idx = idx * mesh.shape[name] + jax.lax.axis_index(name)
    return idx


def _run_kernel_data_parallel(
    impl, arrays, out_ndims, edge_argnums: tuple[int, ...] = ()
):
    """Run `impl(*arrays)` data-parallel via shard_map over the device mesh.

    Shards the leading (graph batch) axis of every input; `out_ndims` gives the
    rank of each output so its leading axis is sharded the same way. Edge-index
    args (`edge_argnums`) carry global offsets into the merged node block, but a
    shard only holds its slice of the nodes (re-indexed from 0), so they are
    rebased by `shard_index * local_n_nodes` (arrays[0] is the node feats).
    """
    # Explicit `shard_map` (manual SPMD) lets us write the per-shard program directly:
    # each device runs the kernel on its own mini-batch of graphs.
    # `@custom_partitioning` instead declares a `sharding_rule` and lets XLA/GSPMD
    # infer the partition, but is unsupported on TPU
    # (https://github.com/jax-ml/jax/issues/38196).
    mesh, axes = _active_mesh_axes()
    if not axes:
        return impl(*arrays)
    in_specs = tuple(_batch_partition_spec(a.ndim, axes) for a in arrays)
    out_specs = tuple(_batch_partition_spec(nd, axes) for nd in out_ndims)
    if len(out_specs) == 1:
        out_specs = out_specs[0]

    def _sharded_op(*shard_arrays):
        if edge_argnums:
            base = _shard_index(mesh, axes) * shard_arrays[0].shape[0]
            shard_arrays = list(shard_arrays)
            for i in edge_argnums:
                shard_arrays[i] = shard_arrays[i] - base
        return impl(*shard_arrays)

    return jax.shard_map(
        _sharded_op, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )(*arrays)


# --------------------------------------------------------------------------- #
# vmap (data-parallel batching) helpers                                       #
# --------------------------------------------------------------------------- #


def _merge_leading_into_batch(a: jax.Array) -> jax.Array:
    """`(V, B, ...)` -> `(V*B, ...)`: fold the leading vmap axis into the batch."""
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


def _split_batch_to_leading(a: jax.Array, v: int) -> jax.Array:
    """Inverse of _merge_leading_into_batch: split `(V*B, ...)` -> `(V, B, ...)`."""
    return a.reshape((v, a.shape[0] // v) + a.shape[1:])


def _shift_edges(senders, receivers, n_nodes: int):
    """Offset graph `v`'s edge indices into the merged node block `[v*N, (v+1)*N)`."""
    v = senders.shape[0]
    offsets = (jnp.arange(v, dtype=senders.dtype) * n_nodes)[:, None]
    return senders + offsets, receivers + offsets


def _fwd_vmappable(x, y, s, senders, receivers, params):
    """Forward kernel call, vmappable for data-parallel training."""

    @jax.custom_batching.custom_vmap
    def _impl(x, y, s, senders, receivers):
        return _fwd_impl(x, y, s, senders, receivers, params)

    @_impl.def_vmap
    def _rule(axis_size, in_batched, x, y, s, senders, receivers):
        _assert_graphs_divide_shards(axis_size)
        args = [x, y, s, senders, receivers]
        for i, b in enumerate(in_batched):
            if not b:
                args[i] = jnp.broadcast_to(args[i][None], (axis_size,) + args[i].shape)
        args[3], args[4] = _shift_edges(args[3], args[4], args[0].shape[1])
        merged = tuple(_merge_leading_into_batch(a) for a in args)
        # z: (V*N, out_dim, channels)
        z = _run_kernel_data_parallel(_impl, merged, (3,), edge_argnums=(3, 4))
        return _split_batch_to_leading(z, axis_size), True

    return _impl(x, y, s, senders, receivers)


def _bwd_vmappable(x, y, s, senders, receivers, dm, params):
    """Backward kernel call, vmappable for data-parallel training."""

    @jax.custom_batching.custom_vmap
    def _impl(x, y, s, senders, receivers, dm):
        return _bwd_impl(x, y, s, senders, receivers, dm, params)

    @_impl.def_vmap
    def _rule(axis_size, in_batched, x, y, s, senders, receivers, dm):
        _assert_graphs_divide_shards(axis_size)
        args = [x, y, s, senders, receivers, dm]
        for i, b in enumerate(in_batched):
            if not b:
                args[i] = jnp.broadcast_to(args[i][None], (axis_size,) + args[i].shape)
        args[3], args[4] = _shift_edges(args[3], args[4], args[0].shape[1])
        merged = tuple(_merge_leading_into_batch(a) for a in args)
        dx, dy, ds = _run_kernel_data_parallel(
            _impl, merged, (3, 2, 3), edge_argnums=(3, 4)
        )
        return (
            _split_batch_to_leading(dx, axis_size),
            _split_batch_to_leading(dy, axis_size),
            _split_batch_to_leading(ds, axis_size),
        ), (True, True, True)

    return _impl(x, y, s, senders, receivers, dm)


# --------------------------------------------------------------------------- #
# Forward convolution op and its autodiff rules                               #
# --------------------------------------------------------------------------- #


def convolution_mosaic_tpu(
    x: jax.Array,
    y: jax.Array,
    s: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    params: PallasMosaicTPUMessagePassingConvolutionParams,
) -> jax.Array:
    @jax.custom_vjp
    def _fwd_differentiable(x, y, s):
        return _fwd_vmappable(x, y, s, receivers, senders, params.swapped())

    def _fwd(x, y, s):
        z = _fwd_differentiable(x, y, s)
        return z, (x, y, s, senders, receivers)

    def _bwd(residuals, dm):
        x, y, s, senders, receivers = residuals
        return _convolution_bwd(x, y, s, senders, receivers, params, dm)

    _fwd_differentiable.defvjp(_fwd, _bwd)
    return _fwd_differentiable(x, y, s)


# --------------------------------------------------------------------------- #
# Differentiable backward and its autodiff rules (enables double-backward)    #
# --------------------------------------------------------------------------- #


def _convolution_bwd(x, y, s, senders, receivers, params, dm):
    @jax.custom_vjp
    def _bwd_differentiable(x, y, s, dm):
        return _bwd_vmappable(x, y, s, senders, receivers, dm, params)

    def _fwd(x, y, s, dm):
        out = _bwd_differentiable(x, y, s, dm)
        return out, (x, y, s, dm, senders, receivers)

    def _bwd(residuals, cts):
        x, y, s, dm, senders, receivers = residuals
        Ddx, Ddy, Dds = cts  # cotangents on (dx, dy, ds)
        A = _bwd_vmappable(Ddx, y, s, senders, receivers, dm, params)  # A[1], A[2]
        B = _bwd_vmappable(x, Ddy, s, senders, receivers, dm, params)  # B[0], B[2]
        C = _bwd_vmappable(x, y, Dds, senders, receivers, dm, params)  # C[0], C[1]
        Dx = B[0] + C[0]
        Dy = A[1] + C[1]
        Ds = A[2] + B[2]
        Ddz = (
            convolution_mosaic_tpu(Ddx, y, s, senders, receivers, params)
            + convolution_mosaic_tpu(x, Ddy, s, senders, receivers, params)
            + convolution_mosaic_tpu(x, y, Dds, senders, receivers, params)
        )
        return Dx, Dy, Ds, Ddz

    _bwd_differentiable.defvjp(_fwd, _bwd)
    return _bwd_differentiable(x, y, s, dm)
