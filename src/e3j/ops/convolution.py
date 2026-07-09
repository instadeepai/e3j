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

import warnings
from dataclasses import dataclass

import jax
import jax.experimental.custom_partitioning
import jax.numpy as jnp
from jax import Array, custom_vjp
from jax.ffi import ffi_call
from numpy import int32

from e3j.data.graph import GraphCSR
from e3j.ops.coef import Coef4D
from e3j.utils import config, is_pow2


@dataclass
class CUDAConvolutionParams:
    num_out: int
    num_scalars: int


class ConvolutionBatchingWarning(UserWarning):
    """Emitted when convolution is vmapped over the graph adjacency.

    Silence with `warnings.filterwarnings("ignore", category=...)`, or turn
    into an error with `-W error` / `warnings.simplefilter("error", ...)`.
    """


def _warn_graph_batching():
    """Emit a single `ConvolutionBatchingWarning` for graph-vmap."""
    warnings.warn(
        "Convolution is being vmapped over graph adjacency. Batch graphs into a "
        "single, disconnected graph upstream or silence this warning by filtering "
        "`ConvolutionBatchingWarning`.",
        ConvolutionBatchingWarning,
        stacklevel=2,
    )


def convolution(
    coef: Array,
    x: Array,
    y: Array,
    s: Array,
    sender: Array,
    receiver: Array,
    params: CUDAConvolutionParams,
) -> Array:
    """Primitive bound to the CUDA convolution kernel.

    The equivariant convolution computes the following operation:

        mⱼ= ∑ᵢ (xᵢ⊗ yᵢⱼ) ⊙  sᵢⱼ

    Where the sum runs over neighbor nodes i of the receiver node j.
    In other words, the convolution kernel fuses the following operations:
    * gather sender node features `x`,
    * compute the tensor product with edge (spherical) embeddings `y`,
    * mix with edge scalars `s`,
    * scatter-reduce by receiver index.

    The edge messages are computed as a single trilinear mixing of x, y, z.

    Note:
        Edges and edge features should be sorted by receiver index, in order
        to compute the CSR adjacency matrix correctly. This ordering should be
        done at graph creation time to avoid having to reorder features at
        every convolution call.

    Args:
        coef: Packed Coef4D coefficients (opaque idx_dtype vector).
        x: Node features, shape (num_nodes, num_x, channels_x).
        y: Edge embeddings, shape (num_edges, num_y).
        s: Radial scalars, shape (num_edges, num_scalars, channels_x).
        sender: CSR sender indices, shape (num_edges,).
        receiver: Receiver indices, shape (num_edges,).
        params: Convolution parameters.

    Returns:
        Output node features, shape (num_nodes, num_out, channels_x).
    """
    num_nodes = x.shape[0]
    num_edges = sender.shape[0]

    channels_x = x.shape[-1] if x.ndim > 2 else 1
    num_out = params.num_out
    has_cx = x.ndim > 2

    if y.ndim == x.ndim and y.shape[-1] != 1:
        raise NotImplementedError("RHS y should have only one channel.")
    if not is_pow2(channels_x):
        raise NotImplementedError("LHS x should have power of 2 number of channels.")

    # ---- custom_partitioning ----
    #
    # All array arguments explicit, only close over static problem sizes.
    @jax.experimental.custom_partitioning.custom_partitioning
    def _sharded_op(coef, x, y, s, sender, receiver):
        n = x.shape[0]
        sender_local = sender % n
        receiver_local = receiver % n
        receiver_local_ptr = GraphCSR(n, sender_local, receiver_local).receiver_ptr
        shape_out = (n, num_out, channels_x)
        return ffi_call(
            "convolution",
            jax.ShapeDtypeStruct(shape_out, x.dtype),
        )(
            coef,
            x,
            y,
            s,
            sender_local,
            receiver_local_ptr,
            num_nodes=int32(n),
            debug=int32(config().debug_level),
        )

    def _partition(mesh, arg_shapes, result_shape):
        return (
            mesh,
            _sharded_op,
            result_shape.sharding,
            tuple(a.sharding for a in arg_shapes),
        )

    x_rule = "nx cx" if has_cx else "nx"
    s_rule = "ns cx"
    out_rule = "nz cx"
    sharding_rule = (
        f"u v, nodes {x_rule}, edges ny, edges {s_rule}, edges, edges"
        f" -> nodes {out_rule}"
    )
    _sharded_op.def_partition(
        partition=_partition,
        sharding_rule=sharding_rule,
        need_replication_factors=("u", "v"),
    )

    # ---- custom_vmap ----
    #
    # Graph adjacency needs to be passed explicitly to vmap over graphs.
    # We warn in this case since apart from SPMD, this in an anti-pattern.
    # Only coefficients are closed over.

    @jax.custom_batching.custom_vmap
    def _batched_op(x, y, s, sender, receiver):
        return _sharded_op(coef, x, y, s, sender, receiver)

    @_batched_op.def_vmap
    def _vmap_rule(axis_size, in_batched, x, y, s, sender, receiver):
        x_b, y_b, s_b, sender_b, receiver_b = in_batched
        graph_b = sender_b or receiver_b
        if not x_b:
            x = jnp.broadcast_to(x[None], (axis_size,) + x.shape)
        if not y_b:
            y = jnp.broadcast_to(y[None], (axis_size,) + y.shape)
        if not s_b:
            s = jnp.broadcast_to(s[None], (axis_size,) + s.shape)

        if graph_b:
            if not (sender_b and receiver_b):
                raise ValueError("Graph adjacency inconsistently batched for vmap.")
            _warn_graph_batching()

        # batched=False tiles the shared graph; batched=True concatenates.
        node_offsets = jnp.arange(axis_size, dtype=sender.dtype)[:, None] * num_nodes
        sender = (sender + node_offsets).reshape(-1)
        receiver = (receiver + node_offsets).reshape(-1)

        x = x.reshape((axis_size * num_nodes,) + x.shape[2:])
        y = y.reshape((axis_size * num_edges,) + y.shape[2:])
        s = s.reshape((axis_size * num_edges,) + s.shape[2:])
        out = _sharded_op(coef, x, y, s, sender, receiver)
        return out.reshape((axis_size, num_nodes) + out.shape[1:]), True

    # ---- custom_vjp ----
    #
    # Differentiable arguments only: (x, y, s).

    @custom_vjp
    def convolution_op(x, y, s):
        return _batched_op(x, y, s, sender, receiver)

    def _fwd(x, y, s):
        z = convolution_op(x, y, s)
        # Store graph adjacency in residuals to avoid tracer leaks in _bwd.
        return z, (x, y, s, sender, receiver)

    def _bwd(res, dm):
        x, y, s, sender, receiver = res
        return convolution_bwd(coef, x, y, s, sender, receiver, dm, params)

    convolution_op.defvjp(_fwd, _bwd)

    return convolution_op(x, y, s)


def convolution_bwd(coef, x, y, s, sender, receiver, dm, params):
    """Primitive bound to the CUDA convolution backward kernel.

    Computes cotangents `dx`, `dy`, `ds` from output cotangent `dm`
    by calling the fused backward kernel with transposed coefficients
    and transposed CSR adjacency.
    """
    num_nodes = x.shape[0]
    num_edges = sender.shape[0]
    has_cx = x.ndim > 2

    with jax.ensure_compile_time_eval():
        # Transpose coefficients for backward kernel, passed as a triple
        # (coef_dx, coef_dy, coef_ds) such that:
        #   - dx = bigotimes(coef_dx, dm, y, s)
        #   - dy = bigotimes(coef_dy, dm, x, s)
        #   - ds = bigotimes(coef_ds, dm, y, x)
        c = Coef4D.unpack(coef, val_dtype="float32")
        coef_dx = c.transpose((1, 0, 2, 3)).pack_jax()
        coef_dy = c.transpose((2, 0, 1, 3)).pack_jax()
        coef_ds = c.transpose((3, 0, 2, 1)).pack_jax()
        coef_bwd = jnp.stack([coef_dx, coef_dy, coef_ds])

    # ---- custom_partitioning ----

    @jax.experimental.custom_partitioning.custom_partitioning
    def _sharded_op(coef_bwd, x, y, s, dm, sender, receiver):
        num_nodes = x.shape[0]
        sender_local = sender % num_nodes
        receiver_local = receiver % num_nodes
        perm, graph_local_t, _ = GraphCSR(
            num_nodes, sender_local, receiver_local
        ).transpose()
        sender_local_t = graph_local_t.sender
        receiver_local_t_ptr = graph_local_t.receiver_ptr

        return ffi_call(
            "convolution_bwd",
            (
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(y.shape, y.dtype),
                jax.ShapeDtypeStruct(s.shape, s.dtype),
            ),
        )(
            coef_bwd,
            x,
            y,
            s,
            dm,
            sender_local_t,
            receiver_local_t_ptr,
            perm,
            num_nodes=int32(num_nodes),
            debug=int32(config().debug_level),
        )

    def _partition(mesh, arg_shapes, result_shape):
        ct_x_shape, ct_y_shape, ct_s_shape = result_shape
        return (
            mesh,
            _sharded_op,
            (ct_x_shape.sharding, ct_y_shape.sharding, ct_s_shape.sharding),
            tuple(a.sharding for a in arg_shapes),
        )

    x_rule = "nx cx" if has_cx else "nx"
    s_rule = "ns cx"
    dm_rule = "nz cx"
    sharding_rule = (
        f"o p q, nodes {x_rule}, edges ny, edges {s_rule}, nodes {dm_rule},"
        f" edges, edges -> nodes {x_rule}, edges ny, edges {s_rule}"
    )
    _sharded_op.def_partition(
        partition=_partition,
        sharding_rule=sharding_rule,
        need_replication_factors=("o", "p", "q"),
    )

    # ---- custom_vmap ----

    @jax.custom_batching.custom_vmap
    def _batched_op(x, y, s, dm, sender, receiver):
        return _sharded_op(coef_bwd, x, y, s, dm, sender, receiver)

    @_batched_op.def_vmap
    def _vmap_rule(axis_size, in_batched, x, y, s, dm, sender, receiver):
        x_b, y_b, s_b, dm_b, sender_b, receiver_b = in_batched
        graph_b = sender_b or receiver_b
        if not x_b:
            x = jnp.broadcast_to(x[None], (axis_size,) + x.shape)
        if not y_b:
            y = jnp.broadcast_to(y[None], (axis_size,) + y.shape)
        if not s_b:
            s = jnp.broadcast_to(s[None], (axis_size,) + s.shape)
        if not dm_b:
            dm = jnp.broadcast_to(dm[None], (axis_size,) + dm.shape)
        if graph_b:
            if not (sender_b and receiver_b):
                raise ValueError("Graph adjacency inconsistently batched for vmap.")
            _warn_graph_batching()

        # batched=False tiles the shared graph; batched=True concatenates.
        node_offsets = jnp.arange(axis_size, dtype=sender.dtype)[:, None] * num_nodes
        sender = (sender + node_offsets).reshape(-1)
        receiver = (receiver + node_offsets).reshape(-1)

        x = x.reshape((axis_size * num_nodes,) + x.shape[2:])
        y = y.reshape((axis_size * num_edges,) + y.shape[2:])
        s = s.reshape((axis_size * num_edges,) + s.shape[2:])
        dm = dm.reshape((axis_size * num_nodes,) + dm.shape[2:])

        dx, dy, ds = _sharded_op(coef_bwd, x, y, s, dm, sender, receiver)

        dx = dx.reshape((axis_size, num_nodes) + dx.shape[1:])
        dy = dy.reshape((axis_size, num_edges) + dy.shape[1:])
        ds = ds.reshape((axis_size, num_edges) + ds.shape[1:])
        return (dx, dy, ds), (True, True, True)

    # ---- custom_vjp ----

    @custom_vjp
    def convolution_bwd_op(x, y, s, dm):
        return _batched_op(x, y, s, dm, sender, receiver)

    def _fwd(x, y, s, dm):
        dx, dy, ds = convolution_bwd_op(x, y, s, dm)
        # Store the graph adjacency in residuals to avoid tracer leaks.
        return (dx, dy, ds), (x, y, s, dm, sender, receiver)

    def _bwd(res, cotangents):
        """Return (Dx, Dy, Ds, Ddm) cotangents from (Ddx, Ddy, Dds)."""
        (Ddx, Ddy, Dds) = cotangents
        (x, y, s, dm, sender, receiver) = res

        def conv(x, y, s):
            return convolution(coef, x, y, s, sender, receiver, params)

        def conv_bwd(x, y, s):
            return convolution_bwd(coef, x, y, s, sender, receiver, dm, params)

        # Second variation of messages: three forward passes
        Ddm = conv(Ddx, y, s) + conv(x, Ddy, s) + conv(x, y, Dds)

        # Primal cotangents
        Dx_x, Dx_y, Dx_s = conv_bwd(Ddx, y, s)
        Dy_x, Dy_y, Dy_s = conv_bwd(x, Ddy, s)
        Ds_x, Ds_y, Ds_s = conv_bwd(x, y, Dds)

        Dx = Dy_x + Ds_x
        Dy = Dx_y + Ds_y
        Ds = Dx_s + Dy_s

        return (Dx, Dy, Ds, Ddm)

    convolution_bwd_op.defvjp(_fwd, _bwd)

    return convolution_bwd_op(x, y, s, dm)


convolution.Params = CUDAConvolutionParams
