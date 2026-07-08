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

from e3j.ops.coef import Coef4D
from e3j.utils import config, is_pow2


@dataclass
class ConvolutionParams:
    num_out: int
    num_scalars: int


@dataclass
class GraphCSR:
    """Compressed Sparse Row (CSR) directed-graph adjacency.

    Edges are grouped by receiver so the convolution kernel can reduce
    messages per receiver without atomics.

    Beyond construction and transposition, this class owns index arithmetic
    required by `vmap` rules to fold a batch axis into a single disjoint
    (block-diagonal) graph, both when a single graph is replicated across
    devices or when distinct homogeneous graphs are stacked.
    """

    def __init__(self, num_nodes: int, sender: Array, receiver: Array):
        self.num_nodes = num_nodes
        self.sender = sender
        self.receiver = receiver
        num_neighbors = jnp.bincount(receiver, length=num_nodes)
        self.receiver_ptr = jnp.append(0, jnp.cumsum(num_neighbors))

    @classmethod
    def sort(
        cls, num_nodes: int, sender: Array, receiver: Array
    ) -> tuple[Array, "GraphCSR", Array]:
        """Return (sigma, graph', sigma_1) sorting edges by receivers."""
        perm = jnp.argsort(receiver)
        graph_sorted = cls(num_nodes, sender[perm], receiver[perm])
        return perm, graph_sorted, jnp.argsort(perm)

    def transpose(self) -> tuple[Array, "GraphCSR", Array]:
        """Return (sigma, graph_t, sigma_1) sorting edges by senders instead."""
        return GraphCSR.sort(
            self.num_nodes,
            self.receiver,
            self.sender,
        )

    @staticmethod
    def fold_adjacency(axis_size, num_nodes, num_edges, sender, receiver_ptr, batched):
        """Fold a batch axis of `(sender, receiver_ptr)` into one disjoint graph.

        Each batch element becomes an independent connected component: senders
        are shifted by `b * num_nodes` and receiver pointers by `b * num_edges`,
        yielding a graph with `axis_size * num_nodes` nodes and
        `axis_size * num_edges` edges.

        `batched` selects the source layout. When `True`, the graph varies per
        element (`sender: (axis_size, num_edges)`) and elements are
        concatenated. When `False`, a single shared graph (`sender:
        (num_edges,)`) is broadcast first, which reduces to tiling it. Both
        cases share the same offset-and-flatten once broadcast.
        """
        if not batched:
            sender = jnp.broadcast_to(sender, (axis_size,) + sender.shape)
            receiver_ptr = jnp.broadcast_to(
                receiver_ptr, (axis_size,) + receiver_ptr.shape
            )
        node_offsets = jnp.arange(axis_size, dtype=sender.dtype)[:, None] * num_nodes
        edge_offsets = (
            jnp.arange(axis_size, dtype=receiver_ptr.dtype)[:, None] * num_edges
        )
        sender_folded = (sender + node_offsets).reshape(-1)
        ptr_body = (receiver_ptr[:, :num_nodes] + edge_offsets).reshape(-1)
        receiver_ptr_folded = jnp.append(ptr_body, axis_size * num_edges)
        return sender_folded, receiver_ptr_folded

    @staticmethod
    def fold_permutation(axis_size, num_edges, perm, batched):
        """Fold a batch axis of an edge permutation into one disjoint graph.

        Offsets each element by `b * num_edges` so it indexes the correct
        segment of the concatenated edge array. `batched` selects per-element
        (`(axis_size, num_edges)`) vs shared (`(num_edges,)`, broadcast) layout.
        """
        if not batched:
            perm = jnp.broadcast_to(perm, (axis_size,) + perm.shape)
        edge_offsets = jnp.arange(axis_size, dtype=perm.dtype)[:, None] * num_edges
        return (perm + edge_offsets).reshape(-1)


class ConvolutionBatchingWarning(UserWarning):
    """Emitted when convolution is vmapped over the graph adjacency.

    Silence with `warnings.filterwarnings("ignore", category=...)`, or turn
    into an error with `-W error` / `warnings.simplefilter("error", ...)`.
    """


_GRAPH_VMAP_WARNING = (
    "convolution is being vmapped over the graph adjacency (`sender`/"
    "`receiver`). Each batch element's graph is folded into a single disjoint "
    "graph by sparse concatenation -- numerically equivalent to `Graph.batch()`"
    " but paying for padding across the batch. This is an anti-pattern for "
    "production training: prefer building one pre-batched graph with "
    "`Graph.batch()`. Only a leading axis over node/edge *features* with a "
    "shared (replicated) graph is free of this cost. Silence by filtering "
    "`ConvolutionBatchingWarning`."
)


_GRAPH_BATCH_INCONSISTENT = (
    "inconsistent graph batching under vmap: the graph adjacency arrays "
    "(sender, receiver_ptr, and the edge permutation in the backward pass) "
    "must share the same batch axis, but only some of them were batched."
)


def _warn_graph_batching():
    """Emit a single `ConvolutionBatchingWarning` for graph-vmap.

    Routed through one call site so `warnings` deduplication collapses the
    forward and backward rules to a single warning per trace under
    `grad(vmap(...))`.
    """
    warnings.warn(_GRAPH_VMAP_WARNING, ConvolutionBatchingWarning, stacklevel=2)


def convolution(
    coef: Array,
    x: Array,
    y: Array,
    s: Array,
    sender: Array,
    receiver: Array,
    params: ConvolutionParams,
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
    receiver_ptr = GraphCSR(num_nodes, sender, receiver).receiver_ptr

    channels_x = x.shape[-1] if x.ndim > 2 else 1
    num_out = params.num_out
    has_cx = x.ndim > 2

    if y.ndim == x.ndim and y.shape[-1] != 1:
        raise NotImplementedError("RHS y should have only one channel.")
    if not is_pow2(channels_x):
        raise NotImplementedError("LHS x should have power of 2 number of channels.")

    # custom_partitioning: all array arguments explicit,
    # closes over compile-time shape constants (num_out, channels_x).

    @jax.experimental.custom_partitioning.custom_partitioning
    def _sharded_op(coef, x, y, s, sender, receiver_ptr):
        n = x.shape[0]
        shape_out = (n, num_out, channels_x)
        return ffi_call(
            "convolution",
            jax.ShapeDtypeStruct(shape_out, x.dtype),
        )(
            coef,
            x,
            y,
            s,
            sender,
            receiver_ptr,
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

    # Explicit row dimensions instead of ellipsis, because node-indexed
    # and edge-indexed arrays have different row counts.
    x_rule = "nx cx" if has_cx else "nx"
    s_rule = "ns cx"
    out_rule = "nz cx"
    sharding_rule = (
        f"u v, nodes {x_rule}, edges ny, edges {s_rule}, p, r" f" -> nodes {out_rule}"
    )
    _sharded_op.def_partition(
        partition=_partition,
        sharding_rule=sharding_rule,
        need_replication_factors=("u", "v", "p", "r"),
    )

    # custom_vmap: batchable feature arguments (x, y, s) plus the graph
    # adjacency (sender, receiver_ptr). The graph is passed explicitly rather
    # than closed over so the rule sees its batch status: a batched graph is
    # folded into one disjoint graph with a warning (see `_warn_graph_batching`),
    # while an unbatched (replicated) graph takes the feature-batching path.
    # Only `coef` stays closed over.

    @jax.custom_batching.custom_vmap
    def _batched_op(x, y, s, sender, receiver_ptr):
        return _sharded_op(coef, x, y, s, sender, receiver_ptr)

    @_batched_op.def_vmap
    def _vmap_rule(axis_size, in_batched, x, y, s, sender, receiver_ptr):
        x_b, y_b, s_b, sender_b, receiver_ptr_b = in_batched
        graph_b = sender_b or receiver_ptr_b
        if not x_b:
            x = jnp.broadcast_to(x[None], (axis_size,) + x.shape)
        if not y_b:
            y = jnp.broadcast_to(y[None], (axis_size,) + y.shape)
        if not s_b:
            s = jnp.broadcast_to(s[None], (axis_size,) + s.shape)
        if graph_b:
            # Batched graph: fold per-element graphs into one disjoint graph.
            if not (sender_b and receiver_ptr_b):
                raise ValueError(_GRAPH_BATCH_INCONSISTENT)
            _warn_graph_batching()
        # `batched=False` tiles the shared (replicated) graph; both share the
        # same block-diagonal fold.
        sender_t, receiver_ptr_t = GraphCSR.fold_adjacency(
            axis_size, num_nodes, num_edges, sender, receiver_ptr, batched=graph_b
        )
        x = x.reshape((axis_size * num_nodes,) + x.shape[2:])
        y = y.reshape((axis_size * num_edges,) + y.shape[2:])
        s = s.reshape((axis_size * num_edges,) + s.shape[2:])
        out = _sharded_op(coef, x, y, s, sender_t, receiver_ptr_t)
        return out.reshape((axis_size, num_nodes) + out.shape[1:]), True

    # custom_vjp: differentiable arguments only (x, y, s).

    @custom_vjp
    def convolution_op(x, y, s):
        return _batched_op(x, y, s, sender, receiver_ptr)

    def _fwd(x, y, s):
        z = convolution_op(x, y, s)
        # Carry the graph adjacency through the residuals rather than closing
        # over it: under a batched graph (vmap over sender/receiver) the closed-
        # over tracers would escape the forward trace and leak into the
        # separately-traced backward. Residuals are batched by JAX correctly.
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

    # Transpose CSR adjacency: group by sender instead of receiver.
    graph = GraphCSR(num_nodes, sender, receiver)
    perm, graph_t, _ = graph.transpose()
    sender_t = graph_t.sender
    receiver_ptr_t = graph_t.receiver_ptr

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

    # custom_partitioning: all array arguments explicit.

    @jax.experimental.custom_partitioning.custom_partitioning
    def _sharded_op(coef_bwd, x, y, s, dm, sender_t, receiver_ptr_t, perm):
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
            sender_t,
            receiver_ptr_t,
            perm,
            num_nodes=int32(x.shape[0]),
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
        f" t, u, v -> nodes {x_rule}, edges ny, edges {s_rule}"
    )
    _sharded_op.def_partition(
        partition=_partition,
        sharding_rule=sharding_rule,
        need_replication_factors=("o", "p", "q", "t", "u", "v"),
    )

    # custom_vmap: batchable feature/cotangent arguments (x, y, s, dm) plus the
    # transposed graph adjacency (sender_t, receiver_ptr_t, perm). As in the
    # forward pass, the graph is passed explicitly so a batched graph is
    # rejected rather than silently mis-tiled; only `coef_bwd` stays closed over.

    @jax.custom_batching.custom_vmap
    def _batched_op(x, y, s, dm, sender_t, receiver_ptr_t, perm):
        return _sharded_op(coef_bwd, x, y, s, dm, sender_t, receiver_ptr_t, perm)

    @_batched_op.def_vmap
    def _vmap_rule(axis_size, in_batched, x, y, s, dm, sender_t, receiver_ptr_t, perm):
        x_b, y_b, s_b, dm_b, sender_b, receiver_ptr_b, perm_b = in_batched
        graph_b = sender_b or receiver_ptr_b or perm_b
        if not x_b:
            x = jnp.broadcast_to(x[None], (axis_size,) + x.shape)
        if not y_b:
            y = jnp.broadcast_to(y[None], (axis_size,) + y.shape)
        if not s_b:
            s = jnp.broadcast_to(s[None], (axis_size,) + s.shape)
        if not dm_b:
            dm = jnp.broadcast_to(dm[None], (axis_size,) + dm.shape)
        if graph_b:
            # Batched graph: fold per-element graphs into one disjoint graph.
            if not (sender_b and receiver_ptr_b and perm_b):
                raise ValueError(_GRAPH_BATCH_INCONSISTENT)
            _warn_graph_batching()
        # `batched=False` tiles the shared (replicated) graph; both share the
        # same block-diagonal fold.
        sender_tt, receiver_ptr_tt = GraphCSR.fold_adjacency(
            axis_size, num_nodes, num_edges, sender_t, receiver_ptr_t, batched=graph_b
        )
        perm_t = GraphCSR.fold_permutation(axis_size, num_edges, perm, batched=graph_b)
        x = x.reshape((axis_size * num_nodes,) + x.shape[2:])
        y = y.reshape((axis_size * num_edges,) + y.shape[2:])
        s = s.reshape((axis_size * num_edges,) + s.shape[2:])
        dm = dm.reshape((axis_size * num_nodes,) + dm.shape[2:])

        dx, dy, ds = _sharded_op(
            coef_bwd, x, y, s, dm, sender_tt, receiver_ptr_tt, perm_t
        )

        dx = dx.reshape((axis_size, num_nodes) + dx.shape[1:])
        dy = dy.reshape((axis_size, num_edges) + dy.shape[1:])
        ds = ds.reshape((axis_size, num_edges) + ds.shape[1:])
        return (dx, dy, ds), (True, True, True)

    # custom_vjp: differentiable arguments only (x, y, s, dm).

    @custom_vjp
    def convolution_bwd_op(x, y, s, dm):
        return _batched_op(x, y, s, dm, sender_t, receiver_ptr_t, perm)

    def _fwd(x, y, s, dm):
        dx, dy, ds = convolution_bwd_op(x, y, s, dm)
        # Carry the graph adjacency through the residuals (as in the forward
        # `convolution._fwd`): the double backward runs in a separate trace, so
        # closing over `sender`/`receiver` -- or the transposed adjacency built
        # from them -- would leak them under a batched graph. `_bwd` rebuilds
        # everything in-trace from these residuals via fresh top-level calls.
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


convolution.Params = ConvolutionParams
