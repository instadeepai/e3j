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

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, custom_vjp
from jax.ffi import ffi_call
from numpy import int32

from e3j.ops.coef import Coef4D
from e3j.utils import config


@dataclass
class ConvolutionParams:
    num_out: int
    num_scalars: int


@dataclass
class GraphCSR:
    """Compressed Sparse Row (CSR) graph representation."""

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


def convolution(
    coef: Array,
    x: Array,
    y: Array,
    s: Array,
    sender: Array,
    receiver: Array,
    params: ConvolutionParams,
) -> Array:
    """Equivariant convolution: fused tensor product + scalar mixing + aggregation.

    Gathers sender node features, computes the tensor product with edge
    spherical embeddings, mixes with radial scalars, and scatter-reduces
    by receiver.

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
    receiver_ptr = GraphCSR(num_nodes, sender, receiver).receiver_ptr

    channels_x = x.shape[-1] if x.ndim > 2 else 1
    num_out = params.num_out
    shape_out = (num_nodes, num_out, channels_x)

    if y.ndim == x.ndim and y.shape[-1] != 1:
        raise NotImplementedError("RHS y should have only one channel.")

    @custom_vjp
    def convolution_op(x, y, s):
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
            num_nodes=int32(num_nodes),
            debug=int32(config().debug_level),
        )

    def _fwd(x, y, s):
        z = convolution_op(x, y, s)
        return z, (x, y, s)

    def _bwd(res, dz):
        x, y, s = res
        return convolution_bwd(coef, x, y, s, sender, receiver, dz, params)

    convolution_op.defvjp(_fwd, _bwd)

    return convolution_op(x, y, s)


def convolution_bwd(coef, x, y, s, sender, receiver, dm, params):
    """Backward pass for equivariant convolution.

    Computes cotangents `dx`, `dy`, `dmix` from output cotangent `ct_z`
    by calling the fused backward kernel with transposed coefficients
    and transposed CSR adjacency.
    """
    num_nodes = x.shape[0]
    # Transpose CSR adjacency: group by sender instead of receiver.
    graph = GraphCSR(num_nodes, sender, receiver)
    perm, graph_t, _ = graph.transpose()

    with jax.ensure_compile_time_eval():
        c = Coef4D.unpack(coef, val_dtype="float32")
        coef_dmix = c.transpose((2, 0, 1, 3)).pack_jax()
        coef_dx = c.transpose((1, 0, 2, 3)).pack_jax()
        coef_dy = c.transpose((3, 0, 2, 1)).pack_jax()
        coef_3x = jnp.stack([coef_dmix, coef_dx, coef_dy])

    @custom_vjp
    def convolution_bwd_op(x, y, s, dm):
        dx, dy, dmix = ffi_call(
            "convolution_bwd",
            (
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(y.shape, y.dtype),
                jax.ShapeDtypeStruct(s.shape, s.dtype),
            ),
        )(
            coef_3x,
            x,
            y,
            dm,
            s,
            graph_t.sender,
            graph_t.receiver_ptr,
            perm,
            num_nodes=int32(num_nodes),
            debug=int32(config().debug_level),
        )

        return dx, dy, dmix

    conv = partial(
        convolution,
        coef=coef,
        sender=sender,
        receiver=receiver,
        params=params,
    )

    def _fwd(x, y, s, dm):
        dx, dy, ds = convolution_bwd_op(x, y, s, dm)
        return (dx, dy, ds), (x, y, s, dm)

    def _bwd(res, cotangents):
        """Return (Dx, Dy, Ds, Ddm) cotangents from (Ddx, Ddy, Dds)."""
        (Ddx, Ddy, Dds) = cotangents
        (x, y, s, dm) = res

        # Double variation of messages: three forward passes
        Ddm = conv(Ddx, y, s) + conv(x, Ddy, s) + conv(x, y, Dds)

        # Primal cotangents
        Dx_x, Dx_y, Dx_s = convolution_bwd_op(Ddx, y, s, dm)
        Dy_x, Dy_y, Dy_s = convolution_bwd_op(x, Ddy, s, dm)
        Ds_x, Ds_y, Ds_s = convolution_bwd_op(x, y, Dds, dm)

        Dx = Dy_x + Ds_x
        Dy = Dx_y + Ds_y
        Ds = Dx_s + Dy_s

        return (Dx, Dy, Ds, Ddm)

    convolution_bwd_op.defvjp(_fwd, _bwd)

    return convolution_bwd_op(x, y, s, dm)


convolution.Params = ConvolutionParams
