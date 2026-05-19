from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, custom_vjp
from jax.ffi import ffi_call
from numpy import int32

from e3j.ops.coef import Coef
from e3j.utils import config


@dataclass
class ConvolutionParams:
    num_out: int
    num_scalars: int


@partial(custom_vjp, nondiff_argnums=(7,))
def convolution(
    coef: Array,
    x: Array,
    y: Array,
    s: Array,
    s_index: Array,
    sender: Array,
    receiver_ptr: Array,
    params: ConvolutionParams,
) -> Array:
    """Equivariant convolution: fused tensor product + scalar mixing + aggregation.

    Gathers sender node features, computes the tensor product with edge
    spherical embeddings, mixes with radial scalars, and scatter-reduces
    by receiver.

    Args:
        coef: Packed CG coefficients (opaque idx_dtype vector).
        x: Node features, shape (num_nodes, num_x, channels_x).
        y: Edge embeddings, shape (num_edges, num_y).
        s: Radial scalars, shape (num_edges, num_scalars, channels_x).
        s_index: Output irrep index map, shape (num_out,).
        sender: CSR sender indices, shape (num_edges,).
        receiver_ptr: CSR receiver pointers, shape (num_nodes + 1,).
        params: Convolution parameters.

    Returns:
        Output node features, shape (num_nodes, num_out, channels_x).
    """
    num_nodes = x.shape[0]
    channels_x = x.shape[-1] if x.ndim > 2 else 1
    num_out = params.num_out
    shape_out = (num_nodes, num_out, channels_x)

    if y.ndim == x.ndim and y.shape[-1] != 1:
        raise NotImplementedError("RHS y should have only one channel.")
    if s_index.size != num_out:
        raise ValueError("Scalar indices `s_index` should be of length `num_out`.")

    return ffi_call(
        "convolution",
        jax.ShapeDtypeStruct(shape_out, x.dtype),
    )(
        coef,
        x,
        y,
        s,
        s_index,
        sender,
        receiver_ptr,
        num_nodes=int32(num_nodes),
        debug=int32(config().debug_level),
    )


def convolution_bwd(coef, x, y, s, s_index, sender, receiver_ptr, ct_z, params):
    """Backward pass for equivariant convolution.

    Computes cotangents `dx`, `dy`, `dmix` from output cotangent `ct_z`
    by calling the fused backward kernel with transposed coefficients
    and transposed CSR adjacency.
    """
    num_nodes = x.shape[0]

    with jax.ensure_compile_time_eval():
        # Transpose and pack 3x coefs: [coef_fwd, coef_dx, coef_dy]
        c = Coef.unpack(coef, val_dtype="float32")
        val, idx = c.val, c.idx.T

        sigma_dx = jnp.argsort(idx[1])
        val_dx = val[sigma_dx]
        idx_dx = jnp.stack([idx[1][sigma_dx], idx[0][sigma_dx], idx[2][sigma_dx]])
        coef_dx = Coef(val_dx, idx_dx.T).pack_jax()

        sigma_dy = jnp.argsort(idx[2])
        val_dy = val[sigma_dy]
        idx_dy = jnp.stack([idx[2][sigma_dy], idx[0][sigma_dy], idx[1][sigma_dy]])
        coef_dy = Coef(val_dy, idx_dy.T).pack_jax()

        coef_3x = jnp.stack([coef, coef_dx, coef_dy])

        # Transpose CSR adjacency: group by sender instead of receiver.
        # Although the graph is symmetric, the edge features may not be.
        num_edges = sender.shape[0]

        # Expand receiver_ptr -> per-edge receiver
        receiver = jnp.zeros(num_edges, dtype=jnp.int32)
        for r in range(num_nodes):
            receiver = receiver.at[receiver_ptr[r] : receiver_ptr[r + 1]].set(r)

        # Argsort edges by sender
        perm = jnp.argsort(sender)
        sorted_sender = sender[perm]

        # Build sender_ptr from sorted sender via bincount + cumsum
        counts = jnp.bincount(sorted_sender, length=num_nodes)
        sender_ptr_t = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(counts)]
        )
        sender_ptr_t = sender_ptr_t.astype(jnp.int32)

        # Permuted receiver and per-edge inputs
        receiver_t = receiver[perm]

        # Inverse permutation for un-permuting edge outputs
        inv_perm = jnp.argsort(perm)

    y_t = y[perm]
    s_t = s[perm]

    dx, dy_t, dmix_t = ffi_call(
        "convolution_bwd",
        (
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(y.shape, y.dtype),
            jax.ShapeDtypeStruct(s.shape, s.dtype),
        ),
    )(
        coef_3x,
        x,
        y_t,
        ct_z,
        s_t,
        s_index,
        receiver_t,
        sender_ptr_t,
        num_nodes=int32(num_nodes),
        debug=int32(config().debug_level),
    )

    dy = dy_t[inv_perm]
    dmix = dmix_t[inv_perm]
    return dx, dy, dmix


def _convolution_fwd(coef, x, y, s, s_index, sender, receiver_ptr, params):
    z = convolution(coef, x, y, s, s_index, sender, receiver_ptr, params)
    return z, (coef, x, y, s, s_index, sender, receiver_ptr)


def _convolution_bwd(params, res, ct_z):
    coef, x, y, s, s_index, sender, receiver_ptr = res
    dx, dy, dmix = convolution_bwd(
        coef, x, y, s, s_index, sender, receiver_ptr, ct_z, params
    )
    return (
        jnp.zeros_like(coef),
        dx,
        dy,
        dmix,
        jnp.zeros_like(s_index),
        jnp.zeros_like(sender),
        jnp.zeros_like(receiver_ptr),
    )


convolution.defvjp(_convolution_fwd, _convolution_bwd)


convolution.Params = ConvolutionParams
