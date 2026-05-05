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


def _convolution_fwd(coef, x, y, s, s_index, sender, receiver_ptr, params):
    z = convolution(coef, x, y, s, s_index, sender, receiver_ptr, params)
    return z, (coef, x, y, s, s_index, sender, receiver_ptr)


def _convolution_bwd(params, res, ct_z):
    raise NotImplementedError("Backward pass for convolution is not yet implemented.")


convolution.defvjp(_convolution_fwd, _convolution_bwd)


convolution.Params = ConvolutionParams
