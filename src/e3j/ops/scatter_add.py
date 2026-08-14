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

from functools import partial

import e3j_ops
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, custom_vjp
from jax.core import ShapedArray
from jax.ffi import ffi_call


@partial(custom_vjp)
def scatter_add_1(indices: Array, values: Array, out: Array) -> Array:
    """Scatter add operation optimised to use in warp sum reductions.

    Semantics are identical to:
    https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scatter_add.html.

    Expects a 1D index array and an N-D value array:

        (num_idx,) x (..., num_idx) -> (..., num_out)

    Args:
        indices: The indices to scatter the values.
        values: The values to scatter.
        out: The output array.

    Returns:
        The output array with the values scattered.

    """
    # The launcher rejects it too, but only once the kernel runs.
    if values.dtype == jnp.float16:
        raise NotImplementedError(
            "float16 values are not supported by scatter_add_1: the reduction "
            "uses atomicAdd, which has no __half overload. Use float32/float64, "
            "or another aggregation method."
        )

    num_out = out.shape[-1]
    # GOTCHA: see e3j/ops/README.md
    args = (indices, values)
    return ffi_call(
        "scatter_add_1",
        jax.ShapeDtypeStruct(out.shape, values.dtype),
    )(
        *args,
        num_out=np.int32(num_out),
    )


def scatter_add_1_fwd(idx, val, out):
    y = scatter_add_1(idx, val, out)
    return y, (idx,)


def scatter_add_1_bwd(res, ct_out):
    idx, *_ = res
    ct_val = ct_out[:, idx]
    return (idx, ct_val, ct_out)


scatter_add_1.defvjp(scatter_add_1_fwd, scatter_add_1_bwd)
