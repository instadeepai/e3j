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

from enum import Enum

from e3j.utils.enum_option import EnumOption


class TensorProduct(Enum):
    """Evaluation strategy for :class:`~e3j.core.TensorProduct`.

    Values:
        SPARSE: Pull-back inputs by CG indices, multiply, then aggregate.
        DENSE: Evaluate via `einsum` on dense CG coefficient array.
        FUSED: Dispatch to custom CUDA kernel via XLA-FFI (requires `e3j_ops`).
        FUSED_MOSAIC_TPU: Dispatch to the Pallas Mosaic TPU kernel.
    """

    FUSED = "FUSED"
    SPARSE = "SPARSE"
    DENSE = "DENSE"
    FUSED_MOSAIC_TPU = "FUSED_MOSAIC_TPU"


class Aggregation(Enum):
    """Aggregation method for the sparse reduction step.

    Values:
        SCATTER: `jax.lax.scatter_add` — JAX-native, CPU and GPU compatible.
        SCATTER_1: Custom CUDA `scatter_add_1` kernel (requires `e3j_ops`).
        SPARSE: Matmul with a sparse BCOO target matrix.
        DENSE: Matmul with a dense target matrix.
    """

    SCATTER = "SCATTER"
    SCATTER_1 = "SCATTER_1"
    SPARSE = "SPARSE"
    DENSE = "DENSE"


class Convolution(Enum):
    """Evaluation strategy for :class:`~e3j.core.Convolution`.

    Values:
        UNFUSED: Naive JAX implementation.
        FUSED_CUDA: Dispatch to custom CUDA kernel via XLA-FFI (requires `e3j_ops`).
        FUSED_MOSAIC_TPU: Single fused Pallas Mosaic-TPU kernel (gather + TP +
            mixing + scatter), ``TRAILING_CHANNELS`` layout. Requires a TPU backend.
    """

    UNFUSED = "UNFUSED"
    FUSED_CUDA = "FUSED_CUDA"
    FUSED_MOSAIC_TPU = "FUSED_MOSAIC_TPU"


class GraphOrdering(EnumOption):
    """Edge ordering contract for the fused CUDA :class:`~e3j.core.Convolution`.

    Values:
        RECEIVER: Edges sorted by receiver index. The forward pass can then
            use the CSR adjacency directly to aggregate messages, while the
            backward pass needs to transpose edges.
        SENDER: Edges sorted by sender index. The forward pass will leverage
            symmetry assumptions on the graph and edge features, while the
            backward pass can naturally aggregate cotangents using the CSR
            adjacency matrix. See :class:`~e3j.core.Convolution`.
        NONE: Use this flag in combination with `Convolution.UNFUSED` option
            as fallback when no explicit edge ordering is enforced.
    """

    RECEIVER = "RECEIVER"
    SENDER = "SENDER"
    NONE = "NONE"


# FIXME: Integer code translation of enums for the XLA handler.
#        See lib/e3j_ops/ffi/e3j_ops.h
# We cannot pass C++ strings to the XLA-FFI handler yet,
# but could rely on a C++ binding for the enum conversion
# outside of the XLA handler.


class Layout(EnumOption):
    """Memory layout for equivariant arrays.

    An equivariant feature space such as `"128x0e + 64x1o"` is built from
    irreducible blocks with multiplicity `m` and irrep dimension `2l + 1`
    for each degree `l`.

    The layout controls how different blocks are arranged in memory, eventually
    factorizing the GCD of multiplicities in a separate `channels` axis.

    Values:
        LEADING_CHANNELS: `(batch, channels, dim)` -- channels on an explicit
            leading axis.
        TRAILING_CHANNELS: `(batch, dim, channels)` -- channels on a trailing
            axis. Faster on GPU thanks to coalesced memory access.
        E3NN: `(batch, channels * dim)` -- channels folded into the feature
            dimension, matching `e3nn_jax.IrrepsArray`.

    .. note::
        The CUDA kernels only support channel counts that are powers of two for now.
    """

    LEADING_CHANNELS = 0
    TRAILING_CHANNELS = 1
    E3NN = 2


class TPMode(EnumOption):
    """Tensor product modes.

    Values:
        OUTER: "u -> v -> (u,v)"
        INNER: "v -> v -> 1"
        MAP: "v -> v -> v"

    The 'MAP' mode is only useful with trailing channels,
    since a map over leading axes is performed in any case.
    """

    OUTER = 0
    INNER = 1
    MAP = 2


class TPNormalization(EnumOption):
    """Tensor product normalization options of Clebsch-Gordan coefficients.

    Values:
        NONE: orthonormal Clebsch-Gordan coefficients (default).
        SQRT_DIM_OUT: coefficients scaled by sqrt(2L+1).
    """

    NONE = "NONE"
    SQRT_DIM_OUT = "SQRT_DIM_OUT"


class LinearInitialization(EnumOption):
    """Linear weight initialization options.

    Values:
        FAN_IN: stddev is 1/sqrt(m_in) per block (e3nn: "path").
        FAN_OUT: stddev is 1/sqrt(m_out) per block (e3nn: "irrep").
    """

    FAN_IN = "FAN_IN"
    FAN_OUT = "FAN_OUT"


class LinearIndexwiseInitialization(EnumOption):
    """LinearIndexwise weight initialization options.

    Values:
        FAN_IN: stddev is 1/sqrt(m_in) per block (e3nn: "path").
        FAN_OUT: stddev is 1/sqrt(m_out) per block (e3nn: "irrep").
        FAN_IN_FCTP: stddev is 1/sqrt(m_in * num_indices), matching
            the effective normalization of a FullyConnectedTensorProduct
            with a one-hot species input.
    """

    FAN_IN = "FAN_IN"
    FAN_OUT = "FAN_OUT"
    FAN_IN_FCTP = "FAN_IN_FCTP"
