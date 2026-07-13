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
        RECEIVER: Edges sorted by receiver index (default). The forward pass
            uses the CSR adjacency directly; the backward pass transposes the
            graph (sort by sender) and threads an edge permutation.
        SENDER: Edges sorted by sender index. Flips the two roles: the backward
            pass becomes natural (no transpose, no permutation), while the
            forward pass aggregates at the sender node and recovers the true
            receiver message through a per-`y`-slice reversal sign baked into
            the coefficients. Only valid for a symmetric graph with graded-
            symmetric edge features and reversal-symmetric scalars (see
            :class:`~e3j.core.Convolution`).
    """

    RECEIVER = "RECEIVER"
    SENDER = "SENDER"


# FIXME: Integer code translation of enums for the XLA handler.
#        See lib/e3j_ops/ffi/e3j_ops.h
# We cannot pass C++ strings to the XLA-FFI handler yet,
# but could rely on a C++ binding for the enum conversion
# outside of the XLA handler.


class Layout(EnumOption):
    """Array layouts.

    Values:
        * LEADING_CHANNELS
        * TRAILING_CHANNELS
        * E3NN
    """

    LEADING_CHANNELS = 0
    TRAILING_CHANNELS = 1
    E3NN = 2


class TPMode(EnumOption):
    """Tensor product modes.

    Values:
        * OUTER:  "u -> v -> (u,v)"
        * INNER:  "v -> v -> 1"
        * MAP:    "v -> v -> v"

    The 'MAP' mode is only useful with trailing channels,
    since a map over leading axes is performed in any case.
    """

    OUTER = 0
    INNER = 1
    MAP = 2


class TPNormalization(EnumOption):
    """Tensor product normalization options of Clebsch-Gordan coefficients.

    Values:
        * NONE: orthonormal Clebsch-Gordan coefficients (default).
        * SQRT_DIM_OUT: coefficients scaled by sqrt(2L+1).
    """

    NONE = "NONE"
    SQRT_DIM_OUT = "SQRT_DIM_OUT"


class LinearInitialization(EnumOption):
    """Linear weight initialization options.

    Values:
        * FAN_IN: stddev is 1/sqrt(m_in) per block (e3nn: "path").
        * FAN_OUT: stddev is 1/sqrt(m_out) per block (e3nn: "irrep").
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
