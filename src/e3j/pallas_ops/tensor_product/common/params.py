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

"""Common parameter definitions for Pallas tensor product kernels."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from e3j.spaces import O3Space
from e3j.utils import options

FwdConfig = TypeVar("FwdConfig")
BwdConfig = TypeVar("BwdConfig")


@dataclass
class PallasTensorProductParams(Generic[FwdConfig, BwdConfig]):
    """Parameters for a Pallas tensor product operation.

    Holds the sparse coefficient structure (idx + value in COO format), irreps
    definitions for inputs/output, and optional kernel tuning configs for the
    forward and backward passes.

    Attributes:
        indices: Sparse index array of shape `(nnz, 3)` in COO format. Each row
            `[oi, xi, yi]` maps an output index, an x-input index, and a
            y-input index to one non-zero coefficient. The forward pass groups
            indices by `oi`; the backward pass re-sorts by `xi`.
        values: Sparse coefficient values of shape `(nnz,)`. Each element
            corresponds to one row in `idx` and is multiplied with
            `x[xi] * y[yi]` during the kernel computation.
        layout: Memory layout for input/output arrays. Only
            `TRAILING_CHANNELS` is currently supported.
        mode: Tensor product contraction mode. `MAP` performs element-wise
            products (x and y share the same multiplicity); `OUTER` computes
            an outer product over channels.
        x_space: Space of the x input.
        y_space: Space of the y input.
        z_space: Space of the z (the output).
        fwd_config: Optional forward-pass kernel configuration. When None,
            default values are used.
        bwd_config: Optional backward-pass kernel configuration. When None,
            default values are used.
    """

    indices: np.ndarray
    values: np.ndarray
    layout: options.Layout
    mode: options.TPMode
    x_space: O3Space
    y_space: O3Space
    z_space: O3Space
    fwd_config: FwdConfig | None = None
    bwd_config: BwdConfig | None = None
