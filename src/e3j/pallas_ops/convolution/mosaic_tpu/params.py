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

"""Pallas Mosaic-TPU message-passing convolution parameters."""

from dataclasses import dataclass, field, replace

import numpy as np
from pydantic import BaseModel

from e3j.pallas_ops.convolution.common.params import (
    PallasMessagePassingConvolutionParams,
)
from e3j.utils import options


class PallasMosaicTPUMessagePassingConvolutionFwdConfig(BaseModel, frozen=True):
    batch_block_size: int | None = None  # None => auto-fit to the VMEM budget
    assume_receiver_sorted: bool = False


class PallasMosaicTPUMessagePassingConvolutionBwdConfig(BaseModel, frozen=True):
    batch_block_size: int | None = None  # None => auto-fit to the VMEM budget
    assume_sender_sorted: bool = False


@dataclass
class PallasMosaicTPUMessagePassingConvolutionParams(
    PallasMessagePassingConvolutionParams[
        PallasMosaicTPUMessagePassingConvolutionFwdConfig,
        PallasMosaicTPUMessagePassingConvolutionBwdConfig,
    ]
):
    """Parameters for a Pallas Mosaic-TPU fused message-passing operation.

    Holds the sparse Clebsch-Gordan structure, irrep spaces of inputs and
    output, and the per-output-coordinate target-irrep-block index used for
    fused scalar mixing.

    Attributes:
        indices: Sparse index array of shape `(nnz, 3)`. Each row
            `[zi, xi, yi]` indexes one non-zero CG coefficient.
        values: Sparse coefficient values of shape `(nnz,)`.
        x_space / y_space / z_space: O3-spaces of the two inputs and output.
        layout: Must be `TRAILING_CHANNELS`.
        fwd_config: Optional forward-pass kernel config.
        bwd_config: Optional backward-pass kernel config.
        irrep_block_of_output: Filled in `__post_init__` — maps each output coord
            to its target-irrep-block index (used for scalar mixing).
    """

    irrep_block_of_output: np.ndarray = field(init=False)

    def __post_init__(self):
        assert (
            self.layout == options.Layout.TRAILING_CHANNELS
        ), f"Layout {self.layout} is not supported; only TRAILING_CHANNELS."
        irrep_block_of_output = np.empty((self.z_space.dim,), dtype=np.int32)
        offset = 0
        block_idx = 0
        for channels, ir in self.z_space:
            d = 2 * ir.l + 1
            for _ in range(channels):
                irrep_block_of_output[offset : offset + d] = block_idx
                offset += d
                block_idx += 1
        assert offset == self.z_space.dim
        self.irrep_block_of_output = irrep_block_of_output
        if self.fwd_config is None:
            self.fwd_config = PallasMosaicTPUMessagePassingConvolutionFwdConfig()
        if self.bwd_config is None:
            self.bwd_config = PallasMosaicTPUMessagePassingConvolutionBwdConfig()

    @classmethod
    def build_from_sender_sorted(cls, indices, values, x_space, y_space, z_space):
        """Build params for a sender-sorted, symmetric edge list.

        The backward runs these natural coefficients over the (sender-sorted)
        edges; the forward runs `swapped()` (parity-folded coef) over the
        edge-reversed list.
        """
        return cls(
            indices=indices,
            values=values,
            layout=options.Layout.TRAILING_CHANNELS,
            x_space=x_space,
            y_space=y_space,
            z_space=z_space,
            fwd_config=PallasMosaicTPUMessagePassingConvolutionFwdConfig(
                assume_receiver_sorted=True
            ),
            bwd_config=PallasMosaicTPUMessagePassingConvolutionBwdConfig(
                assume_sender_sorted=True
            ),
        )

    def swapped(self) -> "PallasMosaicTPUMessagePassingConvolutionParams":
        """Params for the edge-reversed pass (parity-folded coef).

        Reversing edges flips the sign of odd-parity `y` irreps; fold that into
        `values`. Everything else (spaces, configs) is shared.
        """
        y_signs = np.concatenate(
            [
                np.full(channels * (2 * ir.l + 1), float(ir.p))
                for channels, ir in self.y_space
            ]
        )
        return replace(self, values=self.values * y_signs[self.indices[:, 2]])
