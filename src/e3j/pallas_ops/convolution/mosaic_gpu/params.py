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

"""Parameters and configuration for the Pallas Mosaic GPU convolution."""

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from e3j.pallas_ops.convolution.common.params import (
    PallasMessagePassingConvolutionParams,
)
from e3j.spaces import O3Space
from e3j.utils import options

_RECEIVER = options.GraphOrdering.RECEIVER
_SENDER = options.GraphOrdering.SENDER


class PallasMosaicGPUMessagePassingConvolutionFwdConfig(BaseModel, frozen=True):
    """Settings for the forward kernel.

    Attributes:
        pipeline_depth: How many edges each node loads ahead while it works
            through its edge list.
        grouping: Which endpoint of the edges each CTA (one per node) is
            responsible for. `None` uses `graph_ordering`. `RECEIVER` is
            correct on any graph. `SENDER` is only correct under the symmetry
            assumptions of :class:`e3j.core.Convolution`. If this differs
            from `graph_ordering`, each call pays for one `argsort`.
        channel_block: How many channels one kernel launch handles. `None`
            picks a value based on `l_max`. It must divide `channels`, be a
            multiple of 128, and be at most 256.
    """

    pipeline_depth: int = Field(default=2, gt=0)
    grouping: options.GraphOrdering | None = None
    channel_block: int | None = Field(default=None, gt=0)


class PallasMosaicGPUMessagePassingConvolutionBwdConfig(BaseModel, frozen=True):
    """Settings for the backward kernel.

    Attributes:
        pipeline_depth: How many edges each node loads ahead. Only the
            receiver-grouped kernel uses it.
        grouping: Which endpoint of the edges each CTA is responsible for.
            `None` uses `graph_ordering`. Both choices give exact results;
            if this differs from `graph_ordering`, each call pays for one
            `argsort`.
        channel_block: Same as in the forward config.
    """

    pipeline_depth: int = Field(default=2, gt=0)
    grouping: options.GraphOrdering | None = options.GraphOrdering.RECEIVER
    channel_block: int | None = Field(default=None, gt=0)


def y_parity_signs(y_space: O3Space) -> np.ndarray:
    """Parity sign of each coordinate of `y_space`.

    Reversing an edge multiplies its features by these signs:
    `y_ba = signs * y_ab`.
    """
    return np.concatenate(
        [np.full(mul * ir.dim, float(ir.p), np.float32) for mul, ir in y_space]
    )


@dataclass
class PallasMosaicGPUMessagePassingConvolutionParams(
    PallasMessagePassingConvolutionParams[
        PallasMosaicGPUMessagePassingConvolutionFwdConfig,
        PallasMosaicGPUMessagePassingConvolutionBwdConfig,
    ]
):
    """Parameters of the Mosaic GPU fused message-passing convolution.

    Attributes:
        indices: `(nnz, 3)` Clebsch-Gordan indices, one `[zi, xi, yi]` per row.
        values: `(nnz,)` Clebsch-Gordan values.
        layout: Must be `TRAILING_CHANNELS`.
        graph_ordering: The endpoint (sender or receiver) the edge list is
            sorted by.
        scalar_of_output: For each output coordinate, the index of the edge
            scalar that multiplies it.
        num_scalars: Number of edge scalars, one per irrep of `z_space`.
    """

    graph_ordering: options.GraphOrdering = options.GraphOrdering.RECEIVER
    scalar_of_output: np.ndarray = field(init=False)
    num_scalars: int = field(init=False)

    def __post_init__(self):
        if self.layout != options.Layout.TRAILING_CHANNELS:
            raise NotImplementedError(
                f"Mosaic GPU convolution supports only "
                f"{options.Layout.TRAILING_CHANNELS}, got {self.layout}"
            )
        fwd_grouping = self.fwd_config.grouping if self.fwd_config else None
        bwd_grouping = self.bwd_config.grouping if self.bwd_config else None
        for name, ordering, optional in (
            ("graph_ordering", self.graph_ordering, False),
            ("fwd_config.grouping", fwd_grouping, True),
            ("bwd_config.grouping", bwd_grouping, True),
        ):
            if ordering is None and optional:
                continue
            if ordering not in (_RECEIVER, _SENDER):
                raise NotImplementedError(
                    f"Mosaic GPU convolution supports only {_RECEIVER} and "
                    f"{_SENDER}, got {name}={ordering}"
                )
        # Each irrep of the output space has one edge scalar, shared by all of
        # its coordinates. This is the same mapping as
        # `ScalarMixing.mix_indices`, stored as a numpy array so the kernel can
        # use it as a constant while tracing.
        self.num_scalars = sum(mul for mul, _ in self.z_space)
        self.scalar_of_output = np.repeat(
            np.arange(self.num_scalars, dtype=np.int32),
            [ir.dim for mul, ir in self.z_space for _ in range(mul)],
        )
        if self.fwd_config is None:
            self.fwd_config = PallasMosaicGPUMessagePassingConvolutionFwdConfig()
        if self.bwd_config is None:
            self.bwd_config = PallasMosaicGPUMessagePassingConvolutionBwdConfig()

    @property
    def l_max(self) -> int:
        """Maximal angular momentum across the three spaces."""
        return max(self.x_space.l_max, self.y_space.l_max, self.z_space.l_max)

    @property
    def fwd_grouping(self) -> options.GraphOrdering:
        """The grouping the forward actually uses."""
        return self.fwd_config.grouping or self.graph_ordering

    @property
    def bwd_grouping(self) -> options.GraphOrdering:
        """The grouping the backward actually uses."""
        return self.bwd_config.grouping or self.graph_ordering

    @property
    def forward_values(self) -> np.ndarray:
        """Clebsch-Gordan values to use in the forward kernel.

        With `SENDER` grouping, the kernel computes each message on the
        reversed edge. On a symmetric graph, multiplying by the parity signs
        of `y` turns that back into the message the receiver should get, so
        the values are pre-multiplied by those signs in that case.
        """
        if self.fwd_grouping != _SENDER:
            return self.values
        return self.values * y_parity_signs(self.y_space)[self.indices[:, 2]]
