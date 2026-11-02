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

"""Pallas Mosaic GPU SymmetricContraction parameters: the common structure
(:class:`~e3j.pallas_ops.symmetric_contraction.common.params.PallasSymmetricContractionStructure`)
plus tuning knobs."""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from e3j.pallas_ops.symmetric_contraction.common.params import (
    PallasSymmetricContractionStructure,
)


class PallasMosaicGPUSymmetricContractionConfig(BaseModel, frozen=True):
    """Tuning knobs for the fused Mosaic GPU SymmetricContraction kernels.

    `None` picks a default (see `bwd.resolve_launch`).

    Attributes:
        warpgroups: Warpgroups per CTA, each walking its own batch rows.
        pipeline_depth: Rows in flight per warpgroup. By default, as many as
            fit in shared memory, up to `bwd.MAX_PIPELINE_DEPTH`.
        channel_block: Channels per CTA, by default 128. Must divide
            `channels`, be a multiple of 128, and be at most 256.
    """

    warpgroups: int | None = Field(default=None, gt=0)
    pipeline_depth: int | None = Field(default=None, gt=0)
    channel_block: int | None = Field(default=None, gt=0)


@dataclass(eq=False)
class PallasMosaicGPUSymmetricContractionParams(PallasSymmetricContractionStructure):
    """Parameters for a fused Pallas Mosaic GPU symmetric contraction.

    Attributes:
        config: Kernel tuning knobs.
    """

    config: PallasMosaicGPUSymmetricContractionConfig = field(
        default_factory=PallasMosaicGPUSymmetricContractionConfig
    )

    def _key(self) -> tuple:
        return (*super()._key(), self.config)
