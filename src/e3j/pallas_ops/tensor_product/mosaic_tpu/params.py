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
from typing import Annotated, Optional

import numpy as np
from pydantic import AfterValidator, BaseModel

from e3j.pallas_ops.tensor_product.common.params import PallasTensorProductParams
from e3j.pallas_ops.utils.mosaic_tpu import DEFAULT_BATCH_BLOCK_SIZE_CANDIDATES
from e3j.utils import options


def _validate_batch_block_size(v: int) -> int:
    if v not in DEFAULT_BATCH_BLOCK_SIZE_CANDIDATES:
        raise ValueError(
            f"batch_block_size must be one of {DEFAULT_BATCH_BLOCK_SIZE_CANDIDATES}, got {v}"
        )
    return v


BatchBlockSize = Annotated[int, AfterValidator(_validate_batch_block_size)]


class PallasMosaicTPUTensorProductParamsFwdConfig(BaseModel, frozen=True):
    batch_block_size: Optional[BatchBlockSize] = None


class PallasMosaicTPUTensorProductParamsBwdConfig(BaseModel, frozen=True):
    batch_block_size: Optional[BatchBlockSize] = None


@dataclass
class PallasMosaicTPUTensorProductParams(
    PallasTensorProductParams[
        PallasMosaicTPUTensorProductParamsFwdConfig,
        PallasMosaicTPUTensorProductParamsBwdConfig,
    ]
):
    """Parameters for a Pallas mosaic TPU tensor product operation."""

    def __post_init__(self):
        assert (
            self.layout == options.Layout.TRAILING_CHANNELS
        ), f"Layout {self.layout} is not supported."

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PallasMosaicTPUTensorProductParams):
            return NotImplemented
        return (
            np.array_equal(self.indices, other.indices)
            and np.array_equal(self.values, other.values)
            and self.layout == other.layout
            and self.mode == other.mode
            and self.x_space == other.x_space
            and self.y_space == other.y_space
            and self.z_space == other.z_space
            and self.fwd_config == other.fwd_config
            and self.bwd_config == other.bwd_config
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.indices.shape,
                self.indices.tobytes(),
                self.values.shape,
                self.values.tobytes(),
                self.layout,
                self.mode,
                self.x_space,
                self.y_space,
                self.z_space,
                self.fwd_config,
                self.bwd_config,
            )
        )
