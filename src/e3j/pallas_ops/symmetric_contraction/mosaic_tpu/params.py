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

"""Pallas Mosaic TPU symmetric contraction parameters: the common structure
plus batch block sizes."""

from dataclasses import dataclass

from e3j.pallas_ops.symmetric_contraction.common.params import (
    PallasSymmetricContractionStructure,
)
from e3j.pallas_ops.utils.mosaic_tpu import validate_batch_block_size

# Blocks under 8 rows compute garbage on libtpu >= 0.0.42 (v4 CMEM); pad the batch instead.
BATCH_BLOCK_SIZE_CANDIDATES = [8, 16, 32, 64, 128]


@dataclass(eq=False)
class PallasMosaicTPUSymmetricContractionParams(PallasSymmetricContractionStructure):
    """Parameters for a fused Pallas Mosaic TPU symmetric contraction.

    Attributes:
        fwd_batch_block_size: Batch rows per forward pipeline step. None
            picks the largest that fits in VMEM (:func:`.fwd.block_size`).
        bwd_batch_block_size: The same, for the backward.
    """

    fwd_batch_block_size: int | None = None
    bwd_batch_block_size: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for bbs in (self.fwd_batch_block_size, self.bwd_batch_block_size):
            if bbs is not None:
                validate_batch_block_size(bbs, BATCH_BLOCK_SIZE_CANDIDATES)

    def _key(self) -> tuple:
        return (*super()._key(), self.fwd_batch_block_size, self.bwd_batch_block_size)
