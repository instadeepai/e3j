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

"""Fused message-passing convolution written with Pallas Mosaic GPU."""

from e3j.pallas_ops.convolution.mosaic_gpu.params import (
    PallasMosaicGPUMessagePassingConvolutionBwdConfig,
    PallasMosaicGPUMessagePassingConvolutionFwdConfig,
    PallasMosaicGPUMessagePassingConvolutionParams,
    y_parity_signs,
)
from e3j.pallas_ops.convolution.mosaic_gpu.primitives import convolution_mosaic_gpu

__all__ = [
    "convolution_mosaic_gpu",
    "y_parity_signs",
    "PallasMosaicGPUMessagePassingConvolutionBwdConfig",
    "PallasMosaicGPUMessagePassingConvolutionFwdConfig",
    "PallasMosaicGPUMessagePassingConvolutionParams",
]
