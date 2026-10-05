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

"""Fused message-passing convolution kernels written in Pallas."""

# Mosaic GPU is unavailable on jaxlib builds without CUDA support, and this
# module is imported unconditionally by `e3j.core`. Keep `import e3j` working
# there by exposing stubs that fail only when actually used.
try:
    from e3j.pallas_ops.convolution.mosaic_gpu import (
        PallasMosaicGPUMessagePassingConvolutionBwdConfig,
        PallasMosaicGPUMessagePassingConvolutionFwdConfig,
        PallasMosaicGPUMessagePassingConvolutionParams,
        convolution_mosaic_gpu,
    )
except ImportError as exc:  # pragma: no cover - depends on the jaxlib build

    def _mosaic_gpu_unavailable(*_args, _error=exc, **_kwargs):
        raise NotImplementedError(
            "Pallas Mosaic GPU is unavailable in this jaxlib build"
        ) from _error

    PallasMosaicGPUMessagePassingConvolutionParams = _mosaic_gpu_unavailable  # type: ignore[assignment]
    PallasMosaicGPUMessagePassingConvolutionFwdConfig = _mosaic_gpu_unavailable  # type: ignore[assignment]
    PallasMosaicGPUMessagePassingConvolutionBwdConfig = _mosaic_gpu_unavailable  # type: ignore[assignment]
    convolution_mosaic_gpu = _mosaic_gpu_unavailable  # type: ignore[assignment]

__all__ = [
    "PallasMosaicGPUMessagePassingConvolutionParams",
    "PallasMosaicGPUMessagePassingConvolutionFwdConfig",
    "PallasMosaicGPUMessagePassingConvolutionBwdConfig",
    "convolution_mosaic_gpu",
]
