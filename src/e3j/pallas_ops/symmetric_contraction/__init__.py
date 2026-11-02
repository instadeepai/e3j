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

"""Fused SymmetricContraction kernels written in Pallas."""

from e3j.pallas_ops.symmetric_contraction.mosaic_tpu import (
    PallasMosaicTPUSymmetricContractionParams,
    symmetric_contraction_pallas_mosaic_tpu,
)

# jaxlib builds without CUDA lack Mosaic GPU. `e3j.core` imports this module
# unconditionally, so expose stubs that only fail when called (as
# `pallas_ops.tensor_product` does).
try:
    from e3j.pallas_ops.symmetric_contraction.mosaic_gpu import (
        PallasMosaicGPUSymmetricContractionConfig,
        PallasMosaicGPUSymmetricContractionParams,
        symmetric_contraction_pallas_mosaic_gpu,
    )
except ImportError as exc:  # pragma: no cover - depends on the jaxlib build

    def _mosaic_gpu_unavailable(*_args, _error=exc, **_kwargs):
        raise NotImplementedError(
            "Pallas Mosaic GPU is unavailable in this jaxlib build"
        ) from _error

    PallasMosaicGPUSymmetricContractionConfig = _mosaic_gpu_unavailable  # type: ignore[assignment]
    PallasMosaicGPUSymmetricContractionParams = _mosaic_gpu_unavailable  # type: ignore[assignment]
    symmetric_contraction_pallas_mosaic_gpu = _mosaic_gpu_unavailable  # type: ignore[assignment]

__all__ = [
    "PallasMosaicGPUSymmetricContractionConfig",
    "PallasMosaicGPUSymmetricContractionParams",
    "PallasMosaicTPUSymmetricContractionParams",
    "symmetric_contraction_pallas_mosaic_gpu",
    "symmetric_contraction_pallas_mosaic_tpu",
]
