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

from e3j.pallas_ops.tensor_product.common.params import PallasTensorProductParams
from e3j.pallas_ops.tensor_product.mosaic_tpu import tensor_product_pallas_mosaic_tpu
from e3j.pallas_ops.tensor_product.mosaic_tpu.params import (
    PallasMosaicTPUTensorProductParams,
    PallasMosaicTPUTensorProductParamsBwdConfig,
    PallasMosaicTPUTensorProductParamsFwdConfig,
)

__all__ = [
    "PallasTensorProductParams",
    "PallasMosaicTPUTensorProductParams",
    "PallasMosaicTPUTensorProductParamsFwdConfig",
    "PallasMosaicTPUTensorProductParamsBwdConfig",
    "tensor_product_pallas_mosaic_tpu",
]

# Mosaic GPU is unavailable on jaxlib builds without CUDA support, and this
# module is imported unconditionally by `e3j.core`. Keep `import e3j` working
# there by exposing stubs that fail only when actually used, as
# `pallas_ops.convolution` does.
try:
    from e3j.pallas_ops.tensor_product.mosaic_gpu import (  # noqa: F401
        PallasMosaicGPUTensorProductParams,
        PallasMosaicGPUTensorProductParamsBwdConfig,
        PallasMosaicGPUTensorProductParamsFwdConfig,
        tensor_product_pallas_mosaic_gpu,
    )
except ImportError as exc:  # pragma: no cover - depends on the jaxlib build
    _MOSAIC_GPU_IMPORT_ERROR = exc

    def _mosaic_gpu_unavailable(*_args, **_kwargs):
        raise NotImplementedError(
            "Pallas Mosaic GPU is unavailable in this jaxlib build"
        ) from _MOSAIC_GPU_IMPORT_ERROR

    PallasMosaicGPUTensorProductParams = _mosaic_gpu_unavailable  # type: ignore[assignment]
    PallasMosaicGPUTensorProductParamsFwdConfig = _mosaic_gpu_unavailable  # type: ignore[assignment]
    PallasMosaicGPUTensorProductParamsBwdConfig = _mosaic_gpu_unavailable  # type: ignore[assignment]
    tensor_product_pallas_mosaic_gpu = _mosaic_gpu_unavailable  # type: ignore[assignment]

__all__ += [
    "PallasMosaicGPUTensorProductParams",
    "PallasMosaicGPUTensorProductParamsFwdConfig",
    "PallasMosaicGPUTensorProductParamsBwdConfig",
    "tensor_product_pallas_mosaic_gpu",
]
