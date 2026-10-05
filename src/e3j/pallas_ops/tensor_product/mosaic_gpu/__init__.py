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

"""Fused sparse tensor product on Pallas Mosaic GPU."""

from e3j.pallas_ops.tensor_product.mosaic_gpu.params import (
    PallasMosaicGPUTensorProductParams,
    PallasMosaicGPUTensorProductParamsBwdConfig,
    PallasMosaicGPUTensorProductParamsFwdConfig,
)
from e3j.pallas_ops.tensor_product.mosaic_gpu.primitives import (
    tensor_product_pallas_mosaic_gpu,
)

__all__ = [
    "PallasMosaicGPUTensorProductParams",
    "PallasMosaicGPUTensorProductParamsFwdConfig",
    "PallasMosaicGPUTensorProductParamsBwdConfig",
    "tensor_product_pallas_mosaic_gpu",
]
