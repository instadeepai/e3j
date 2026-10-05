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

"""Parameters and configuration for the Pallas Mosaic GPU tensor product."""

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from e3j.pallas_ops.tensor_product.common.params import PallasTensorProductParams
from e3j.utils import options


class _CommonConfig(BaseModel, frozen=True):
    """Tuning knobs shared by the forward and backward kernels.

    Attributes:
        rows_per_block: Batch rows per CTA; must divide the batch. `None` uses
            :data:`common.DEFAULT_ROWS_PER_BLOCK`.
        pipeline_depth: `emit_pipeline`'s `max_concurrent_steps`. Each slot
            uses SMEM. Has no effect when `rows_per_block=1`.
        channel_block: Channels per CTA, 128 or 256; must divide `channels`.
            `None` uses 128.
    """

    rows_per_block: int | None = Field(default=None, gt=0)
    pipeline_depth: int = Field(default=2, gt=0)
    channel_block: int | None = Field(default=None, gt=0)


class PallasMosaicGPUTensorProductParamsFwdConfig(_CommonConfig, frozen=True):
    """Forward-pass kernel configuration."""


class PallasMosaicGPUTensorProductParamsBwdConfig(_CommonConfig, frozen=True):
    """Backward-pass kernel configuration."""


@dataclass
class PallasMosaicGPUTensorProductParams(
    PallasTensorProductParams[
        PallasMosaicGPUTensorProductParamsFwdConfig,
        PallasMosaicGPUTensorProductParamsBwdConfig,
    ]
):
    """Parameters for the Pallas Mosaic GPU tensor product.

    Supports only the `TRAILING_CHANNELS` layout, in `OUTER` and `MAP` modes.
    """

    def __post_init__(self):
        if self.layout != options.Layout.TRAILING_CHANNELS:
            raise NotImplementedError(
                f"Mosaic GPU tensor product supports only "
                f"{options.Layout.TRAILING_CHANNELS}, got {self.layout}"
            )
        if self.mode not in (options.MixingMode.OUTER, options.MixingMode.MAP):
            raise NotImplementedError(
                f"Mosaic GPU tensor product supports only "
                f"{options.MixingMode.OUTER} and {options.MixingMode.MAP} "
                f"modes, got {self.mode}"
            )
        if self.indices.ndim != 2 or self.indices.shape[1] != 3:
            raise ValueError(
                f"indices must be (nnz, 3) in (zi, xi, yi) order, got "
                f"{self.indices.shape}"
            )
        if self.values.shape != self.indices.shape[:1]:
            raise ValueError(
                f"values must be ({self.indices.shape[0]},), one per index row, "
                f"got {self.values.shape}"
            )
        # The kernels do not clamp indices, so an out-of-range one reads SMEM
        # out of bounds and faults the whole CUDA context.
        for axis, (name, space) in enumerate(
            (
                ("z_space", self.z_space),
                ("x_space", self.x_space),
                ("y_space", self.y_space),
            )
        ):
            column = self.indices[:, axis]
            if column.size and (column.min() < 0 or column.max() >= space.dim):
                raise ValueError(
                    f"coefficient indices column {axis} runs "
                    f"[{int(column.min())}, {int(column.max())}], outside "
                    f"{name} {space} of dim {space.dim}"
                )
        if self.fwd_config is None:
            self.fwd_config = PallasMosaicGPUTensorProductParamsFwdConfig()
        if self.bwd_config is None:
            self.bwd_config = PallasMosaicGPUTensorProductParamsBwdConfig()

    @property
    def outer(self) -> bool:
        """Whether `y` carries no channel axis (`OUTER` mode)."""
        return self.mode == options.MixingMode.OUTER

    # Compared and hashed by value, as `jax.custom_vjp`'s `nondiff_argnums`
    # requires.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PallasMosaicGPUTensorProductParams):
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
        # Cast to fixed dtypes so equal arrays hash the same; `+ 0.0` turns
        # `-0.0` into `0.0`.
        return hash(
            (
                self.indices.shape,
                self.indices.astype(np.int64).tobytes(),
                self.values.shape,
                (self.values.astype(np.float64) + 0.0).tobytes(),
                self.layout,
                self.mode,
                self.x_space,
                self.y_space,
                self.z_space,
                self.fwd_config,
                self.bwd_config,
            )
        )
