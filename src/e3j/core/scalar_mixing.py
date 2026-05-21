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

from dataclasses import dataclass, field

import jax.numpy as np
from jax import Array

from e3j.spaces.o3 import O3Space
from e3j.utils.config import config
from e3j.utils.options import Layout


@dataclass
class ScalarMixing:
    """Rescale equivariant features by as many scalars.

    This performs the same operation as `scalar * irreps_array` in e3nn.
    """

    source: str | O3Space
    layout: Layout = field(default_factory=lambda: config().layout)

    def __post_init__(self):
        self.source = O3Space(self.source)

    @property
    def num_irreps(self):
        return sum(m for m, ir in self.source)

    def __call__(self, scalars: Array, features: Array) -> Array:
        repeats = []
        for mul, ir in self.source:
            repeats.extend([ir.dim] * mul)

        layout = Layout.parse(self.layout)

        if layout == Layout.TRAILING_CHANNELS:
            axis_lm, axis_k = -2, -1
            scalar_shape = (self.num_irreps, features.shape[axis_k])

        elif layout == Layout.LEADING_CHANNELS:
            axis_lm, axis_k = -1, -2
            scalar_shape = (features.shape[axis_k], self.num_irreps)

        elif layout == Layout.E3NN:
            axis_lm = -1
            scalar_shape = (self.num_irreps,)

        num_irrep = np.repeat(
            np.arange(self.num_irreps),
            np.array(repeats),
            total_repeat_length=features.shape[axis_lm],
        )

        if scalars.shape[1:] != scalar_shape:
            scalars = scalars.reshape((-1, *scalar_shape))

        if layout == Layout.LEADING_CHANNELS:
            return scalars[..., num_irrep] * features
        return scalars[:, num_irrep] * features
