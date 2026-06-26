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
from jax.experimental import sparse

from e3j.spaces.o3 import O3Space
from e3j.utils.config import config
from e3j.utils.options import Layout
from e3j.utils.sparse import sparse_bcoo


@dataclass
class ScalarMixing:
    """Rescale equivariant features by as many scalars.

    This performs the same operation as `scalar * irreps_array` in e3nn.
    """

    source: O3Space
    layout: Layout = field(default_factory=lambda: config().layout)

    def __post_init__(self):
        self.source = O3Space(self.source)

    @property
    def num_irreps(self) -> int:
        return sum(m for m, ir in self.source)

    @property
    def mix_indices(self) -> Array:
        """Return index map from equivariant coordinates to scalars."""
        repeats = []
        for mul, ir in self.source:
            repeats.extend([ir.dim] * mul)
        return np.repeat(
            np.arange(self.num_irreps),
            np.array(repeats),
            total_repeat_length=self.source.dim,
        )

    @property
    def coef(self) -> sparse.BCOO:
        """Return coefficients for an equivalent tensor product operation."""
        n_feats, n_scalars = self.source.dim, self.num_irreps
        values = np.ones(n_feats)
        idx_feats = np.arange(n_feats)
        idx_scalars = self.mix_indices[idx_feats]
        indices = np.stack((idx_feats, idx_scalars, idx_feats), axis=-1)
        shape = (n_feats, n_scalars, n_feats)
        return sparse_bcoo(values, indices, shape)

    def __call__(self, scalars: Array, features: Array) -> Array:
        layout = Layout.parse(self.layout)
        scalar_shape: tuple[int, ...] = ()

        if layout == Layout.TRAILING_CHANNELS:
            axis_k = -1
            scalar_shape = (self.num_irreps, features.shape[axis_k])

        elif layout == Layout.LEADING_CHANNELS:
            axis_k = -2
            scalar_shape = (features.shape[axis_k], self.num_irreps)

        elif layout == Layout.E3NN:
            scalar_shape = (self.num_irreps,)

        if scalars.shape[1:] != scalar_shape:
            scalars = scalars.reshape((-1, *scalar_shape))

        mix_idx = self.mix_indices
        if layout == Layout.LEADING_CHANNELS:
            return scalars[..., mix_idx] * features
        return scalars[:, mix_idx] * features
