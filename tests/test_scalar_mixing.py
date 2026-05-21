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

import e3nn_jax as e3nn
import jax
import pytest
from conftest import assert_allclose
from jax import numpy as np
from jax import random

from e3j.core.scalar_mixing import ScalarMixing


class _TestScalarMixing:
    """Base class for ScalarMixing tests."""

    source: str
    num_channels: int = 16
    batch_size: int = 32
    layout: str = "LEADING_CHANNELS"
    _seed: int = 42

    @property
    def key(self) -> jax.Array:
        return random.key(self._seed)

    @pytest.fixture(scope="class")
    def module(self):
        return ScalarMixing(self.source, layout=self.layout)

    @pytest.fixture
    def inputs(self, module):
        irreps = e3nn.Irreps(self.source)
        num_scalars = module.num_irreps
        if self.layout == "LEADING_CHANNELS":
            feat_shape = (self.batch_size, self.num_channels, irreps.dim)
            scalar_shape = (self.batch_size, self.num_channels * num_scalars)
        elif self.layout == "TRAILING_CHANNELS":
            feat_shape = (self.batch_size, irreps.dim, self.num_channels)
            scalar_shape = (self.batch_size, num_scalars * self.num_channels)
        else:
            feat_shape = (self.batch_size, irreps.dim)
            scalar_shape = (self.batch_size, num_scalars)
        scalars = random.normal(self.key, scalar_shape)
        features = random.normal(random.key(1), feat_shape)
        return scalars, features

    def test_linearity(self, module, inputs):
        """Output scales linearly with scalar values."""
        scalars, features = inputs
        alpha = 2.5
        y1 = module(scalars, features)
        y2 = module(alpha * scalars, features)
        assert_allclose(alpha * y1, y2)

    def test_output_shape(self, module, inputs):
        """Output has the same shape as features."""
        scalars, features = inputs
        result = module(scalars, features)
        assert result.shape == features.shape

    def test_equivariance(self, module, inputs):
        """ScalarMixing commutes with SO3 rotations on features."""
        scalars, features = inputs
        irreps = e3nn.Irreps(self.source)
        rotation = e3nn.rand_matrix(self.key)
        D = irreps.D_from_matrix(rotation)

        if self.layout == "LEADING_CHANNELS":
            rotated = features @ D
            y_then_rotate = module(scalars, features) @ D
        elif self.layout == "TRAILING_CHANNELS":
            rotated = np.einsum("nic,ij->njc", features, D)
            y_then_rotate = np.einsum("nic,ij->njc", module(scalars, features), D)
        else:
            rotated = features @ D
            y_then_rotate = module(scalars, features) @ D

        rotate_then_y = module(scalars, rotated)
        assert_allclose(y_then_rotate, rotate_then_y)

    def test_layout_consistency(self, module, inputs):
        """Module commutes with layout cast."""
        scalars, features = inputs
        irreps = e3nn.Irreps(self.source)

        if self.layout == "E3NN":
            gcd = irreps.mul_gcd
            if gcd <= 1:
                pytest.skip("gcd of multiplicities is 1")

            feat_leading = e3nn.IrrepsArray(irreps, features).mul_to_axis()
            reduced_source = str(feat_leading.irreps)

            scalar_irreps = e3nn.Irreps([(m, (0, 1)) for m, _ in irreps])
            scalar_leading = e3nn.IrrepsArray(scalar_irreps, scalars).mul_to_axis()

            other_module = ScalarMixing(reduced_source, layout="LEADING_CHANNELS")

            y1 = e3nn.IrrepsArray(irreps, module(scalars, features)).mul_to_axis().array
            y2 = other_module(
                scalar_leading.array.reshape(scalars.shape[0], -1),
                feat_leading.array,
            )
            assert_allclose(y1, y2)
            return

        other_layout = (
            "TRAILING_CHANNELS"
            if self.layout == "LEADING_CHANNELS"
            else "LEADING_CHANNELS"
        )
        other_module = ScalarMixing(self.source, layout=other_layout)

        num_scalars = module.num_irreps
        if self.layout == "LEADING_CHANNELS":
            scalars_3d = scalars.reshape(-1, self.num_channels, num_scalars)
        else:
            scalars_3d = scalars.reshape(-1, num_scalars, self.num_channels)
        other_scalars = np.swapaxes(scalars_3d, -1, -2).reshape(scalars.shape[0], -1)
        other_features = np.swapaxes(features, -1, -2)

        y1 = np.swapaxes(module(scalars, features), -1, -2)
        y2 = other_module(other_scalars, other_features)

        assert_allclose(y1, y2)


# --- Leading channels (N, C, dim) ---


class TestScalarMixingLeading(_TestScalarMixing):

    source = "2x0e + 3x1o + 2x2e"


class TestScalarMixingLeading_scalars(_TestScalarMixing):

    source = "8x0e"


# --- Trailing channels (N, dim, C) ---


class TestScalarMixingTrailing(_TestScalarMixing):

    source = "2x0e + 3x1o + 2x2e"
    layout = "TRAILING_CHANNELS"


class TestScalarMixingTrailing_scalars(_TestScalarMixing):

    source = "8x0e"
    layout = "TRAILING_CHANNELS"


# --- E3NN layout (N, dim) ---


class TestScalarMixingE3NN(_TestScalarMixing):

    source = "2x0e + 3x1o + 2x2e"
    layout = "E3NN"
