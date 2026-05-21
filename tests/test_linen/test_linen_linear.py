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

from e3j.linen.linear import Linear


class _TestLinear:
    """Base class for linear module tests."""

    out: str
    in1: str
    batch_size: int = 32
    _seed: int = 42
    channels: tuple[int, int] = (1, 1)

    @property
    def key(self) -> jax.Array:
        return random.key(self._seed)

    layout: str = "E3NN"

    @pytest.fixture
    def inputs(self):
        """Arrange inputs for Linear modules."""
        rep_in = e3nn.Irreps(self.in1)
        nb, dim_in, c_in = self.batch_size, rep_in.dim, self.channels[0]
        if self.layout == "E3NN":
            return random.normal(self.key, (nb, dim_in))
        elif self.layout == "LEADING_CHANNELS":
            return random.normal(self.key, (nb, c_in, dim_in))
        elif self.layout == "TRAILING_CHANNELS":
            return random.normal(self.key, (nb, dim_in, c_in))

    @pytest.fixture(scope="class")
    def e3j_module(self):
        return Linear(
            self.in1,
            self.out,
            layout=self.layout,
            channels=self.channels,
        )

    @pytest.fixture
    def params(self, e3j_module, inputs):
        """Arrange parameters for Linear modules."""
        return e3j_module.init(self.key, inputs)

    @pytest.fixture(scope="class")
    def rotations(self, e3j_module) -> list[np.ndarray]:
        """Prepare output and input rotations."""
        in_irreps = e3nn.Irreps(str(e3j_module.source))
        out_irreps = e3nn.Irreps(str(e3j_module.target))
        rotation = e3nn.rand_matrix(self.key)
        rotation_in = in_irreps.D_from_matrix(rotation)
        rotation_out = out_irreps.D_from_matrix(rotation)
        return [rotation_in, rotation_out]

    def rotate(self, g, x):
        if self.layout == "E3NN":
            return x @ g
        elif self.layout == "LEADING_CHANNELS":
            return np.einsum("...i,ij->...j", x, g)
        elif self.layout == "TRAILING_CHANNELS":
            return np.einsum("nic,ij->njc", x, g)

    def assert_zero(self, data, tol: float = 1e-7):
        norm = float(np.sqrt(np.sum(data**2))) / data.size
        assert norm < tol

    # --- Test functions ---

    def test_equivariance(self, params, inputs, e3j_module, rotations):
        """Check that module commutes with SO3."""
        gfx = self.rotate(
            rotations[1],
            e3j_module.apply(params, inputs),
        )
        fgx = e3j_module.apply(
            params,
            self.rotate(rotations[0], inputs),
        )
        assert_allclose(gfx, fgx, atol=5e-6, rtol=5e-6, debug=2)

    def test_layout_consistency(self, params, inputs, e3j_module):
        """Module commutes with LEADING <-> TRAILING layout cast."""
        if self.layout == "E3NN":
            pytest.skip("no channel axis to swap")

        other_layout = (
            "TRAILING_CHANNELS"
            if self.layout == "LEADING_CHANNELS"
            else "LEADING_CHANNELS"
        )
        other_module = Linear(
            self.in1,
            self.out,
            channels=self.channels,
            layout=other_layout,
            kernel_init=e3j_module.kernel_init,
        )

        y1 = np.swapaxes(e3j_module.apply(params, inputs), -1, -2)
        y2 = other_module.apply(params, np.swapaxes(inputs, -1, -2))

        assert_allclose(y1, y2)


class TestLinear_1(_TestLinear):

    out = "2x1e"
    in1 = "2x1e"
    batch_size = 32


class TestLinear_012(_TestLinear):

    out = "4x0e + 4x1o + 4x2e"
    in1 = "2x0e + 3x1o + 4x2e"
    batch_size = 32


class TestLinear_012_1234(_TestLinear):

    out = "4x0e + 4x1o + 4x2e"
    in1 = "3x1o + 3x2e + 3x3o + 3x4e"
    batch_size = 32


@pytest.mark.xfail
class TestLinear_224_24(_TestLinear):

    out = "0e + 2e + 2e + 4e"
    in1 = "2x2e + 4e"
    batch_size = 32


@pytest.mark.xfail
class TestLinear_21_12(_TestLinear):

    out = "8x2e + 8x1o"
    in1 = "8x1o + 8x2e"
    batch_size = 32


class TestLinearLeading(_TestLinear):
    out = "0e + 1o + 2e + 3o"
    in1 = "1o + 2e + 3o"
    batch_size = 10
    layout = "LEADING_CHANNELS"
    channels = (64, 32)


class TestLinearTrailing(_TestLinear):
    out = "0e + 1o + 2e + 3o"
    in1 = "0e + 1o + 2e + 3o"
    batch_size = 10
    layout = "TRAILING_CHANNELS"
    channels = (64, 32)
