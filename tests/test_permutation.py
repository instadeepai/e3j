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
import jax.numpy as np
import pytest
from jax import random

from e3j.core.permutation import Permutation


class _TestPermutationSort:

    rep_in: str
    batch_size: int = 8
    _seed: int = 42

    @property
    def key(self) -> jax.Array:
        return random.key(self._seed)

    @pytest.fixture
    def inputs(self) -> np.ndarray:
        key = random.key(123)
        dim_in = e3nn.Irreps(self.rep_in).dim
        return random.normal(key, (self.batch_size, dim_in))

    @pytest.fixture(scope="class")
    def e3j_module(self):
        return Permutation.sort(self.rep_in)

    @pytest.fixture(scope="class")
    def e3nn_module(self):
        def sort(x: np.ndarray) -> np.ndarray:
            x_in = e3nn.IrrepsArray(self.rep_in, x)
            return x_in.sort().array

        return sort

    @pytest.fixture(scope="class")
    def rotations(self, e3j_module) -> list[np.ndarray]:
        """Prepare output and input rotations."""
        rotation = e3nn.rand_matrix(self.key)
        rotation_in = e3j_module.source.action(rotation)
        rotation_out = e3j_module.target.action(rotation)
        return [rotation_in, rotation_out]

    def assert_zero(self, data, tol=1e-7):
        norm = float(np.sqrt(np.sum(data**2)) / data.size)
        assert norm < tol

    # --- Test functions ---

    def test_e3nn(self, e3j_module, e3nn_module, inputs):
        expect = e3nn_module(inputs)
        result = e3j_module(inputs)
        self.assert_zero(expect - result)

    def test_equivariance(self, inputs, e3j_module, rotations):
        """Check that module commutes with SO3."""
        gfx = e3j_module(inputs) @ rotations[1]
        fgx = e3j_module(inputs @ rotations[0])
        self.assert_zero(gfx - fgx)


class TestPermutationSort_121(_TestPermutationSort):

    rep_in = "2x1o + 3x2e + 3x1o"


class TestPermutation_2143(_TestPermutationSort):

    rep_in = "2x2e + 4x1o + 2x4e + 3x3o"


class TestPermutation_12012(_TestPermutationSort):

    rep_in = "1e + 2e + 0e + 1o + 2e"
