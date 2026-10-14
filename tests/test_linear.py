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

import jax
import pytest
from jax import numpy as np

from e3j.core.linear import Linear as LinearCore


class TestLinearCoreWeights:
    """The core rejects weight lists that do not match its blocks."""

    core = LinearCore("2x0e + 1o + 2e", "0e + 2x1o + 2e", "E3NN")

    @property
    def weights(self) -> list[np.ndarray]:
        return [np.ones(block.shape) for block in self.core.blocks]

    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("drop", [-1, 0])
    def test_short_weights(self, jit, drop):
        weights = self.weights
        del weights[drop]
        apply = jax.jit(self.core) if jit else self.core
        with pytest.raises(ValueError, match="Expected 3 weights"):
            apply(np.ones((5, self.core.blocks[-1].end)), weights)

    def test_extra_weights(self):
        with pytest.raises(ValueError, match="Expected 3 weights"):
            self.core(np.ones((5, 10)), self.weights + [np.ones((1, 1))])

    def test_misshapen_weight(self):
        weights = self.weights
        weights[1] = weights[1].T
        with pytest.raises(ValueError, match=r"Weight 1 has shape \(1, 2\)"):
            self.core(np.ones((5, 10)), weights)
