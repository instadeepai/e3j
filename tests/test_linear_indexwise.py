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

from e3j.core.linear_indexwise import LinearIndexwise as LinearIndexwiseCore
from e3j.utils.options import Layout


class TestLinearIndexwiseCoreWeights:
    """The core rejects weight lists that do not match its blocks."""

    num_indices = 3
    num_channels = 4

    @pytest.fixture(params=["LEADING_CHANNELS", "TRAILING_CHANNELS"])
    def core(self, request) -> LinearIndexwiseCore:
        return LinearIndexwiseCore(
            "2x0e + 1o + 2e", "0e + 2x1o + 2e", request.param, self.num_channels
        )

    def weights(self, core) -> list[np.ndarray]:
        shape = (self.num_indices, self.num_channels)
        return [np.ones((*shape, *block.shape)) for block in core.blocks]

    def inputs(self, core):
        dim = core.blocks[-1].end
        feats = np.ones((5, self.num_channels, dim))
        if core.layout == Layout.TRAILING_CHANNELS:
            feats = np.swapaxes(feats, -1, -2)
        return feats, np.zeros(5, dtype=int)

    @pytest.mark.parametrize("jit", [False, True])
    @pytest.mark.parametrize("drop", [-1, 0])
    def test_short_weights(self, core, jit, drop):
        weights = self.weights(core)
        del weights[drop]
        apply = jax.jit(core) if jit else core
        with pytest.raises(ValueError, match="Expected 3 weights"):
            apply(*self.inputs(core), weights)

    def test_extra_weights(self, core):
        weights = self.weights(core) + [np.ones((self.num_indices, 1, 1, 1))]
        with pytest.raises(ValueError, match="Expected 3 weights"):
            core(*self.inputs(core), weights)

    @pytest.mark.parametrize(
        "shape", [(3, 4, 1, 2), (3, 2, 2, 1), (3, 2, 1), (3, 4, 2, 1, 1)]
    )
    def test_misshapen_weight(self, core, shape):
        weights = self.weights(core)
        weights[1] = np.ones(shape)
        with pytest.raises(ValueError, match=r"Weight 1 has shape .* \(_, 4, 2, 1\)"):
            core(*self.inputs(core), weights)
