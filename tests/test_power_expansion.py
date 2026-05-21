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
import jax.numpy as np
import jax.random as random
import pytest

from e3j.core.power_expansion import PowerExpansion
from e3j.spaces import O3Space
from e3j.utils.irreps import irrep_range


class _TestPowerExpansion:

    source: str
    exponent: int
    keep_ir_out: str | None
    l_max: int | None
    layout: str = "LEADING_CHANNELS"

    batch_size: int = 32
    _seed: int = 321

    @property
    def key(self):
        return random.key(self._seed)

    @pytest.fixture(scope="class")
    def module(self):
        return PowerExpansion(
            source=self.source,
            exponent=self.exponent,
            target_filter=self.keep_ir_out,
            l_max=self.l_max,
            layout=self.layout,
        )

    @pytest.fixture(scope="class")
    def inputs(self):
        src = e3nn.Irreps(self.source)
        x = random.normal(self.key, (self.batch_size, src.dim))
        return (x,)

    def test_hidden(self, module):
        assert len(module.hidden) == self.exponent

    def test_target(self, module):
        assert len(module.target) == self.exponent
        if self.keep_ir_out is not None:
            irreps = set(rep for _, rep in O3Space(self.keep_ir_out))
        else:
            irreps = set(rep for _, rep in irrep_range(self.l_max, True))
        # Check that all targets are within output filter
        for tgt in module.target:
            for _, ir in tgt:
                assert ir in irreps

    # TODO: test equivariance and numerical outputs

    def test_output(self, module, inputs):
        out = module(*inputs)
        target = module.target
        for x_nu, src_nu in zip(out, target):
            assert x_nu.shape[-1] == src_nu.dim
        # Powers 1, ..., nu (inclusive)
        assert len(out) == self.exponent


class TestPowerExpansionSPD2SPD(_TestPowerExpansion):

    source = "0e + 1o + 2e"
    exponent = 2
    keep_ir_out = "0e + 1o + 2e"
    l_max = None


@pytest.mark.e3j_ops
class TestPowerExpansionTrailing(_TestPowerExpansion):

    source = "0e + 1o + 2e"
    exponent = 2
    keep_ir_out = "0e + 1o + 2e"
    l_max = None
    layout = "TRAILING_CHANNELS"


class TestPowerExpansionSP2SPLmax(_TestPowerExpansion):

    source = "0e + 1o"
    exponent = 2
    keep_ir_out = None
    l_max = 2
