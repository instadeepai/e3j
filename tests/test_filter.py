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

import jax.numpy as np
import jax.random as random
import pytest


class _TestFilter:

    source: str
    target: str
    batch_size: int = 32
    _seed: int = 321

    @property
    def key(self):
        return random.key(self._seed)

    @pytest.fixture(scope="class")
    def module(self):
        return Filter(source, target)

    @pytest.fixture(scope="class")
    def inputs(self, module) -> tuple[np.ndarray, ...]:
        nb = self.batch_size
        return (random.normal(key, (nb, module.source.dim)),)
