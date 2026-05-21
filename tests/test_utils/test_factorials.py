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

from e3j.utils.factorials import bifact, bifact_even, bifact_odd


def test_bifact_even():
    evens = np.array([0, 2, 4, 6, 8])
    expect = np.array([1, 2, 8, 48, 384])
    result = bifact_even(evens)
    assert np.allclose(expect, result)


def test_bifact_odd():
    odds = np.array([1, 3, 5, 7])
    expect = np.array([1, 3, 15, 105])
    result = bifact_odd(odds)
    assert np.allclose(expect, result)


def test_bifact():
    ints = np.arange(9)
    expect = np.array([1, 1, 2, 3, 8, 15, 48, 105, 384])
    result = bifact(ints)
    assert np.allclose(expect, result)
