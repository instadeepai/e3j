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

from typing import Optional

import numpy as np
from scipy.special import factorial


def fact(n: np.ndarray) -> np.ndarray:
    """Factorial n!"""
    return factorial(n)


def bifact(n: np.ndarray) -> np.ndarray:
    """Bifactorial n!!"""
    even = n % 2 == 0
    out = even * bifact_even(n) + (~even) * bifact_odd(n)
    return np.round(out)


def bifact_odd(
    odd: Optional[np.ndarray] = None, n: Optional[np.ndarray] = None
) -> np.ndarray:
    """Bifactorials of the form (2n + 1)!!"""
    if odd is not None:
        n = (odd - 1) / 2
    return fact(2 * n + 1) / bifact_even(n=n)


def bifact_even(
    even: Optional[np.ndarray] = None, n: Optional[np.ndarray] = None
) -> np.ndarray:
    """Bifactorials of the form (2n)!!"""
    if even is not None:
        n = even / 2
    return (2**n) * fact(n)
