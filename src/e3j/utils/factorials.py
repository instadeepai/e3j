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

import jax
import jax.numpy as np
from jax.scipy.special import factorial as jax_factorial

Array = jax.Array


def fact(n: Array) -> Array:
    """Factorial n!"""
    return jax_factorial(n)


def bifact(n: Array) -> Array:
    """Bifactorial n!!"""
    even = n % 2 == 0
    out = even * bifact_even(n) + (~even) * bifact_odd(n)
    return np.round(out)


def bifact_odd(odd: Optional[Array] = None, n: Optional[Array] = None) -> Array:
    """Bifactorials of the form (2n + 1)!!"""
    if odd is not None:
        n = (odd - 1) / 2
    return fact(2 * n + 1) / bifact_even(n=n)


def bifact_even(even: Optional[Array] = None, n: Optional[Array] = None) -> Array:
    """Bifactorials of the form (2n)!!"""
    if even is not None:
        n = even / 2
    return (2**n) * fact(n)
