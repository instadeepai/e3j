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

from typing import Iterator

import jax.numpy as np
from jax import Array

from e3j.spaces.space import Irrep, Space


class Finite(Space):
    """Finite sets of indices.

    The discrete space `Finite(N)` can be used to annotate
    input argument such as atomic numbers or atomic specie
    indices.

    Any group acts trivially on the disjoint union of points.
    """

    def __init__(self, n: int):
        self.cardinal = n

    @property
    def dim(self) -> int:
        return 1

    def action(self, matrix: Array) -> Array:
        return np.array(1.0)

    def __iter__(self) -> Iterator[tuple[int, Irrep]]:
        return iter([])

    def slices(self) -> list[slice]:
        return []

    def in_bounds(self, i: int | np.ndarray) -> bool | np.ndarray:
        return (i >= 0) & (i < self.cardinal)
