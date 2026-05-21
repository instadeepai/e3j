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

from e3j.spaces import O3Space


def irrep_range(l_max: int, pseudotensors: bool = False) -> O3Space:
    """Return direct sum of irreps with angular momentum l <= l_max."""
    irreps = [(1, (l, (-1) ** l)) for l in range(l_max + 1)]
    if pseudotensors:
        irreps += [(1, (l, -((-1) ** l))) for l in range(l_max + 1)]
    return O3Space(irreps)
