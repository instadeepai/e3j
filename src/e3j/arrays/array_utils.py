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

from typing import Any

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy

from e3j.arrays.array import Array, IndexArray, O3Array, SO3Array
from e3j.spaces import Finite, O3Space, SO3Space, Space
from e3j.utils.options import Layout


def as_array(space: Space | str, x: Any, layout: Layout | None = None) -> Array:
    """Helper constructor for `e3j.Array` instances.

    Currently, string descriptors for the space argument will be cast to `O3Array`.
    Input data can be either `jax.Array`, `e3nn.IrrepsArray`, `numpy.ndarray` or
    list.

    Note that shape checks on the feature axis will be enforced by the `O3Array`
    class itself.
    """

    if isinstance(space, str):
        return as_array(O3Space(space), x, layout)

    Arr = space._array_type

    if isinstance(x, jax.Array):
        return Arr(space, x)

    if isinstance(x, (numpy.ndarray, list)):
        return Arr(space, jnp.asarray(x))

    if isinstance(x, Array):
        if x.space == space:
            return x
        raise ValueError(f"Cannot cast e3j.Array from {x.space} to {space}")

    if isinstance(x, e3nn.IrrepsArray):
        src = O3Space([(mul, (ir.l, ir.p)) for mul, ir in x.irreps])
        if src == space:
            return Arr(space, x.array)
        raise ValueError(f"Cannot cast e3nn.IrrepsArray from {src} to {space}")

    raise ValueError(f"Cannot cast value of type {type(x)} to e3j.Array")
