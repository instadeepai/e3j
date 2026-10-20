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

from functools import partial
from typing import Iterable, TypeAlias

import jax
import jax.numpy as jnp

Axis: TypeAlias = int | Iterable[int]


def safe_sqrt(x: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    """Safe sqrt."""
    return jnp.sqrt(x + eps)


def safe_abs(x: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    """Safe absolute value."""
    return safe_sqrt(x**2)


@partial(jax.jit, static_argnames=["axis"])
def safe_norm(
    x: jnp.ndarray,
    order: int = 2,
    eps: float = 1e-7,
    axis: Axis | None = None,
) -> jnp.ndarray:
    """Safe norm."""
    if order == 2:
        return safe_sqrt(jnp.sum(x**2, axis=axis), eps)
    else:
        abs_p = safe_abs(x, eps) ** order
        return jnp.sum(abs_p, axis=axis) ** (1 / order)


def safe_cdist(x: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-7) -> jnp.ndarray:
    """Safe pairwise distances.

    See https://github.com/google/jax/discussions/11841.
    """
    return safe_norm(x[:, None, :] - y[None, :, :], 2, eps, axis=-1)
