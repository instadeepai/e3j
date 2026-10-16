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

import dataclasses
import functools

import flax.linen as nn
import jax.numpy as jnp

from e3j.core.symmetric_contraction import (
    SymmetricContraction as _CoreSymmetricContraction,
)
from e3j.linen.linear_indexwise import LinearIndexwise
from e3j.spaces import O3Space
from e3j.utils import Config
from e3j.utils.config import config
from e3j.utils.options import Layout


class SymmetricContraction(nn.Module):
    """Index-wise symmetric contraction `B(A, i) = W[i] @ (A + A⊗A + ...)`.

    Note
    ----
    This module only owns the weights, the computation is done by
    :class:`e3j.core.SymmetricContraction`.
    """

    source_irreps: str
    correlation: int
    keep_irrep_out: str
    num_species: int
    num_channels: int
    layout: Layout | str
    l_max: int | None = None
    # frozen copy, so the module hashes and unpickling leaves the global alone
    config: Config = dataclasses.field(default_factory=config.state)

    def __post_init__(self):
        object.__setattr__(self, "config", config.snapshot(self.config))
        super().__post_init__()

    @property
    def target(self) -> O3Space:
        return O3Space(self.keep_irrep_out).regroup()

    @functools.cached_property
    def core(self) -> _CoreSymmetricContraction:
        """Flax-free compute core."""
        return _CoreSymmetricContraction(
            self.source_irreps,
            self.correlation,
            self.keep_irrep_out,
            self.num_species,
            self.num_channels,
            self.layout,
            self.l_max,
            self.config,
        )

    @nn.compact
    def __call__(self, features: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
        """See :meth:`e3j.core.SymmetricContraction.__call__`."""
        core = self.core
        # weights from a `LinearIndexwise` submodule keep its init and param paths
        weights = LinearIndexwise(
            core.linear.source_irreps,
            core.linear.target_irreps,
            num_indices=self.num_species,
            num_channels=self.num_channels,
            layout=core.layout,
        ).weights
        return core(features, indices, weights)
