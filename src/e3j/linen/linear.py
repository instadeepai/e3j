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
import jax
from jax.nn.initializers import Initializer

from e3j.core.linear import Linear as _LinearCore
from e3j.core.linear import LinearBlock
from e3j.spaces.o3 import O3Space
from e3j.utils.config import config
from e3j.utils.options import Layout, LinearInitialization
from e3j.utils.weights_initialization import _get_weights_std_and_scaling

__all__ = ["Linear", "LinearBlock"]


class Linear(nn.Module):
    """E3-equivariant linear channel mixing.

    The `Linear` module mixes equivariant channels with same angular momentum `l` by:

    * iterating over input slices indexed by `l`,
    * reshaping each slice `x_l` from shape `(-1, (2l+1) *m_in)`
      to shape `(-1, m_in)`,
    * linearly transforming slices as `y_l = x_l @ weights_l`,
    * reshaping output slices `y_l` from shape `(-1, m_out)`
      to shape `(-1, (2l+1) * m_out)`,
    * concatenating output slices on axis -1.

    Iteration over slices is not parallelised by `jax.tree.map`.

    Note
    ----
    This module only owns the weights, the computation is done by
    :class:`e3j.core.Linear`.
    """

    source_irreps: str
    target_irreps: str
    channels: tuple[int, int] = (1, 1)
    layout: str | Layout = dataclasses.field(default_factory=lambda: config().layout)
    kernel_init: Initializer | LinearInitialization | str = "FAN_IN"
    rescale_gradients: bool = True

    @property
    def source(self):
        return O3Space(self.source_irreps)

    @property
    def target(self):
        return O3Space(self.target_irreps)

    @functools.cached_property
    def core(self) -> _LinearCore:
        """Flax-free compute core."""
        return _LinearCore(
            self.source_irreps,
            self.target_irreps,
            self.layout,
            self.channels,
        )

    @property
    def blocks(self) -> list[LinearBlock]:
        """`LinearBlock` descriptors acting on irreducibles, in weight order."""
        return self.core.blocks

    def slice_weights(self) -> list[jax.Array]:
        """Return arrays of weights acting on all degrees."""
        weights = []
        for i, block in enumerate(self.blocks):
            if isinstance(self.kernel_init, Initializer):
                kernel_init, scale = self.kernel_init, 1.0
                assert (
                    not self.rescale_gradients
                ), "`rescale_gradients` requires LinearInitialization option."
            else:
                std, scale = _get_weights_std_and_scaling(
                    m_out=block.shape[0],
                    m_in=block.shape[1],
                    weights_normalization=self.kernel_init,
                    rescale_gradients=self.rescale_gradients,
                )
                kernel_init = nn.initializers.normal(std)

            wi = self.param(f"weight{i}_{block.l}_{block.p}", kernel_init, block.shape)
            weights.append(wi * scale)

        return weights

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Transform equivariant tensors linearly."""
        weights = self.slice_weights()
        return self.core(x, weights)
