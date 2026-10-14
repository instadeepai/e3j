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

import e3nn_jax as e3nn
import flax.linen as nn
import jax
from jax.nn.initializers import Initializer

from e3j.core.linear import LinearBlock
from e3j.core.linear_indexwise import LinearIndexwise as _LinearIndexwiseCore
from e3j.utils.config import config
from e3j.utils.options import Layout, LinearIndexwiseInitialization
from e3j.utils.weights_initialization import _get_weights_std_and_scaling


class LinearIndexwise(nn.Module):
    """E3-equivariant specie-wise linear mixing.

    Weights are learned for each of the `C` independent channels and
    `I` distinct indices, where `C` is inferred during `init`.

    Each linear block acts on a reducible representation of constant
    momentum `l` and parity `p`. This means that for every `(l, p)`
    pair, the module carries a weight array

        w_lp : (I, C, m_out, m_in)

    where `m_in` and `m_out` denote the multiplicities of `(l, p)`
    in the source and target representations respectively.

    Source and target irreps may have mismatching (l, p) blocks:
    target blocks absent from source produce zero outputs, and
    source blocks absent from target are discarded. This is useful
    e.g. for skip-connections from a first scalar-only layer in
    MLIPs, where the source space is ``"Kx0e"`` and the target
    includes higher-order irreps.

    Note
    ----
    This module only owns the weights, the computation is done by
    :class:`e3j.core.LinearIndexwise`.
    """

    source_irreps: str
    target_irreps: str
    num_indices: int
    num_channels: int | None = None
    layout: str | Layout = dataclasses.field(default_factory=lambda: config().layout)
    kernel_init: Initializer | LinearIndexwiseInitialization | str = "FAN_IN"
    rescale_gradients: bool = True

    @property
    def source(self):
        return e3nn.Irreps(self.source_irreps)

    @property
    def target(self):
        return e3nn.Irreps(self.target_irreps)

    @functools.cached_property
    def core(self) -> _LinearIndexwiseCore:
        """Flax-free compute core."""
        # normalise through e3nn: O3Space rejects e.g. stray spaces or lists
        return _LinearIndexwiseCore(
            str(self.source),
            str(self.target),
            self.layout,
            self.num_channels,
        )

    @property
    def blocks(self) -> list[LinearBlock]:
        """`LinearBlock` descriptors acting on irreducibles, in weight order."""
        return self.core.blocks

    def setup(self):
        """Initialize weights from list of `LinearBlock` descriptors."""
        channels = (self.num_channels,) if self.num_channels is not None else ()
        weights = []
        for i, block in enumerate(self.blocks):
            if isinstance(self.kernel_init, Initializer):
                kernel_init, scale = self.kernel_init, 1.0
                assert (
                    not self.rescale_gradients
                ), "`rescale_gradients` requires LinearIndexwiseInitialization option."
            else:
                std, scale = _get_weights_std_and_scaling(
                    m_out=block.shape[0],
                    m_in=block.shape[1],
                    weights_normalization=self.kernel_init,
                    rescale_gradients=self.rescale_gradients,
                    num_indices=self.num_indices,
                )
                kernel_init = nn.initializers.normal(stddev=std)
            wi = self.param(
                f"weight{i}_{block.l}_{block.p}",
                kernel_init,
                (self.num_indices, *channels, *block.shape),
            )
            weights.append(wi * scale)
        self.weights = weights

    def __call__(self, x_feats: jax.Array, x_indices: jax.Array) -> jax.Array:
        """Transform `x_feats` linearly with `x_indices`-dependent weights.

        Args:
            x_feats (jax.Array): `(..., C, source.dim)`-array of equivariant
                features (leading channels) or `(..., source.dim, C)` (trailing
                channels), with any number of leading batch axes.
            x_indices (jax.Array): `(...)`-array of specie indices (positive and
                lower than `I = num_indices`), matching the batch axes.

        Returns:
            jax.Array: Array of shape `(..., C, target.dim)` or
                `(..., target.dim, C)` depending on layout.
        """
        return self.core(x_feats, x_indices, self.weights)
