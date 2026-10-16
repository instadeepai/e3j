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

import functools

import jax
import jax.numpy as jnp

from e3j import utils
from e3j.core.linear import check_weights
from e3j.core.linear_indexwise import LinearBlock, LinearIndexwise
from e3j.core.permutation import Permutation
from e3j.core.power_expansion import PowerExpansion
from e3j.spaces import O3Space
from e3j.utils import options
from e3j.utils.options import Layout


class SymmetricContraction:
    """Contracts the power expansion of E3-features with index-wise weights,

        B(A, i) = W[i] @ (A + (A ⊗ A) + ... + A**(⊗ ν))

    where:

    * `ν` is the correlation order (typically lower than 4),
    * `A : (N, C, D)` is an array of `D`-dimensional E3-features,
    * `i : (N,)` is the vector of species indices `0 <= i < S`,
    * `W` stores one `(S, C, m_out, m_in)` weight per block of
      :attr:`linear_blocks`, passed explicitly to :meth:`__call__`.

    References
    ----------
    See MACE, eq.10.

    * `MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and
      Accurate Force Fields <https://arxiv.org/abs/2206.07697>`_.
      Batatia, Kovacs, Simm, Ortner & Csanyi, 2022.
    """

    def __init__(
        self,
        source_irreps: str,
        correlation: int,
        keep_irrep_out: str,
        num_species: int,
        num_channels: int,
        layout: str | Layout,
        l_max: int | None = None,
        config: utils.Config | None = None,
    ):
        self.source_irreps = source_irreps
        self.correlation = correlation
        self.keep_irrep_out = keep_irrep_out
        self.num_species = num_species
        self.num_channels = num_channels
        self.layout = Layout.parse(layout)
        self.l_max = l_max
        self.config = utils.config.snapshot(config)
        if self.layout == Layout.E3NN:
            raise NotImplementedError(
                "SymmetricContraction does not support the E3NN layout; "
                "use LEADING_CHANNELS or TRAILING_CHANNELS."
            )

    @property
    def source(self) -> O3Space:
        return O3Space(self.source_irreps)

    @property
    def target(self) -> O3Space:
        # regrouped: the index-wise linear needs sorted, merged (l, p) blocks
        return O3Space(self.keep_irrep_out).regroup()

    @functools.cached_property
    def expansion(self) -> PowerExpansion:
        """Direct sum of equivariant powers."""
        return PowerExpansion(
            source=self.source_irreps,
            exponent=self.correlation,
            target_filter=self.keep_irrep_out,
            l_max=self.l_max,
            layout=self.layout,
            config=self.config,
        )

    @functools.cached_property
    def permutation(self) -> Permutation:
        """Groups equivalent irreps before the output linear transform."""
        with jax.ensure_compile_time_eval():
            combined = " + ".join(str(s) for s in self.expansion.target if s.dim)
            if not combined:
                raise ValueError(
                    f"No power of {self.source_irreps!r} up to correlation "
                    f"{self.correlation} holds any of {self.keep_irrep_out!r}."
                )
            return Permutation.sort(combined, axis=self.layout.axis_lm)

    @functools.cached_property
    def linear(self) -> LinearIndexwise:
        """Index-wise linear mixing the sorted powers."""
        return LinearIndexwise(
            str(self.permutation.target.regroup()),
            str(self.target),
            self.layout,
            self.num_channels,
        )

    @property
    def linear_blocks(self) -> list[LinearBlock]:
        """Per-(l, p) blocks of the index-wise linear, in weight order."""
        return self.linear.blocks

    def __call__(
        self,
        features: jnp.ndarray,
        indices: jnp.ndarray,
        weight: list[jnp.ndarray],
    ) -> jnp.ndarray:
        """Power expansion of `features`, contracted with index-wise weights.

        Args:
            features (jnp.ndarray): `(..., C, dim)` or `(..., dim, C)` features
                depending on :attr:`layout`.
            indices (jnp.ndarray): `(...)`-array of species indices.
            weight (list[jnp.ndarray]): one `(S, C, m_out, m_in)` array per
                block of :attr:`linear_blocks`.

        Returns:
            jnp.ndarray: `target` features, in :attr:`layout`.
        """
        shape = (self.num_species, self.num_channels)
        check_weights(weight, [(*shape, *b.shape) for b in self.linear_blocks])
        # flatten batch axes: the fused tensor products take exactly one
        batch = features.shape[:-2]
        if indices.shape != batch:
            raise ValueError(
                f"Indices of shape {indices.shape} do not match the batch axes "
                f"{batch} of features of shape {features.shape}."
            )
        features = features.reshape(-1, *features.shape[-2:])
        indices = indices.reshape(-1)
        match self.config.symmetric_contraction:
            case options.SymmetricContraction.UNFUSED:
                powers = self.expansion(features)
                x_powers = jnp.concat(powers, axis=self.layout.axis_lm)
                y = self.linear(self.permutation(x_powers), indices, weight)
            case _:
                raise RuntimeError(
                    f"Unknown implementation {self.config.symmetric_contraction}"
                )
        return y.reshape(*batch, *y.shape[1:])
