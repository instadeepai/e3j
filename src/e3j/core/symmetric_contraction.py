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

import jax.numpy as jnp
import numpy as np

from e3j import utils
from e3j.core.filter import Filter
from e3j.core.linear import check_weights
from e3j.core.linear_indexwise import LinearBlock, LinearIndexwise
from e3j.core.permutation import Permutation
from e3j.core.power_expansion import PowerExpansion
from e3j.pallas_ops.symmetric_contraction import (
    PallasMosaicGPUSymmetricContractionParams,
    PallasMosaicTPUSymmetricContractionParams,
    symmetric_contraction_pallas_mosaic_gpu,
    symmetric_contraction_pallas_mosaic_tpu,
)
from e3j.pallas_ops.symmetric_contraction.common.params import BlockPlan
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

    Species indices outside `[0, S)` are undefined behaviour: they are not
    checked, and the result may differ between implementations.

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

    @functools.cached_property
    def mosaic_tpu_params(self) -> PallasMosaicTPUSymmetricContractionParams:
        """Static metadata of the fused Mosaic TPU kernel."""
        return self._kernel_params(PallasMosaicTPUSymmetricContractionParams)

    @functools.cached_property
    def mosaic_gpu_params(self) -> PallasMosaicGPUSymmetricContractionParams:
        """Static metadata of the fused Mosaic GPU kernel."""
        return self._kernel_params(PallasMosaicGPUSymmetricContractionParams)

    def _block_plans(self) -> list[BlockPlan]:
        """Static assembly plan for every non-degenerate final linear block.

        Resolves, once at trace time, exactly the block-copy
        :class:`~e3j.core.Permutation` performs: for each input coordinate of
        each final `(l, p)` block of the index-wise linear, which power
        computed it, and at which of that power's own (unfiltered)
        coordinates. `Permutation.sigma` is followed one coordinate at a time,
        so nothing is assumed about how its spans line up with the powers:
        same-`(l, p)` blocks of consecutive powers may sit side by side (with
        `keep_irrep_out="0e"`, every power's do).
        """
        filters: list[Filter] = self.expansion.filter_layers
        sigma = self.permutation.sigma
        # Where each power's filtered coordinates start in the concatenation
        # `sigma` indexes into.
        offsets = np.cumsum([0] + [int(f.target.dim) for f in filters])

        plans = []
        out_offset = 0
        for block in self.linear_blocks:
            m_out, m_in = block.shape
            # Where the block lands in the output, degenerate blocks included.
            block_offset = out_offset
            out_offset += m_out * (2 * block.l + 1)
            if m_in == 0 or m_out == 0:
                continue
            raw = sigma[block.begin : block.end]
            # `side="right"` steps over empty powers, which start and end at
            # the same offset.
            stages = np.searchsorted(offsets, raw, side="right") - 1
            # `Filter.indices` maps a filtered position back to its raw
            # (unfiltered) coordinate in that power's hidden space.
            input_map = tuple(
                (int(s), int(filters[s].indices[r - offsets[s]]))
                for s, r in zip(stages, raw)
            )
            plans.append(
                BlockPlan(
                    l=block.l,
                    p=block.p,
                    m_out=m_out,
                    input_map=input_map,
                    offset=block_offset,
                )
            )
        return plans

    def _kernel_params(self, cls):
        """Build fused kernel parameters from this contraction: `cls`, a
        :class:`~e3j.pallas_ops.symmetric_contraction.common.params.PallasSymmetricContractionStructure`
        with its kernel's default config.

        The Clebsch-Gordan structure of every power transition, and the final
        block assembly plan, are unrolled at trace time from their NumPy
        data -- mirroring :meth:`e3j.core.TensorProduct._mgpu_params`.
        """
        layers = self.expansion.otimes_layers
        stage_irreps = tuple(
            tuple((int(mul), int(ir.l), int(ir.p)) for mul, ir in space)
            for space in (self.source, *(layer.target for layer in layers))
        )
        stage_indices = tuple(layer.coef.indices for layer in layers)
        stage_values = tuple(layer.coef.data for layer in layers)
        return cls(
            stage_irreps=stage_irreps,
            stage_indices=stage_indices,
            stage_values=stage_values,
            blocks=tuple(self._block_plans()),
            num_species=self.num_species,
            out_dim=self.target.dim,
        )

    def _kernel_weights(self, weight: list[jnp.ndarray]) -> list[jnp.ndarray]:
        """Weights of the non-degenerate blocks, the fused kernels' (`_block_plans`)."""
        return [w for w, b in zip(weight, self.linear_blocks) if all(b.shape)]

    def _unfused_eval(
        self,
        features: jnp.ndarray,
        indices: jnp.ndarray,
        weight: list[jnp.ndarray],
    ) -> jnp.ndarray:
        """Plain JAX implementation: the powers, permuted, then the linear."""
        powers = self.expansion(features)
        x_powers = jnp.concat(powers, axis=self.layout.axis_lm)
        return self.linear(self.permutation(x_powers), indices, weight)

    def _fused_mosaic_gpu_eval(
        self,
        features: jnp.ndarray,
        indices: jnp.ndarray,
        weight: list[jnp.ndarray],
    ) -> jnp.ndarray:
        """Pallas Mosaic GPU implementation: one kernel per pass.

        Evaluates every power's Clebsch-Gordan contraction and the final
        index-wise linear mix in a single kernel launch -- no power, and no
        permuted concatenation of them, ever reaches GMEM. Differentiable in
        `features` and `weight`, through fused backward kernels (see
        `pallas_ops.symmetric_contraction.mosaic_gpu.primitives`), twice
        too (e.g. to train on forces): where no fused kernel takes the
        second derivative, through :meth:`_unfused_eval`.
        """
        indices = jnp.asarray(indices)

        def reference(features, indices, kept_weights):
            # zeros, not the empty original: that may be an outer tracer
            given = iter(kept_weights)
            full = [
                next(given) if all(b.shape) else jnp.zeros(w.shape, w.dtype)
                for w, b in zip(weight, self.linear_blocks)
            ]
            return self._unfused_eval(features, indices, full)

        return symmetric_contraction_pallas_mosaic_gpu(
            features,
            indices,
            self._kernel_weights(weight),
            self.mosaic_gpu_params,
            reference=reference,
        )

    def _fused_mosaic_tpu_eval(
        self,
        features: jnp.ndarray,
        indices: jnp.ndarray,
        weight: list[jnp.ndarray],
    ) -> jnp.ndarray:
        """Mosaic TPU implementation, on flattened `(N, dim, C)` features."""
        return symmetric_contraction_pallas_mosaic_tpu(
            features,
            jnp.asarray(indices),
            self._kernel_weights(weight),
            self.mosaic_tpu_params,
        )

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
            indices (jnp.ndarray): `(...)`-array of species indices in
                `[0, S)`; out-of-range indices are undefined behaviour.
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
        fused = (
            options.SymmetricContraction.FUSED_MOSAIC_TPU,
            options.SymmetricContraction.FUSED_MOSAIC_GPU,
        )
        if (
            self.config.symmetric_contraction in fused
            and self.layout != Layout.TRAILING_CHANNELS
        ):
            raise NotImplementedError(
                f"{self.config.symmetric_contraction.name} SymmetricContraction "
                "only supports TRAILING_CHANNELS layout."
            )
        match self.config.symmetric_contraction:
            case options.SymmetricContraction.UNFUSED:
                y = self._unfused_eval(features, indices, weight)
            case options.SymmetricContraction.FUSED_MOSAIC_TPU:
                y = self._fused_mosaic_tpu_eval(features, indices, weight)
            case options.SymmetricContraction.FUSED_MOSAIC_GPU:
                y = self._fused_mosaic_gpu_eval(features, indices, weight)
            case _:
                raise RuntimeError(
                    f"Unknown implementation {self.config.symmetric_contraction}"
                )
        return y.reshape(*batch, *y.shape[1:])
