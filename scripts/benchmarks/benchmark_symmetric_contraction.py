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

from functools import cached_property, partial
from typing import Set, Union

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import vmap

import e3j
import e3j.core as core
from e3j.core.bigotimes import Bigotimes
from e3j.core.permutation import Permutation
from e3j.core.power_expansion import PowerExpansion
from e3j.linen.linear_indexwise import LinearIndexwise
from e3j.spaces import O3Space
from e3j.utils.options import Layout

from .e3_benchmark import E3Benchmark, E3FlaxBenchmark, E3Rep

A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]

TP_MODE = "FUSED"

e3j.config(tensor_product="FUSED")


class SymmetricContractionE3nn(nn.Module):
    correlation: int
    keep_irrep_out: Set[e3nn.Irrep]
    num_species: int
    gradient_normalization: Union[str, float] = None
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False

    @property
    def _keep_irrep_out(self) -> e3nn.Irreps:
        """Parse `keep_irrep_out` attribute, possibly a string."""
        out = e3nn.Irreps(self.keep_irrep_out)
        if not all(mul == 1 for mul, _ in out):
            raise ValueError("Expecting mul = 1 for `keep_irrep_out` filter")
        return out

    @nn.compact
    def __call__(
        self, node_feats: e3nn.IrrepsArray, index: jnp.ndarray
    ) -> e3nn.IrrepsArray:
        """Power expansion of node_feats, mapped through index-wise weights.

        This module should return the equivalent of

            B = W[index] @ (A + (A ⊗ A) + ... + A**(⊗ ν))

        where `A = node_feats`, and `W` represents learnable weights acting
        specie-index-wise and momentum-wise on the equivariant powers of
        the node features.
        """
        gradient_normalization = self.gradient_normalization
        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
            # possibly a string now
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]

        def fn(features: e3nn.IrrepsArray, index: jnp.ndarray):
            # - This operation is parallel on the feature dimension (but each feature has its own parameters)
            # This operation is an efficient implementation of
            # vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)
            # up to x power self.correlation
            assert features.ndim == 2  # [num_features, irreps_x.dim]
            assert index.ndim == 0  # int
            out = {}
            for order in range(self.correlation, 0, -1):  # correlation, ..., 1
                if self.off_diagonal:
                    x_ = jnp.roll(features.array, A025582[order - 1])
                else:
                    x_ = features.array
                if self.symmetric_tensor_product_basis:
                    U = e3nn.reduced_symmetric_tensor_product_basis(
                        features.irreps, order, keep_ir=self._keep_irrep_out
                    )
                else:
                    U = e3nn.reduced_tensor_product_basis(
                        [features.irreps] * order, keep_ir=self._keep_irrep_out
                    )
                # U = U / order  # normalization TODO(mario): put back after testing
                # NOTE(mario): The normalization constants (/order and /mul**0.5)
                # has been numerically checked to be correct.
                # TODO(mario) implement norm_p
                # ((w3 x + w2) x + w1) x
                #  \-----------/
                #       out
                for (mul, ir_out), u in zip(U.irreps, U.list):
                    u = u.astype(x_.dtype)
                    # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]
                    w = self.param(
                        f"w{order}_{ir_out}",
                        nn.initializers.normal(
                            stddev=(mul**-0.5) ** (1.0 - gradient_normalization)
                        ),
                        (self.num_species, mul, features.shape[0]),
                        dtype=jnp.float32,
                    )[
                        index
                    ]  # [multiplicity, num_features]
                    w = w * (mul**-0.5) ** gradient_normalization  # normalize weights
                    if ir_out not in out:
                        out[ir_out] = (
                            "special",
                            jnp.einsum("...jki,kc,cj->c...i", u, w, x_),
                        )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                    else:
                        out[ir_out] += jnp.einsum(
                            "...ki,kc->c...i", u, w
                        )  # [num_features, (irreps_x.dim)^order, ir_out.dim]
                # ((w3 x + w2) x + w1) x
                #  \----------------/
                #         out (in the normal case)
                for ir_out in out:
                    if isinstance(out[ir_out], tuple):
                        out[ir_out] = out[ir_out][1]
                        continue  # already done (special case optimization above)
                    out[ir_out] = jnp.einsum(
                        "c...ji,cj->c...i", out[ir_out], x_
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                # ((w3 x + w2) x + w1) x
                #  \-------------------/
                #           out
            # out[irrep_out] : [num_features, ir_out.dim]
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.IrrepsArray.from_list(
                irreps_out,
                [out[ir][:, None, :] for (_, ir) in irreps_out],
                (features.shape[0],),
            )

        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(node_feats.shape[:-2], index.shape)
        node_feats = node_feats.broadcast_to(shape + node_feats.shape[-2:])
        index = jnp.broadcast_to(index, shape)
        fn_mapped = fn
        for _ in range(node_feats.ndim - 2):
            fn_mapped = vmap(fn_mapped)
        return fn_mapped(node_feats, index)


class SymmetricContractionE3jBigotimes(nn.Module):
    correlation: int
    keep_irrep_out: Set[e3nn.Irrep]
    num_species: int
    num_channels: int
    gradient_normalization: Union[str, float] = None
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False
    l_max: int | None = None

    def bigotimes_layers(self, irreps_in, irreps_out, l_max) -> list[Bigotimes]:
        """Direct sum of equivariant power modules, for each exponent."""
        return [
            Bigotimes([irreps_in] * c, irreps_out, l_max=l_max)
            for c in range(2, self.correlation + 1)
        ]

    def permutation_layer(self, irreps_list: list[e3nn.Irreps]) -> Permutation:
        """Groups equivalent irreps before the output linear transform."""
        with jax.ensure_compile_time_eval():
            irreps_in = e3nn.Irreps()
            for rep in irreps_list:
                irreps_in += rep
            perm = Permutation.sort(irreps_in)
        return perm

    @nn.compact
    def __call__(
        self, node_feats: e3nn.IrrepsArray, index: jnp.ndarray
    ) -> e3nn.IrrepsArray:
        """Power expansion of node_feats, contracted with index-wise weights.

        Args:
            node_feats (e3nn.IrrepsArray): array of input node features of
                shape `(-1, num_channels, irreps_in.dim)`
            index (jnp.ndarray): vector of node specie-indices, in the bounds
                `[0, num_species)`.

        Returns:
            e3nn.IrrepsArray: array of output node features of shape
                `(-1, num_channels, irreps_out.dim)`
        """
        irreps_out = e3nn.Irreps(self.keep_irrep_out).regroup()
        irreps_in = node_feats.irreps

        bigotimes_layers = self.bigotimes_layers(irreps_in, irreps_out, self.l_max)
        permutation = self.permutation_layer(
            [irreps_in, *(bigotimes.target for bigotimes in bigotimes_layers)]
        )
        linear_out = LinearIndexwise(
            source=permutation.target.regroup(),
            target=irreps_out,
            num_channels=self.num_channels,
            num_indices=self.num_species,
        )
        # group batch and channels for Bigotimes modules
        batch_shape = jnp.shape(node_feats)[:-1]
        x_feats = node_feats.array.reshape(-1, irreps_in.dim)
        # accumulate equivariant powers
        powers = [x_feats]
        for k, bigotimes in enumerate(bigotimes_layers):
            c = k + 2
            x_c = bigotimes(*([x_feats] * c))
            powers.append(x_c)
        x_powers = jnp.concat(powers, axis=-1)
        # restore batch and channels dimensions
        x_powers = x_powers.reshape(*batch_shape, -1)
        # reorder irreps for linear mixing
        x_sorted = permutation(x_powers)
        node_out = linear_out(x_sorted, index)
        node_out = e3nn.IrrepsArray(irreps_out, node_out)
        return node_out


class SymmetricContractionE3j(nn.Module):
    """
    Contracts the power expansion of E3-features with index-wise weights,

        B(A, i) = W[i] @ (A + (A ⊗ A) + ... + A**(⊗ ν))

    where:

    * `ν` is the correlation order (typically lower than 4),
    * `A : (N, C, D)` is an array of `D`-dimensional E3-features,
    * `i : (N,)` is the vector of specie indices `0 <= i < S`,
    * `W` stores `(S, C)` batches of `(1, M_lp)` matrices `W_lp`
      acting on momentum-l and parity-p blocks to combine the
      multiplicities linearly. See :class:`LinearIndexwise`.

    The shapes given above depend on

    * `N` the batch size,
    * `C` the number of independent channels,
    * `S` the number of species,
    * `D` the dimension of the input representation.

    References
    ----------
    See MACE, eq.10.

    * `MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and
      Accurate Force Fields <https://arxiv.org/abs/2206.07697>`_.
      Batatia, Kovacs, Simm, Ortner & Csanyi, 2022.
    """

    source_irreps: str | O3Space
    correlation: int
    keep_irrep_out: str | O3Space
    num_species: int
    num_channels: int
    layout: str | Layout = "TRAILING_CHANNELS"
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False
    l_max: int | None = None

    @property
    def source(self) -> O3Space:
        return O3Space(self.source_irreps)

    @property
    def target(self) -> e3nn.Irreps:
        return O3Space(self.keep_irrep_out).regroup()

    def expansion_layer(self) -> PowerExpansion:
        """Direct sum of equivariant powers."""
        power_expansion = PowerExpansion(
            source=self.source,
            exponent=self.correlation,
            target_filter=self.keep_irrep_out,
            l_max=self.l_max,
            layout=self.layout,
        )
        for layer in power_expansion.otimes_layers():
            d1, d2 = (src.dim for src in layer.source)
        return power_expansion

    def permutation_layer(self, irreps_list: list[e3nn.Irreps]) -> Permutation:
        """Groups equivalent irreps before the output linear transform."""
        with jax.ensure_compile_time_eval():
            irreps_in = e3nn.Irreps()
            for rep in irreps_list:
                # TODO: IrrepSum / O3Space should have +, +=
                irreps_in += rep._to_e3nn()
            perm = Permutation.sort(O3Space(irreps_in))
        return perm

    def linear_indexwise_layer(self, source: e3nn.Irreps) -> LinearIndexwise:
        """Linear layer aggregating multiplicities with specie-dependent weights."""
        return LinearIndexwise(
            str(source),
            str(self.target),
            num_indices=self.num_species,
            num_channels=self.num_channels,
        )

    @nn.compact
    def __call__(
        self, node_feats: e3nn.IrrepsArray, index: jnp.ndarray
    ) -> e3nn.IrrepsArray:
        """Power expansion of node_feats, contracted with index-wise weights.

        Args:
            node_feats (e3nn.IrrepsArray): array of input node features of
                shape `(-1, num_channels, irreps_in.dim)`
            index (jnp.ndarray): vector of node specie-indices, in the bounds
                `[0, num_species)`.

        Returns:
            e3nn.IrrepsArray: array of output node features of shape
                `(-1, num_channels, target.dim)`
        """
        expansion_layer = self.expansion_layer()
        permutation = self.permutation_layer(expansion_layer.target)
        linear_out = self.linear_indexwise_layer(permutation.target.regroup())

        # input layout conversion
        layout = Layout.parse(self.layout)
        if layout == Layout.TRAILING_CHANNELS:
            x_feats = jnp.matrix_transpose(node_feats.array)
            axis_lm = -2
        else:
            x_feats = node_feats.array
            axis_lm = -1
        # accumulate equivariant powers
        powers = expansion_layer(x_feats)
        x_powers = jnp.concat(powers, axis=axis_lm)
        # output layout conversion
        if layout == Layout.TRAILING_CHANNELS:
            x_powers = jnp.matrix_transpose(x_powers)
        # reorder irreps for linear mixing
        x_sorted = permutation(x_powers)
        node_out = linear_out(x_sorted, index)
        node_out = e3nn.IrrepsArray(str(self.target), node_out)
        return node_out


class SymmetricContractionBenchmark(E3FlaxBenchmark):

    keys = ("e3j", "e3nn")  # "e3j_bigotimes"

    def __init__(
        self,
        source: E3Rep | tuple[E3Rep, ...],
        target: E3Rep,
        num_species: int,
        correlation: int,
        num_channels: int,
        layout: str = "TRAILING_CHANNELS",
        **kwargs,
    ):
        super().__init__(source, target, **kwargs)
        self.num_species = num_species
        self.correlation = correlation
        self.num_channels = num_channels
        self.layout = layout

    def _inputs(self) -> tuple[e3nn.IrrepsArray, jax.Array]:
        feats_shape = (
            self.n_samp,
            self.batch_size,
            self.num_channels,
            self.source[0].dim,
        )
        raw_feats = jax.random.normal(self.rng, feats_shape)
        feats = e3nn.IrrepsArray(self.source[0]._to_e3nn(), raw_feats)
        species_shape = (self.n_samp, self.batch_size)
        species = jax.random.choice(self.rng, self.num_species, species_shape)
        return (feats, species)

    def e3j_inputs(self) -> tuple[e3nn.IrrepsArray, jax.Array]:
        return self._inputs()

    def e3nn_inputs(self) -> tuple[e3nn.IrrepsArray, jax.Array]:
        return self._inputs()

    def e3j_bigotimes_inputs(self) -> tuple[e3nn.IrrepsArray, jax.Array]:
        return self._inputs()

    @cached_property
    def e3j_module(self):
        with e3j.config.use(tensor_product="FUSED"):
            return SymmetricContractionE3j(
                source_irreps=str(self.source[0]),
                correlation=self.correlation,
                keep_irrep_out=str(self.target),
                num_species=self.num_species,
                num_channels=self.num_channels,
                layout=self.layout,
            )

    @cached_property
    def e3j_bigotimes_module(self):
        with e3j.config.use(aggregation="SCATTER_1"):
            return SymmetricContractionE3jBigotimes(
                correlation=self.correlation,
                keep_irrep_out=self.target._to_e3nn(),
                num_species=self.num_species,
                num_channels=self.num_channels,
            )

    @cached_property
    def e3j_bigotimes_params(self):
        inputs = self.e3j_bigotimes_inputs()
        return self.e3j_bigotimes_module.init(self.rng, *(x[0] for x in inputs))

    @cached_property
    def e3j_bigotimes(self):
        return partial(
            self.e3j_bigotimes_module.apply,
            self.e3j_bigotimes_params,
        )

    @cached_property
    def e3nn_module(self):
        return SymmetricContractionE3nn(
            correlation=self.correlation,
            keep_irrep_out=self.target._to_e3nn(),
            num_species=self.num_species,
        )

    def __str__(self):
        return f"SymmetricContraction({self.source[0]} ** {self.correlation} → {self.target})"

    @property
    def dim_in(self):
        return self.source[0].dim * self.num_channels + 1

    @property
    def dim_out(self):
        return self.e3j_module.target.dim * self.num_channels

    @property
    def nnz_and_dim_out(self):
        irreps_in = self.source[0]
        irreps_out = self.target
        nnz_and_dim_out_dict = {}
        for k, c in enumerate(range(2, self.correlation + 1)):
            bigotimes = core.Bigotimes([irreps_in] * c, irreps_out)
            nnz_and_dim_out_dict[f"Bigotimes_{k}"] = {
                "dim_out": bigotimes.shape[0],
                "nnz": bigotimes.nnz,
                "nnz_ratio": bigotimes.nnz_ratio,
            }
        return nnz_and_dim_out_dict


class SymmetricContractionGradBenchmark(E3Benchmark):

    @property
    def dim_in(self):
        return self.primitive.dim_in

    @property
    def dim_out(self):
        return self.primitive.dim_out

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primitive = TensorProductBenchmark(*args, **kwargs)

    @cached_property
    def e3j(self):
        primitive = self.primitive.e3j
        return jax.grad(
            lambda x, y: np.sum(primitive(x, y)),
            argnums=(0, 1),
        )

    @cached_property
    def e3nn(self):
        primitive = self.primitive.e3nn
        return jax.grad(
            lambda x, y: np.sum(primitive(x, y).array),
            argnums=(0, 1),
        )

    def __str__(self):
        return "Bwd " + str(self.primitive)


if __name__ == "__main__":

    benchmark = SymmetricContractionBenchmark(
        ("0e + 1o + 2e + 3o"),
        "0e + 1o",
        correlation=3,
        num_channels=64,
        num_species=4,
        batch_size=2048,
        n_it=50,
    )
    benchmark.run()
    # benchmark.trace("tmp/symmetric-contraction")
