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

from __future__ import annotations

from typing import Iterable, Literal, Optional

import e3nn_jax as e3nn
import jax.numpy as jnp
import numpy as np

import e3j.utils as utils
from e3j.spaces import O3Space
from e3j.utils.cache import cache
from e3j.utils.irreps import irrep_range
from e3j.utils.sparse import SparseCOO, SparseMixin, sparse_coo

from .permutation import Permutation
from .tensor_product import TensorProduct


class Bigotimes(SparseMixin):
    """Multilinear tensor products."""

    @cache
    def coef(self):
        return self.clebsch_gordan(self.target, *self.source, self.l_max)

    @classmethod
    def clebsch_gordan(
        cls,
        target: O3Space | str,
        *sources: Iterable[str],
        l_max: int | None = None,
    ) -> Bigotimes:
        """Compute generalized CG coefficients by successive compositions.

        Returns an (N+1)-dimensional array whose dimensions appear in the
        same order as the arguments.

        N.B. This factory is called by `__init__` when `coef = None`.
        """
        ins = sources
        # filter hidden outputs by `l <= l_max`, but last outputs by `target`
        out_filter_hidden = None if l_max is None else irrep_range(l_max)
        out_filter_last = out_filter_hidden if target is None else O3Space(target)
        # multiply first 2 arguments
        upstream = cls(ins[:2], out_filter_hidden, None, sort=False)
        # multiply with remaining arguments, left to right
        for k in range(2, len(ins)):
            out_filter = out_filter_hidden if k < len(ins) - 1 else out_filter_last
            downstream = cls((upstream.target, ins[k]), out_filter, None, sort=False)
            # accumulate composed coefficients
            upstream = downstream.pullback(upstream)
        return upstream

    def __init__(
        self,
        source: list[O3Space | str],
        target: O3Space | str,
        coef: SparseCOO | None = None,
        l_max: int | None = None,
        sort: bool = True,
        config: utils.Config | None = None,
    ):
        """
        Compute and stack generalized CG coefficients, or use given ones.
        """
        self.arity = len(source)
        self.source = tuple(O3Space(in_k) for in_k in source)
        self.l_max = l_max

        if coef is not None:
            self.target = O3Space(target)
            self.coef = coef

        elif len(source) == 2:
            # Fallback to TensorProduct in binary case
            target_filter = (
                irrep_range(l_max, pseudotensors=True)
                if target is None and l_max is not None
                else target
            )
            tp = TensorProduct(source, target_filter, sort=False, config=config)
            self.target = tp.target
            self.coef = tp.coef

        else:
            # Recursively compose coefficients in mulilinear case
            other = self.clebsch_gordan(target, *source, l_max=l_max)
            self.target = other.target
            self.coef = other.coef

        if sort:
            other = self.sort()
            self.coef = other.coef
            self.target = other.target

        self.config = utils.config.snapshot(config)
        self.aggregation_method = self.config.aggregation

    @classmethod
    def infer_target(
        self,
        source: tuple[str, ...],
        target: str | None = None,
        l_max: int | None = None,
        sort: bool = True,
    ) -> O3Space:
        """Infer output representation without computing CG coefficients.

        This method is useful e.g. for the :class:`e3j.linen.Bigotimes`
        flax wrapper.
        """
        out_filter = irrep_range(l_max) if l_max is not None else None
        # accumulator
        in_0 = O3Space(source[0])
        for k in range(1, len(source)):
            # last output filter
            if k == len(source) - 1 and target is not None:
                out_filter = O3Space(target)
            # binary output inference
            in_k = O3Space(source[k])
            in_0 = TensorProduct.infer_target((in_0, in_k), out_filter)
        out = O3Space(in_0)
        return out.regroup() if sort else out

    def __call__(self, *xs: jnp.ndarray) -> jnp.ndarray:
        """Evaluate N-linear product on N inputs.

        By default, this method uses a dense-dense matrix multiplication
        for the aggregation. The aggregation method can be changed for a
        sparse on by setting the environment variable E3J_AGGREGATION_METHOD
        to "scatter".
        """
        n = self.arity
        if not len(xs) == self.arity:
            raise TypeError(f"{n}-ary tensor product called with {len(xs)} arguments")
        ijk = self.indices
        xs_by_coef = jnp.stack(
            [xr[:, ijk[:, r + 1]] for r, xr in enumerate(xs)], axis=0
        )
        prod_xs_coef = self.values * jnp.prod(xs_by_coef, axis=0)
        return self.aggregate(prod_xs_coef)

    def pullback(self, prod: Bigotimes, position: int = 0) -> Bigotimes:
        """
        Flatten a composition of tensor products.

        By associativity of the binary tensor product,
        any succession of multilinear tensor products can be
        flattened down to a single operation.::

                x0 ... xk
                  ` | /
                   `|/
                    y0 ... yn          <=>         x0 ... xk y1 ... yn
                     `  |  /                         `    |   |    /
                      ` | /                            `___`_/___/
                       `|/                                  |
                        z                                   z

        Composing tensor products requires to repeat and multiply coefficients
        of the two l.h.s. trees, as any coefficient downstream is pulled-back
        by all the upstream coefficients matching its `y0` coordinate.

        Parameters
        ----------
            prod: `Bigotimes`
                an upstream tensor product.
            position: `int`
                the position where `prod`'s output is fed in as argument.

        Returns
        -------
            composed: `Bigotimes`
                the m-ary tensor product obtained by applying `prod`
                at `position`, where::

                    m = self.arity + prod.arity - 1
        """
        # pointers to incoming coefs by pulled coordinate
        # (upstream coefs are sorted by output coordinate)
        sizes_y = np.bincount(prod.indices[:, 0], minlength=prod.shape[0])
        ptr_y = np.concatenate(([0], np.cumsum(sizes_y)))

        # repeat downstream coefs by nb of incoming coefs above coordinate
        y_down = self.indices[:, position + 1]
        repeats = sizes_y[y_down]
        indices_down = np.repeat(self.indices, repeats, axis=0)
        values_down = np.repeat(self.values, repeats)

        # gather upstream coef blocks by pulled coordinate value:
        # rows ptr_y[i], ..., ptr_y[i + 1] - 1 for each i in y_down
        block_start = np.repeat(ptr_y[y_down], repeats)
        block_offset = np.arange(repeats.sum()) - np.repeat(
            np.cumsum(repeats) - repeats, repeats
        )
        rows_up = block_start + block_offset
        indices_up = prod.indices[rows_up, 1:]
        values_up = prod.values[rows_up]
        # compose values
        values = values_up * values_down

        # stack indices, removing pulled position

        # -1 + 1 = 0!
        p = len(self.source) + position if position < 0 else position

        indices = [
            indices_down[:, 0:1],  # output
            indices_down[:, 1 : p + 1],  # l.h.s. branches
            indices_up,  # upstream tree
            indices_down[:, p + 2 :],  # r.h.s. branches
        ]
        indices = np.concatenate([idx.astype(np.int64) for idx in indices], axis=-1)

        # return Bigotimes module with COO coefficients
        source = [*self.source[:position], *prod.source, *self.source[position + 1 :]]
        shape = (self.target.dim, *(in_k.dim for in_k in source))
        coef = sparse_coo(values, indices, shape=shape)

        return self.__class__(
            source,
            self.target,
            coef=coef,
            l_max=self.l_max,
            sort=False,
        )

    def sort(self) -> Bigotimes:
        """Sort irreducible output blocks."""
        perm = Permutation.sort(self.target)
        target = perm.target
        idx_out = perm.sigma_1[self.indices[:, 0]]
        indices = np.concatenate((idx_out[:, None], self.indices[:, 1:]), axis=-1)
        coef = sparse_coo(self.values, indices, shape=self.shape)

        return self.__class__(
            self.source,
            target,
            coef=coef,
            l_max=self.l_max,
            sort=False,
        )
