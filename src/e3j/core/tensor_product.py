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
import itertools
import os
from math import log2
from typing import Optional

import e3nn_jax as e3nn
import jax
import jax.numpy as np
import numpy
from jax import Array
from jax.experimental import sparse

import e3j.utils as utils
from e3j.ops import TensorProductParams as Params
from e3j.ops import tensor_product
from e3j.ops.coef import Coef
from e3j.pallas_ops.tensor_product import (
    PallasMosaicTPUTensorProductParams,
    tensor_product_pallas_mosaic_tpu,
)
from e3j.spaces import O3Space
from e3j.utils import cache, options
from e3j.utils.options import Layout
from e3j.utils.sparse import SparseMixin, sparse_bcoo

from .permutation import Permutation


class TensorProduct(SparseMixin):
    """Bilinear tensor products.

    This class implements (equivariant) bilinear maps of the form::

        z[:,i] = Σⱼₖ c[i,j,k] * x[:,j] * y[:,k],

    defined by a 3D coefficient array `c` in sparse BCOO format.

    Evaluation is implemented by:

    1. a pull-back of `x` and `y` by coefficient indices `j` and `k`,
    2. a product of `x[:,j]`, `y[:,k]` and coefficient values `c[i,j,k]`,
    3. a sparse reduction step to aggregate output coordinates `z[:,i]`.
    """

    @utils.cache
    def coef(self):
        """Clebsch-Gordan coefficients.

        Either computed or retrieved from cache. The `coef` descriptor
        is writable and may be assigned a value during `__init__`.
        """
        with jax.ensure_compile_time_eval():
            return self.clebsch_gordan(self.target, *self.source, self.normalization)

    def __init__(
        self,
        source: tuple[O3Space, O3Space] | tuple[str, str],
        target: O3Space | str | None,
        coef: Optional[sparse.BCOO] = None,
        sort: bool = True,
        config: utils.Config | None = None,
        layout: str | options.Layout = "LEADING_CHANNELS",
        mode: str | options.TPMode = "OUTER",
        normalization: str | options.TPNormalization = "NONE",
    ):
        """
        Stack Clebsch Gordan coefficients or use explicitly given ones.

        Parameters
        ----------
        source: representations of both inputs.

        target: output representation, inferred by default.

                If coefficients are `None`, then `out` is interpreted as
                a momentum and parity filter which should only contain
                irreducible blocks of multiplicity 1.

        coef: sparse array, constructed by stacking Clebsch-Gordan
              coefficients by default.

        sort: whether to sort the output coordinates by grouping momenta and
              parities, `True` by default.

        layout: specifies the channel axis, `TRAILING_CHANNELS` is faster.

        mode: specifies how to mix the channels, can take three values:

          * `OUTER` : tensor product of channels in `np.outer` fashion.
            Also used to broadcast one operand on the channel axis.
          * `INNER` : scalar product of channels, summed over after the
            Clebsch-Gordan tensor product.
          * `MAP` : channel-wise tensor products. Only useful with trailing
            channels layout, since leading axes are mapped over by default.
        """
        # This switch decides whether:
        # - Clebsch-Gordan coefficients should be computed and stacked
        # - Output representation should be inferred from optional filter
        infer_out = coef is None

        # --- I/O irreps ---

        if len(source) != 2:
            raise TypeError("TensorProduct expects a binary tuple as source argument.")
        self.source = tuple(O3Space(src_i) for src_i in source)

        if target is None:
            # infer total target from CG rules
            self.target = self.infer_target(self.source)
        elif infer_out:
            # infer target from CG rules + filter
            keep_ir_out = O3Space(target)
            self.target = self.infer_target(self.source, keep_ir_out)
        else:
            # programmatic call with known coefficients and target
            self.target = O3Space(target)

        # --- Sparse/Dense switch ---

        self.config = utils.config.state() if config is None else config
        self.aggregation_method = self.config.aggregation
        self.layout = Layout.parse(layout)
        self.mode = options.TPMode.parse(mode)
        self.normalization = options.TPNormalization.parse(normalization)

        # --- Coefficients ---

        # Fill cache from pre-computed coefficients
        if coef is not None:
            self.coef = coef

        # sort irreducible output blocks
        if sort and infer_out:
            with jax.ensure_compile_time_eval():
                other = self.sort()
                self.target, self.coef = other.target, other.coef

    @property
    def is_dense(self):
        return self.config.tensor_product == options.TensorProduct.DENSE

    @property
    def is_fused(self):
        return self.config.tensor_product == options.TensorProduct.FUSED

    @property
    def is_mtpu(self):
        return self.config.tensor_product == options.TensorProduct.FUSED_MOSAIC_TPU

    @classmethod
    def infer_target(
        cls,
        source: tuple[O3Space, O3Space],
        target: O3Space | None = None,
        sort: bool = False,
    ) -> O3Space:
        """Infer target representation from Clebsch-Gordan rules.

        The optional `target` argument should only contain irreducible
        representations with multiplicity 1, and `ValueError` will
        be raised otherwise.
        """
        return O3Space.otimes(source[0], source[1], target, sort)

    def sort(self) -> "TensorProduct":
        """Sort irreducible output blocks.

        In contrast with `e3nn`, the reordering of output coordinates is
        performed once and for all on the full coefficient tensor,
        avoiding the associated overhead at evaluation time.
        """
        perm = Permutation.sort(self.target)
        if not self.is_dense:
            idx_out = perm.sigma_1[self.indices[:, 0]]
            indices = np.concat(
                (idx_out[:, None], self.indices[:, 1:]),
                axis=-1,
            )
            coef = sparse_bcoo(self.values, indices, self.shape)
        else:
            coef = self.coef[perm.sigma]
        return self.__class__(
            self.source,
            perm.target,
            coef=coef,
            sort=False,
        )

    @staticmethod
    def clebsch_gordan(
        target: O3Space,
        source_1: O3Space,
        source_2: O3Space,
        normalization: options.TPNormalization = options.TPNormalization.NONE,
    ) -> sparse.BCOO:
        """
        Stack Clebsch-Gordan coefficients.

        Returns a 3D-array whose dimensions appear in the same order as the arguments.
        """
        values, indices = [], []
        in1, in2 = source_1, source_2
        keep_ir_out = set((ir.l, ir.p) for _, ir in target)
        # pointer to first (output) coordinate of (l0, l1, l2) block
        begin_out = 0

        for ((m1, ir1), slc1), ((m2, ir2), slc2) in itertools.product(
            zip(in1, in1.slices()),
            zip(in2, in2.slices()),
        ):
            l1, l2 = ir1.l, ir2.l
            k1, k2 = numpy.mgrid[:m1, :m2].reshape(2, -1)

            # Fully-connected: ⊗  of channel dimensions
            m0 = m1 * m2

            for _, ir0 in ir1 * ir2:
                l0, p0 = ir0.l, ir0.p

                # filter blocks by output l0
                if (l0, p0) not in keep_ir_out:
                    continue

                # 3D Clebsch-Gordan tensor on irreducible block (convert from JAX)
                cg_dense = numpy.array(e3nn.clebsch_gordan(l0, l1, l2))
                nz = numpy.nonzero(cg_dense)
                cg_data = cg_dense[nz]
                cg_indices = numpy.stack(nz, axis=-1)  # shape (nnz, 3)

                # optionally, rescale coefficients
                if normalization == options.TPNormalization.SQRT_DIM_OUT:
                    cg_data = cg_data * numpy.sqrt(2 * l0 + 1)

                # accumulate repeated coefficient values
                values.append(
                    numpy.tile(cg_data, m1 * m2),
                )

                # accumulate indices shifted by offsets:
                #   1) offset_0 points to one reducible 3D block
                #   2) offsets point to irreducible 3D sub-blocks
                offset_0 = numpy.array(
                    [begin_out, slc1.start, slc2.start],
                )
                dim_irreps = numpy.array([ir0.dim, ir1.dim, ir2.dim])

                # k0 = row-major of k1, k2 for e3nn compatibility
                k0 = k1 * m2 + k2
                mul_grid = numpy.stack((k0, k1, k2), axis=-1)

                offsets = offset_0 + mul_grid * dim_irreps

                indices.append((cg_indices[None, :] + offsets[:, None]).reshape(-1, 3))
                begin_out += m0 * ir0.dim

        # return sparse COO coefficients
        values = numpy.concatenate(values)
        indices = numpy.concatenate(indices, axis=0)
        shape = (target.dim, in1.dim, in2.dim)

        return sparse_bcoo(values, indices, shape)

    def dense_eval(self, x: Array, y: Array, coef: Array) -> Array:
        """Evaluate bilinear map on pair of inputs."""
        return np.einsum("ijk, ...j, ...k -> ...i", coef, x, y)

    @staticmethod
    def _broadcast_operands_leading(x: Array, y: Array) -> tuple[Array, Array]:
        if x.ndim == y.ndim:
            return x, y
        if x.ndim == y.ndim + 1:
            return x, y[..., None, :]
        if y.ndim == x.ndim + 1:
            return x[..., None, :], y
        raise ValueError(f"Incompatible operand ranks: {x.ndim} vs {y.ndim}")

    @staticmethod
    def _broadcast_operands_trailing(x: Array, y: Array) -> tuple[Array, Array]:
        if x.ndim == y.ndim:
            return x, y
        if x.ndim == y.ndim + 1:
            return x, y[..., None]
        if y.ndim == x.ndim + 1:
            return x[..., None], y
        raise ValueError(f"Incompatible operand ranks: {x.ndim} vs {y.ndim}")

    def sparse_eval(self, x: Array, y: Array, coef: Array) -> Array:
        """Evaluate bilinear map on pair of inputs."""
        ijk, c_ijk = coef.indices, coef.data
        # Gather inputs
        if self.layout in (Layout.E3NN, Layout.LEADING_CHANNELS):
            x, y = self._broadcast_operands_leading(x, y)
            x_j = x[..., ijk[:, 1]]
            y_k = y[..., ijk[:, 2]]
        elif self.layout == Layout.TRAILING_CHANNELS:
            x, y = self._broadcast_operands_trailing(x, y)
            x_j = x[..., ijk[:, 1], :]
            y_k = y[..., ijk[:, 2], :]
            c_ijk = c_ijk[..., None]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        # Multiply into summand terms
        cxy_ijk = x_j * y_k * c_ijk
        # Aggregate summands by output coordinate.
        # Note: can't be staticmethod there because of SparseMixin
        return self.aggregate(cxy_ijk, layout=self.layout)

    def fused_eval(
        self,
        x: Array,
        y: Array,
        coef: Array,
    ) -> Array:
        """Evaluate bilinear map on pair of inputs."""
        idx = coef.indices
        val = coef.data
        params = Params(
            num_out=self.target.dim,
            layout=self.layout,
            mode=self.mode,
        )
        # Pack coefficients as opaque `idx_t` vector.
        with jax.ensure_compile_time_eval():
            coef = Coef(val, idx, val_dtype=val.dtype, idx_dtype=idx.dtype).pack_jax()
        return tensor_product(coef, x, y, params)

    def _mtpu_params(self, coef: sparse.BCOO | None = None):
        """Build Pallas Mosaic TPU kernel parameters from this tensor product.

        The kernel statically unrolls the Clebsch-Gordan structure from the
        COO `(idx, coef)` arrays, so they are materialized as concrete numpy
        arrays at trace time.
        """

        if coef is None:
            coef = self.coef
        if self.mode not in (options.TPMode.OUTER, options.TPMode.MAP):
            raise NotImplementedError(
                f"Mosaic TPU tensor product does not support mode {self.mode}; "
                "use 'OUTER' or 'MAP'."
            )
        with jax.ensure_compile_time_eval():
            idx = numpy.asarray(coef.indices)
            val = numpy.asarray(coef.data)
        return PallasMosaicTPUTensorProductParams(
            indices=idx,
            values=val,
            layout=self.layout,
            mode=self.mode,
            x_space=self.source[0],
            y_space=self.source[1],
            z_space=self.target,
        )

    def mtpu_eval(self, x: Array, y: Array, coef: Array) -> Array:
        """Evaluate bilinear map via the Pallas Mosaic TPU kernel."""

        params = self._mtpu_params(coef)
        return tensor_product_pallas_mosaic_tpu(x, y, params)

    def __call__(self, x: Array, y: Array, coef: Array | None = None) -> Array:
        """Evaluate bilinear map on pair of inputs."""
        # Only load coefficients from `self` if not passed explicitly
        if coef is None:
            coef = self.coef
        # Algorithm branching
        if self.is_dense:
            return self.dense_eval(x, y, coef)
        if self.is_fused:
            return self.fused_eval(x, y, coef)
        if self.is_mtpu:
            return self.mtpu_eval(x, y, coef)
        else:
            return self.sparse_eval(x, y, coef)
