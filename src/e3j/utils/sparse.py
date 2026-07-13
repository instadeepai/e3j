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

import os
from enum import Enum
from functools import cached_property

import jax
import jax.numpy as np
from jax.experimental import sparse

from e3j.ops import scatter_add_1
from e3j.utils import config, options
from e3j.utils.cache import cache


def narrow_index_dtype(shape: tuple[int, ...]) -> np.dtype:
    """Select narrowest supported index dtype for a sparse array shape.

    Args:
        shape: tuple of feature dimensions. In practice, this the 3D shape
               of the joined Clebsch-Gordan coefficients.

    Returns:
        A numpy dtype which is one of 'uint8' or 'int32' (JAX default).
        The return type should be aligned to either 32, 64 or 128 bits
        (excluding 32 multiple 92) for consistency with our alignment
        on the CUDA side.
    """
    # TODO: pair uint16 with float16 only (check from Coef class).
    #       Always skip uint16 with 32-bit value type to avoid 92bit alignment.
    max_dim = max(shape)
    if max_dim <= np.iinfo(np.uint8).max:
        return np.uint8
    return np.int32


def sparse_bcoo(
    values: np.ndarray,
    indices: np.ndarray,
    shape: tuple[int, ...],
    coalesce: bool = True,
) -> sparse.BCOO:
    """Return a coalesced `sparse.BCOO` array."""
    with jax.ensure_compile_time_eval():
        # Widen to int32 for coalescing: BCOO internals (sort_indices,
        # sum_duplicates) use shape values which may not fit in narrow dtypes.
        if indices.dtype != np.int32:
            indices = indices.astype(np.int32)
        array = sparse.BCOO((values, indices), shape=shape)
        if coalesce:
            array = array.sort_indices().sum_duplicates()
        idx_dtype = narrow_index_dtype(shape)
        if idx_dtype != array.indices.dtype:
            array = sparse.BCOO(
                (array.data, array.indices.astype(idx_dtype)), shape=shape
            )
        return array


class SparseMixin:
    """Properties and methods depending on a `coef` attribute.

    The `coef` attribute can in fact be either of type:

    * `jax.experimental.sparse.BCOO`, or

    * `np.ndarray` : in this case some properties and methods will fail, such as
      `.aggregate()`, `.nnz`...

    To switch between possible equivalent methods  -- typically,
    a matrix product or scatter-add operation in a final reduction stage,
    child classes such as :class:`TensorProduct` or :class:`Bigotimes`
    override the `aggregation_method` descriptor during `__init__`.
    The default attempts to read `$E3J_AGGREGATION_METHOD` from environment.
    """

    @cache
    def coef(self) -> jax.Array:
        """Compute and cache coefficients."""
        raise NotImplementedError(
            "`SparseMixin` requires `self` to have a `.coef` attribute or (cached) property."
        )

    @property
    def indices(self) -> np.ndarray:
        return self.coef.indices

    @property
    def values(self) -> np.ndarray:
        return self.coef.data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coef.shape

    @property
    def nnz(self) -> int:
        """Number of non-zero coefficients."""
        return self.indices.shape[0]

    @property
    def nnz_ratio(self) -> float:
        """Density of the coefficient tensor, i.e. `nnz` over its size."""
        shape = np.array(self.shape)
        return self.nnz / float(np.prod(shape))

    @property
    def aggregation_method(self) -> options.Aggregation:
        """Cache `self._aggregation_method` with default."""
        if hasattr(self, "_aggregation_method"):
            return options.Aggregation(self._aggregation_method)
        return config().aggregation

    @aggregation_method.setter
    def aggregation_method(self, value: options.Aggregation):
        try:
            self._aggregation_method = options.Aggregation(value)
        except ValueError as err:
            raise RuntimeError(
                f"Unknown aggregation method {value}, available options are:\n"
                f"{list(options.Aggregation)}"
            )

    @cached_property
    def target_matrix(self) -> sparse.BCOO | np.ndarray:
        """
        Return aggregation matrix of shape `(nnz, dim_out)`.

        By default, the aggregation matrix is a dense array. To get a sparse array,
        set the environment variable E3J_AGGREGATION_METHOD to "sparse".

        Example
        -------
        Number of coefficients above each output coordinate::

            >>> num_coefs = jnp.ones(op.nnz) @ op.target_matrix
        """
        with jax.ensure_compile_time_eval():

            dim_out = self.shape[0]
            shape = (self.nnz, dim_out)

            idx_out = self.indices[:, 0]
            tgt = np.stack(
                (
                    np.arange(self.nnz, dtype=np.int32),
                    idx_out.astype(np.int32),
                ),
                axis=1,
            )
            ones = np.ones(self.nnz, dtype=np.float32)

            if self.aggregation_method == options.Aggregation.DENSE:
                dense_zeros = np.zeros(shape, dtype=np.float32)
                target_matrix = dense_zeros.at[tgt[:, 0], tgt[:, 1]].add(ones)

            else:
                target_matrix = sparse_bcoo(ones, tgt, shape=shape)

        return target_matrix

    def _aggregate_matmul(self, values):
        """Aggregate values on output coordinates using `matmul`.

        Args:
            values (np.ndarray): an array of shape (N, self.nnz),
                e.g. the product of CG coefficients with input coordinates.

        Returns:
            np.ndarray: array of shape `(N, target.dim)`.
        """
        return values @ self.target_matrix

    def _aggregate_scatter(self, values, layout: options.Layout | str = "E3NN"):
        """Aggregate values on output coordinates using `scatter_add`.

        Args:
            values (np.ndarray): an array of shape (N, self.nnz),
                e.g. the product of CG coefficients with input coordinates.

        Returns:
            np.ndarray: array of shape `(N, target.dim)`.
        """
        dim_out = self.target.dim
        idx_out = self.indices[:, 0]
        layout = options.Layout.parse(layout)

        # jax.lax.scatter_add
        if self.aggregation_method == options.Aggregation.SCATTER:
            if layout in (options.Layout.E3NN, options.Layout.LEADING_CHANNELS):
                y_out = np.zeros((*values.shape[:-1], dim_out))
                return y_out.at[..., idx_out].add(
                    values,
                    indices_are_sorted=False,
                    mode="promise_in_bounds",
                )
            elif layout == options.Layout.TRAILING_CHANNELS:
                y_out = np.zeros((*values.shape[:-2], dim_out, values.shape[-1]))
                return y_out.at[..., idx_out, :].add(
                    values,
                    indices_are_sorted=False,
                    mode="promise_in_bounds",
                )

        # e3j_ops.scatter_add_1
        elif self.aggregation_method == options.Aggregation.SCATTER_1:
            if layout == options.Layout.TRAILING_CHANNELS:
                raise NotImplementedError(
                    "NYI: scatter-reduction kernel with trailing channels"
                )
            assert values.ndim == 2, "values must be 2D"
            y_out = np.zeros((values.shape[0], dim_out))
            return scatter_add_1(idx_out, values, y_out)

    def aggregate(self, values, layout: options.Layout | str = "E3NN"):
        """Aggregate a values-like vector on output coordinates.

        Args:
            values (np.ndarray): a vector of length `self.nnz`, e.g. the
                product of CG coefficients with input coordinates.
            layout: array layout for scatter-based aggregation.

        Returns:
            np.ndarray: a vector of shape `target.dim`.
        """
        if self.aggregation_method in (
            options.Aggregation.DENSE,
            options.Aggregation.SPARSE,
        ):
            return self._aggregate_matmul(values)

        if self.aggregation_method in (
            options.Aggregation.SCATTER,
            options.Aggregation.SCATTER_1,
        ):
            return self._aggregate_scatter(values, layout=layout)

        raise RuntimeError(self.aggregation_method)
