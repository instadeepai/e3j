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

import math
from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
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
    # NOTE: the value dtype is chosen independently, see `ops.coef.ValDtype`.
    max_dim = max(shape)
    if max_dim <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    return np.dtype(np.int32)


@dataclass(frozen=True, eq=False)
class SparseCOO:
    """Sparse COO array of compile-time constant coefficients.

    Coefficients are held as numpy arrays, so that all manipulations
    (sorting, composition, packing...) happen on the host at trace time.

    Attributes:
        data: `(nnz,)` array of non-zero values.
        indices: `(nnz, ndim)` array of coordinates.
        shape: dense shape of the array.
    """

    data: np.ndarray
    indices: np.ndarray
    shape: tuple[int, ...]

    @property
    def nnz(self) -> int:
        return self.indices.shape[0]

    def todense(self) -> np.ndarray:
        dense = np.zeros(self.shape, dtype=self.data.dtype)
        np.add.at(dense, tuple(self.indices.T), self.data)
        return dense


def sparse_coo(
    values: np.ndarray,
    indices: np.ndarray,
    shape: tuple[int, ...],
    coalesce: bool = True,
) -> SparseCOO:
    """Return a `SparseCOO` array, coalesced by default.

    Coalescing sorts indices lexicographically, sums duplicate entries and
    drops the resulting zeros.
    Indices are narrowed to the smallest supported dtype, see
    :func:`narrow_index_dtype`.
    """
    shape = tuple(int(n) for n in shape)
    # Match the value dtype JAX would use for these coefficients.
    values = np.asarray(values)
    values = values.astype(jax.dtypes.canonicalize_dtype(values.dtype))
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != len(shape):
        raise ValueError(
            f"Expected indices of shape (nnz, {len(shape)}) for shape {shape}, "
            f"got {indices.shape}."
        )
    if indices.shape[0] != values.shape[0]:
        raise ValueError(
            f"Got {values.shape[0]} values for {indices.shape[0]} indices."
        )
    if indices.size and ((indices < 0).any() or (indices >= shape).any()):
        raise ValueError(f"Out of bounds indices for shape {shape}.")
    if coalesce:
        first = np.ones(indices.shape[0], dtype=bool)
        if math.prod(shape) <= np.iinfo(np.intp).max:
            # Sorting flat keys is much faster than sorting rows, and row-major
            # raveling preserves the lexicographic order. A stable sort is
            # fastest here as callers pass mostly sorted indices.
            keys = np.ravel_multi_index(tuple(indices.T), shape)
            order = np.argsort(keys, kind="stable")
            keys = keys[order]
            np.not_equal(keys[1:], keys[:-1], out=first[1:])
        else:
            order = np.lexsort(indices.T[::-1])
            rows = indices[order]
            np.any(rows[1:] != rows[:-1], axis=1, out=first[1:])
        # Sum each run of duplicate indices.
        summed = np.zeros(np.count_nonzero(first), dtype=values.dtype)
        np.add.at(summed, np.cumsum(first) - 1, values[order])
        indices = indices[order[first]]
        # Drop coefficients cancelling out, as `BCOO.sum_duplicates` does.
        nonzero = summed != 0
        indices, values = indices[nonzero], summed[nonzero]
    return SparseCOO(values, indices.astype(narrow_index_dtype(shape)), shape)


class SparseMixin:
    """Properties and methods depending on a `coef` attribute.

    The `coef` attribute is a :class:`SparseCOO` array of numpy coefficients.

    To switch between possible equivalent methods  -- typically,
    a matrix product or scatter-add operation in a final reduction stage,
    child classes such as :class:`TensorProduct` or :class:`Bigotimes`
    override the `aggregation_method` descriptor during `__init__`.
    The default attempts to read `$E3J_AGGREGATION_METHOD` from environment.
    """

    @cache
    def coef(self) -> SparseCOO:
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
        return self.coef.nnz

    @property
    def nnz_ratio(self) -> float:
        """Density of the coefficient tensor, i.e. `nnz` over its size."""
        return self.nnz / math.prod(self.shape)

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
            ) from err

    @property
    def target_matrix(self) -> sparse.BCOO | np.ndarray:
        """
        Return aggregation matrix of shape `(nnz, dim_out)`.

        By default, the aggregation matrix is a dense array. To get a sparse array,
        set the environment variable E3J_AGGREGATION_METHOD to "sparse".

        Example
        -------
        Number of coefficients above each output coordinate::

            >>> num_coefs = np.ones(op.nnz) @ op.target_matrix
        """
        dim_out = self.shape[0]
        shape = (self.nnz, dim_out)
        tgt = np.stack(
            (np.arange(self.nnz, dtype=np.int32), self.indices[:, 0].astype(np.int32)),
            axis=1,
        )
        ones = np.ones(self.nnz, dtype=np.float32)

        if self.aggregation_method == options.Aggregation.DENSE:
            target_matrix = np.zeros(shape, dtype=np.float32)
            target_matrix[tgt[:, 0], tgt[:, 1]] = ones
            return target_matrix

        # Rows of `tgt` are sorted and unique: no need to coalesce.
        return sparse.BCOO(
            (ones, tgt),
            shape=shape,
            indices_sorted=True,
            unique_indices=True,
        )

    def _aggregate_matmul(self, values):
        """Aggregate values on output coordinates using `matmul`.

        Args:
            values (jnp.ndarray): an array of shape (N, self.nnz),
                e.g. the product of CG coefficients with input coordinates.

        Returns:
            jnp.ndarray: array of shape `(N, target.dim)`.
        """
        return values @ self.target_matrix

    def _aggregate_scatter(self, values, layout: options.Layout | str = "E3NN"):
        """Aggregate values on output coordinates using `scatter_add`.

        Args:
            values (jnp.ndarray): an array of shape (N, self.nnz),
                e.g. the product of CG coefficients with input coordinates.

        Returns:
            jnp.ndarray: array of shape `(N, target.dim)`.
        """
        dim_out = self.target.dim
        idx_out = self.indices[:, 0]
        layout = options.Layout.parse(layout)

        # jax.lax.scatter_add
        # Accumulate in `values.dtype`: `jnp.zeros` defaults to float64 under
        # `jax_enable_x64`, promoting the output of a float32 tensor product.
        if self.aggregation_method == options.Aggregation.SCATTER:
            if layout in (options.Layout.E3NN, options.Layout.LEADING_CHANNELS):
                y_out = jnp.zeros((*values.shape[:-1], dim_out), dtype=values.dtype)
                return y_out.at[..., idx_out].add(
                    values,
                    indices_are_sorted=False,
                    mode="promise_in_bounds",
                )
            elif layout == options.Layout.TRAILING_CHANNELS:
                y_out = jnp.zeros(
                    (*values.shape[:-2], dim_out, values.shape[-1]),
                    dtype=values.dtype,
                )
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
            y_out = jnp.zeros((values.shape[0], dim_out), dtype=values.dtype)
            return scatter_add_1(idx_out, values, y_out)

    def aggregate(self, values, layout: options.Layout | str = "E3NN"):
        """Aggregate a values-like vector on output coordinates.

        Args:
            values (jnp.ndarray): a vector of length `self.nnz`, e.g. the
                product of CG coefficients with input coordinates.
            layout: array layout for scatter-based aggregation.

        Returns:
            jnp.ndarray: a vector of shape `target.dim`.
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
