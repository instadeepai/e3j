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

"""
Structured dtype arrays used to pack sparse COO coefficients.

While Numpy supports structured dtype such as:

    _Coef = numpy.dtype([
        ("val", "float32"), ("i", "int32"), ("j", "int32"), ("k", "int32")
    ])

JAX does not, so we have to reinterpret_cast the packed Numpy array as
an opaque `idx_t*` array to pass coefficients through the XLA-FFI call.
"""

from enum import Enum
from typing import ClassVar, Self

import jax
import jax.numpy as jnp
import numpy
from flax import struct

from e3j.utils import next_pow2


class ValDtype(Enum):
    """Scalar value dtypes supported by the e3j_ops binary."""

    F16 = "float16"
    F32 = "float32"
    F64 = "float64"

    # TODO: add support for BF16

    def __str__(self):
        return self.value


class IdxDtype(Enum):
    """Index dtypes supported by the e3j_ops binary."""

    I32 = "int32"
    U8 = "uint8"

    # TODO: reintroduce support for U16

    def __str__(self):
        return self.value


#: Value dtypes the fused CUDA kernels dispatch on, as numpy dtype names.
SUPPORTED_VAL_DTYPES = tuple(dtype.value for dtype in ValDtype)


def resolve_val_dtype(input_dtype) -> str:
    """Return the value dtype the fused CUDA kernels should run in.

    Operations inherit the dtype of their operands: it drives both the operand
    casts and the coefficient packing, so the kernel reads coefficients at the
    stride the buffers were written at.

    Args:
        input_dtype: Promoted dtype of the operands, e.g. `jnp.result_type(x, y)`.

    Raises:
        TypeError: If `input_dtype` is not in `SUPPORTED_VAL_DTYPES`.
    """
    dtype = numpy.dtype(input_dtype).name
    if dtype not in SUPPORTED_VAL_DTYPES:
        raise TypeError(
            f"Fused CUDA ops support value dtypes {SUPPORTED_VAL_DTYPES}, "
            f"got {dtype!r}. Cast the operands."
        )
    return dtype


@struct.dataclass
class Coef:
    """Interface between a JAX-native BCOO format and a packed representation.

    While Numpy supports structured dtypes, JAX does not.
    This class allows to pass a packed representation of a BCOO array
    as an opaque `idx_dtype` array.

    Value and index dtypes are independent. A value smaller than one index slot
    (`float16` next to `int32` indices) just leaves the rest of that slot empty.

    Note:
        The base class assumes 3D coefficients for bilinear Clebsch-Gordan
        tensor products by default.
    """

    val: jnp.ndarray | numpy.ndarray
    idx: jnp.ndarray | numpy.ndarray

    val_dtype: ValDtype | str = "float32"
    idx_dtype: IdxDtype | str = "int32"

    rank: ClassVar[int] = 3
    index_names: ClassVar[list[str]] = ["i", "j", "k"]

    @property
    def dtype(self) -> numpy.dtype:
        val_t = ValDtype(self.val_dtype).value
        idx_t = IdxDtype(self.idx_dtype).value
        return self._numpy_dtype(val_t, idx_t)

    @classmethod
    def _numpy_dtype(cls, val_t: numpy.dtype, idx_t: numpy.dtype) -> numpy.dtype:
        """Structured dtype matching CUDA `alignas(next_pow2(...))` layout."""
        val_t, idx_t = numpy.dtype(val_t), numpy.dtype(idx_t)
        fields = [("val", val_t)] + [(name, idx_t) for name in cls.index_names]
        dt = numpy.dtype(fields, align=True)
        target = next_pow2(val_t.itemsize + cls.rank * idx_t.itemsize)
        if dt.itemsize < target:
            dt = numpy.dtype(
                {
                    "names": dt.names,
                    "formats": [dt.fields[n][0] for n in dt.names],
                    "offsets": [dt.fields[n][1] for n in dt.names],
                    "itemsize": target,
                }
            )
        return dt

    def __post_init__(self):
        # Normalize and validate dtypes:
        # accepts numpy/JAX dtypes, strings, or enum members.
        val_t = ValDtype(numpy.dtype(self.val_dtype).name).value
        idx_t = IdxDtype(numpy.dtype(self.idx_dtype).name).value
        object.__setattr__(self, "val_dtype", val_t)
        object.__setattr__(self, "idx_dtype", idx_t)
        assert self.idx.shape[:-1] == self.val.shape
        assert self.idx.shape[-1] == self.rank

    def to_numpy(self) -> Self:
        return self.__class__(self.val.__array__(), self.idx.__array__())

    def pack_numpy(self) -> numpy.ndarray:
        return numpy.array(
            [(val, *idx) for val, idx in self],
            dtype=self.dtype,
        )

    def pack_jax(self) -> jnp.ndarray:
        coef_np = self.pack_numpy()
        idx_t = numpy.dtype(IdxDtype(self.idx_dtype).value)
        numel, mod = divmod(self.dtype.itemsize, idx_t.itemsize)
        if mod != 0:
            raise TypeError(
                f"Cannot pack dtype pair: size of {idx_t} ({idx_t.itemsize}B) "
                f"does not divide size of structured dtype ({self.dtype.itemsize}B)."
            )
        # reinterpret_cast<idx_t*>(c)
        coef_cast = coef_np.view(idx_t).reshape(-1, numel)
        return jnp.asarray(coef_cast)

    def transpose(self, argnums: tuple[int, int, int]) -> Self:
        """Transposed COO coefficients with sorted indices and values."""
        # Note: Lexsorting indices *might* reduce bank conflicts.
        val, idx = self.val, self.idx.T
        first = argnums[0]
        sigma = jnp.argsort(idx[first])
        val_sorted = val[sigma]
        idx_sorted = jnp.stack([idx[i][sigma] for i in argnums])
        return self.__class__(
            val_sorted, idx_sorted.T, val_dtype=self.val_dtype, idx_dtype=self.idx_dtype
        )

    @classmethod
    def unpack(
        cls,
        data: jax.Array | numpy.ndarray,
        val_dtype: ValDtype | str,
    ) -> Self:
        """Unpack a packed array into (val, idx) pair.

        Args:
            data: An array of type `idx_dtype` and shape (..., N) where
                `N = sizeof(coef) / sizeof(idx_dtype)`.
            val_dtype: Supported `ValDtype` argument or string of corresponding
                numpy dtype.
        """
        # JAX stores as idx_t vectors, Numpy as structured {'val', 'i', 'j', 'k'} dtype.
        idx_t = data.dtype["i"] if data.dtype.fields is not None else data.dtype
        val_t = (
            val_dtype
            if isinstance(val_dtype, numpy.dtype)
            else numpy.dtype(f"{val_dtype}")
        )
        coef_t = cls._numpy_dtype(numpy.dtype(val_t), numpy.dtype(idx_t))
        numel = coef_t.itemsize // idx_t.itemsize
        # Index slots `val` spans, rounded up: a narrower value still takes one.
        numval = -(-val_t.itemsize // idx_t.itemsize)
        # Reinterpret as idx_t vector for slicing
        if isinstance(data, numpy.ndarray):
            data = data.view(dtype=idx_t).reshape(-1, numel)
        # Slice N slots, then keep their first value and drop the padding.
        val_as_idx = data[:, :numval]
        val = val_as_idx.view(dtype=val_t).reshape(val_as_idx.shape[0], -1)[:, 0]
        # Retrieve `cls.rank` index columns, skipping any trailing padding.
        idx = data[:, numval : numval + cls.rank]
        return cls(val, idx, val_dtype=val_t, idx_dtype=idx_t)

    def __iter__(self):
        """Yield `val, (i, j, k)` value/3D-index pairs."""
        return zip(self.val, self.idx)


@struct.dataclass
class Coef4D(Coef):
    """Interface between a 4D JAX-native BCOO format and a packed representation.

    See :class:`Coef` for more details.
    """

    rank: ClassVar[int] = 4
    index_names: ClassVar[list[str]] = ["i", "j", "k", "l"]
