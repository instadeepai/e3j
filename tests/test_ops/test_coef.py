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

import jax.numpy as jnp
import numpy
import pytest

from e3j.ops.coef import Coef, Coef4D, IdxDtype, ValDtype


class TestCoefDtypeSize:
    """Check that aligned numpy dtypes match CUDA struct sizes.

    CUDA structs are padded to a multiple of the largest member's
    alignment (float32 = 32 bits). numpy.dtype(align=True) must agree.
    """

    def test_coef_int32_is_16B(self):
        # 32 + 32 + 32 + 32 = 128 bits, no padding needed
        dt = Coef._numpy_dtype(numpy.dtype("float32"), numpy.dtype("int32"))
        assert dt.itemsize == 16

    # uint16 disabled — numpy gives 12B, CUDA gives 16B (alignas(16) inflates
    # 10 raw bytes to next power of 2). This mismatch is the bug; the test is
    # left commented to document the Python side of it.
    # def test_coef_uint16_is_12B(self):
    #     dt = Coef._numpy_dtype(numpy.dtype("float32"), numpy.dtype("uint16"))
    #     assert dt.itemsize == 12  # != 16 (CUDA sizeof) — that's the problem

    def test_coef_uint8_is_8B(self):
        # 32 + 8 + 8 + 8 = 56 bits, padded to 64 (multiple of 32)
        dt = Coef._numpy_dtype(numpy.dtype("float32"), numpy.dtype("uint8"))
        assert dt.itemsize == 8


class TestCoef4DDtypeSize:
    """Check that Coef4D numpy dtypes match CUDA alignas(next_pow2(...)) sizes."""

    def test_coef4d_int32_is_32B(self):
        # raw = 4 + 4*4 = 20B, next_pow2(20) = 32
        dt = Coef4D._numpy_dtype(numpy.dtype("float32"), numpy.dtype("int32"))
        assert dt.itemsize == 32

    def test_coef4d_uint8_is_8B(self):
        # raw = 4 + 4*1 = 8B, next_pow2(8) = 8
        dt = Coef4D._numpy_dtype(numpy.dtype("float32"), numpy.dtype("uint8"))
        assert dt.itemsize == 8


# --- Round-trip tests (shared base) ---


DTYPE_PAIRS = [
    ("float32", "int32"),
    # ("float32", "uint16"),  # disabled: IdxDtype.U16 removed
    ("float32", "uint8"),
]


class _TestIdxDtypeRoundTrip:
    """Base: packing into an opaque array and recovering values is lossless."""

    coef_cls: type[Coef]
    sample_idx: list[list[int]]

    @pytest.fixture(params=DTYPE_PAIRS)
    def coef(self, request):
        val_dtype, idx_dtype = request.param
        val = numpy.array([1.0, -0.5, 3.14], dtype=val_dtype)
        idx = numpy.array(self.sample_idx, dtype=idx_dtype)
        return self.coef_cls(val, idx, val_dtype=val_dtype, idx_dtype=idx_dtype)

    def _assert_fields(self, structured, coef):
        numpy.testing.assert_array_equal(structured["val"], coef.val)
        for col, name in enumerate(type(coef).index_names):
            numpy.testing.assert_array_equal(structured[name], coef.idx[:, col])

    def test_pack_numpy_roundtrip(self, coef):
        """Structured numpy array preserves val and idx fields exactly."""
        packed = coef.pack_numpy()
        assert packed.dtype == coef.dtype
        self._assert_fields(packed, coef)

    def test_view_cast_roundtrip(self, coef):
        """reinterpret_cast via numpy.view -> JAX -> numpy recovers original data."""
        packed = coef.pack_numpy()
        idx_t = IdxDtype(coef.idx_dtype).value
        itemsize = packed.dtype.itemsize
        idx_itemsize = numpy.dtype(idx_t).itemsize
        numel = itemsize // idx_itemsize

        # reinterpret_cast to flat idx_t array
        opaque = packed.view(idx_t).reshape(-1, numel)

        # round-trip through JAX
        jax_arr = jnp.asarray(opaque)
        recovered = numpy.asarray(jax_arr)

        # cast back to structured dtype
        structured = recovered.view(packed.dtype).reshape(-1)
        self._assert_fields(structured, coef)

    def test_pack_jax_roundtrip(self, coef):
        """pack_jax() produces a JAX array that round-trips back to original values."""
        jax_arr = coef.pack_jax()

        # recover structured array from JAX
        np_arr = numpy.asarray(jax_arr)
        structured = np_arr.view(coef.dtype).reshape(-1)
        self._assert_fields(structured, coef)

    def test_unpack_from_jax(self, coef):
        """unpack() recovers original val and idx from a packed JAX array."""
        jax_arr = coef.pack_jax()
        recovered = type(coef).unpack(jax_arr, val_dtype=coef.val_dtype)

        numpy.testing.assert_array_equal(
            numpy.asarray(recovered.val), numpy.asarray(coef.val)
        )
        numpy.testing.assert_array_equal(
            numpy.asarray(recovered.idx), numpy.asarray(coef.idx)
        )

    def test_unpack_from_numpy(self, coef):
        """unpack() recovers original val and idx from a packed numpy structured array."""
        packed = coef.pack_numpy()
        recovered = type(coef).unpack(packed, val_dtype=coef.val_dtype)

        numpy.testing.assert_array_equal(
            numpy.asarray(recovered.val), numpy.asarray(coef.val)
        )
        numpy.testing.assert_array_equal(
            numpy.asarray(recovered.idx), numpy.asarray(coef.idx)
        )


class TestIdxDtypeRoundTrip(_TestIdxDtypeRoundTrip):
    coef_cls = Coef
    sample_idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


class TestCoef4DIdxDtypeRoundTrip(_TestIdxDtypeRoundTrip):
    coef_cls = Coef4D
    sample_idx = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


# --- Transpose tests (shared base) ---


class _TestTranspose:
    """Base: column permutation + sort by new first column."""

    coef_cls: type[Coef]
    sample_val: list[float]
    sample_idx: list[list[int]]
    permutations: list[tuple[int, ...]]

    @pytest.fixture
    def coef(self):
        val = jnp.array(self.sample_val)
        idx = jnp.array(self.sample_idx, dtype=jnp.int32)
        return self.coef_cls(val, idx)

    @pytest.mark.parametrize(
        "argnums",
        [pytest.param(p, id=str(p)) for p in []],
    )
    def test_values_follow_indices(self, coef, argnums):
        """Each (val, indices...) tuple is preserved after transpose."""
        t = coef.transpose(argnums)
        rank = type(coef).rank

        def to_set(c):
            return {(float(v), *(int(x) for x in row)) for v, row in zip(c.val, c.idx)}

        orig_set = to_set(coef)
        trans_set = to_set(t)

        inv = [0] * rank
        for dst, src in enumerate(argnums):
            inv[src] = dst
        trans_unpermuted = {
            (entry[0], *(entry[1 + inv[r]] for r in range(rank))) for entry in trans_set
        }
        assert orig_set == trans_unpermuted


class TestCoefTranspose(_TestTranspose):
    coef_cls = Coef
    sample_val = [1.0, -0.5, 3.14, 2.0]
    sample_idx = [[2, 0, 1], [0, 2, 1], [1, 1, 0], [0, 1, 2]]

    @pytest.mark.parametrize(
        "argnums",
        [(0, 1, 2), (1, 0, 2), (2, 0, 1)],
    )
    def test_values_follow_indices(self, coef, argnums):
        super().test_values_follow_indices(coef, argnums)


class TestCoef4DTranspose(_TestTranspose):
    coef_cls = Coef4D
    sample_val = [1.0, -0.5, 3.14, 2.0]
    sample_idx = [[2, 0, 1, 3], [0, 2, 1, 0], [1, 1, 0, 2], [0, 1, 2, 1]]

    @pytest.mark.parametrize(
        "argnums",
        [(0, 1, 2, 3), (1, 0, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1)],
    )
    def test_values_follow_indices(self, coef, argnums):
        super().test_values_follow_indices(coef, argnums)
