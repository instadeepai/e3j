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

from e3j.ops.coef import Coef, IdxDtype, ValDtype


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


class TestIdxDtypeRoundTrip:
    """Test that packing Coef into an opaque array and recovering values is lossless."""

    @pytest.fixture(
        params=[
            ("float32", "int32"),
            # ("float32", "uint16"),  # disabled: IdxDtype.U16 removed
            ("float32", "uint8"),
        ]
    )
    def coef(self, request):
        val_dtype, idx_dtype = request.param
        val = numpy.array([1.0, -0.5, 3.14], dtype=val_dtype)
        idx = numpy.array(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            dtype=idx_dtype,
        )
        return Coef(val, idx, val_dtype=val_dtype, idx_dtype=idx_dtype)

    def test_pack_numpy_roundtrip(self, coef):
        """Structured numpy array preserves val and idx fields exactly."""
        packed = coef.pack_numpy()
        assert packed.dtype == coef.dtype
        numpy.testing.assert_array_equal(packed["val"], coef.val)
        numpy.testing.assert_array_equal(packed["i"], coef.idx[:, 0])
        numpy.testing.assert_array_equal(packed["j"], coef.idx[:, 1])
        numpy.testing.assert_array_equal(packed["k"], coef.idx[:, 2])

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
        numpy.testing.assert_array_equal(structured["val"], coef.val)
        numpy.testing.assert_array_equal(structured["i"], coef.idx[:, 0])
        numpy.testing.assert_array_equal(structured["j"], coef.idx[:, 1])
        numpy.testing.assert_array_equal(structured["k"], coef.idx[:, 2])

    def test_pack_jax_roundtrip(self, coef):
        """Coef.pack_jax() produces a JAX array that round-trips back to original values."""
        jax_arr = coef.pack_jax()

        # recover structured array from JAX
        idx_t = IdxDtype(coef.idx_dtype).value
        np_arr = numpy.asarray(jax_arr)
        structured = np_arr.view(coef.dtype).reshape(-1)

        numpy.testing.assert_array_equal(structured["val"], coef.val)
        numpy.testing.assert_array_equal(structured["i"], coef.idx[:, 0])
        numpy.testing.assert_array_equal(structured["j"], coef.idx[:, 1])
        numpy.testing.assert_array_equal(structured["k"], coef.idx[:, 2])

    def test_unpack_from_jax(self, coef):
        """Coef.unpack() recovers original val and idx from a packed JAX array."""
        jax_arr = coef.pack_jax()
        recovered = Coef.unpack(jax_arr, val_dtype=coef.val_dtype)

        numpy.testing.assert_array_equal(
            numpy.asarray(recovered.val), numpy.asarray(coef.val)
        )
        numpy.testing.assert_array_equal(
            numpy.asarray(recovered.idx), numpy.asarray(coef.idx)
        )

    def test_unpack_from_numpy(self, coef):
        """Coef.unpack() recovers original val and idx from a packed numpy structured array."""
        packed = coef.pack_numpy()
        recovered = Coef.unpack(packed, val_dtype=coef.val_dtype)

        numpy.testing.assert_array_equal(
            numpy.asarray(recovered.val), numpy.asarray(coef.val)
        )
        numpy.testing.assert_array_equal(
            numpy.asarray(recovered.idx), numpy.asarray(coef.idx)
        )
