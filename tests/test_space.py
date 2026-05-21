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

from typing import Any

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from e3j.arrays import O3Array, SO3Array
from e3j.arrays.array import Array, IndexArray
from e3j.spaces import Finite, O3Irrep, O3Space, SO3Space


def test_space_array_type():
    """
    Test that space classes have the correct array type attribute.

    Verifies that each space class is associated with its corresponding array type:
    - O3Space uses O3Array for array operations
    - SO3Space uses SO3Array for array operations
    - Finite uses IndexArray for array operations

    This ensures proper type consistency across the space and array hierarchy.
    """
    assert O3Space._array_type is O3Array
    assert SO3Space._array_type is SO3Array
    assert Finite._array_type is IndexArray


def test_duplicate_space_array_type_raises():
    """Defining a second Array subclass for the same Space must fail."""
    with pytest.raises(RuntimeError):

        class AnotherSO3Array(Array[SO3Space]):
            pass


class _TestO3Irrep:
    irrep: O3Irrep

    def test_rmul_dim(self):
        space = 8 * self.irrep
        assert isinstance(space, O3Space)
        assert space.dim == 8 * self.irrep.dim

    def test_mul_1o(self):
        other = "1o"
        self_to_e3nn = e3nn.Irrep(self.irrep.l, self.irrep.p)
        result = [(ir.l, ir.p) for _, ir in self.irrep * O3Irrep(other)]
        expect = [(ir.l, ir.p) for ir in self_to_e3nn * e3nn.Irrep(other)]
        assert all(ir1 == ir2 for ir1, ir2 in zip(expect, result))

    def test_mul_3e(self):
        other = "3e"
        self_to_e3nn = e3nn.Irrep(self.irrep.l, self.irrep.p)
        result = [(ir.l, ir.p) for _, ir in self.irrep * O3Irrep(other)]
        expect = [(ir.l, ir.p) for ir in self_to_e3nn * e3nn.Irrep(other)]
        assert all(ir1 == ir2 for ir1, ir2 in zip(expect, result))

    def test_eq(self):
        assert self.irrep == O3Irrep(self.irrep.l, self.irrep.p)

    def test_hashable(self):
        out = {self.irrep: 8}
        assert len(out)

    def test_iterable(self):
        l, p = self.irrep
        assert l == self.irrep.l and p == self.irrep.p

    def test_str(self):
        other = O3Irrep(str(self.irrep))
        l, p = self.irrep.l, self.irrep.p
        assert l == other.l and p == other.p


class TestO3Irrep2e(_TestO3Irrep):
    irrep = O3Irrep("2e")


class TestO3Irrep5o(_TestO3Irrep):
    irrep = O3Irrep(5, -1)


class _TestSpaceHashable:
    space: Any

    def test_hashable(self):
        hash(self.space)


class _TestO3Space(_TestSpaceHashable):
    space = O3Space

    def test_iter_dim(self):
        flat = [(m, ir.l, ir.p) for m, ir in self.space]
        assert self.space.dim == sum(m * (2 * l + 1) for m, l, _ in flat)

    def test_slices(self):
        slices = self.space.slices()
        assert len(slices) == len(self.space.blocks)
        assert slices[0].start == 0
        assert slices[-1].stop == self.space.dim
        for (m, ir), slc in zip(self.space.blocks, slices):
            assert slc.stop - slc.start == m * (2 * ir.l + 1)

    def test_sort(self):
        expect = e3nn.Irreps(str(self.space)).sort().irreps
        result = self.space.sort()
        assert O3Space(str(expect)) == result

    def test_sort_inverse(self):
        inv_to_e3nn = e3nn.Irreps(str(self.space)).sort().inv
        _, inv_e3j = self.space.sort(return_inverse=True)
        for i in range(len(inv_e3j)):
            assert inv_to_e3nn[i] == inv_e3j[i]

    def test_regroup(self):
        expect = e3nn.Irreps(str(self.space)).regroup()
        result = self.space.regroup()
        assert O3Space(str(expect)) == result


class TestO3Space1(_TestO3Space):
    space = O3Space("8x2o + 8x2e + 8x1o + 4x0e")


class TestO3Space2(_TestO3Space):
    space = O3Space([(8, "0e"), (2, "1o"), (8, "1e")])


class TestO3Space3(_TestO3Space):
    space = O3Space([(1, (l, (-1) ** l)) for l in range(8)])


class TestO3Space4(_TestO3Space):
    space = 8 * O3Space("0e + 1e + 2e")


class TestO3Space5(_TestO3Space):
    space = O3Space("1o + 2e + 3o + 0e + 1o + 2e + 0o")


class _TestSO3Space(_TestSpaceHashable):
    pass


class TestSO3Space1(_TestSO3Space):
    space = SO3Space("8x2 + 8x1 + 4x0")


class TestSO3Space2(_TestSO3Space):
    space = SO3Space([(8, 0), (2, 1), (8, 2)])


def test_o3space_from_mul_irrep_tuple():
    """O3Space must accept tuple[e3nn.MulIrrep].

    Flax nn.Module casts e3nn.Irreps attributes to tuple[MulIrrep] during __call__.
    O3Space must handle this so that modules can build spaces from their irreps fields.
    """
    irreps = e3nn.Irreps("2x1o + 4x2e")
    cast_by_flax = tuple(irreps)
    space = O3Space(cast_by_flax)
    assert space == O3Space(irreps)


def test_nn_module_o3space_from_irreps_attribute():
    """nn.Module with e3nn.Irreps attribute must be usable to construct O3Space.

    Flax casts e3nn.Irreps to tuple[MulIrrep] inside __call__, so O3Space(self.irreps)
    fails at runtime even though the module is constructed correctly.
    """

    class _Mod(nn.Module):
        irreps: e3nn.Irreps

        @nn.compact
        def __call__(self, x):
            _ = O3Space(self.irreps)
            return x

    m = _Mod(irreps=e3nn.Irreps("2x1o + 4x2e"))
    m.init(jax.random.key(0), jnp.ones((3,)))
