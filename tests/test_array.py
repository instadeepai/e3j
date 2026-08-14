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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from e3j.arrays import Array, O3Array
from e3j.spaces import O3Space


class _TestO3Array:
    _space: str = "0e+1o"
    dim: int = 4
    batch_size: int = 128
    num_channels: int = 32
    layout = "LEADING_CHANNELS"

    @property
    def space(self):
        return O3Space(self._space)

    @property
    def shape(self):
        n, nd, nc = (self.batch_size, self.dim, self.num_channels)
        if self.layout == "LEADING_CHANNELS":
            return (n, nc, nd)
        elif self.layout == "TRAILING_CHANNELS":
            return (n, nd, nc)

    @pytest.fixture(scope="class")
    def inputs(self) -> tuple[jax.Array, jax.Array]:
        x = jnp.ones(self.shape)
        y = jnp.zeros(self.shape) + jnp.arange(self.shape[-1])[None, None, :]
        return x, y

    @pytest.fixture(scope="class")
    def o3_inputs(self, inputs) -> tuple[O3Array, O3Array]:
        x, y = inputs
        return (
            O3Array(self.space, x, self.layout),
            O3Array(self.space, y, self.layout),
        )

    def test_blocks(self, o3_inputs):
        x, _ = o3_inputs
        blocks = list(x.blocks())
        assert len(blocks) == len(self.space.blocks)
        for (mul, ir), block in zip(self.space, blocks):
            assert isinstance(block, type(x))
            assert block.space == type(self.space)([(mul, ir)])
            assert block.layout == x.layout
            assert block.shape[block.feature_axis] == ir.dim

    def test_add(self, o3_inputs):
        x, y = o3_inputs
        z = x + y
        assert isinstance(z, O3Array) and z.space == self.space
        assert jnp.all(z.array == x.array + y.array)

    def test_sub(self, o3_inputs):
        x, y = o3_inputs
        z = x - y
        assert isinstance(z, O3Array) and z.space == self.space
        assert jnp.all(z.array == x.array - y.array)

    def test_add_sub_incompatible_space_raises(self, o3_inputs):
        x, y = o3_inputs
        other_space = O3Space(f"{self.dim}x0e")
        other = type(x)(other_space, y.array, self.layout)
        with pytest.raises(ValueError):
            x + other
        with pytest.raises(ValueError):
            x - other

    def test_sub_incompatible_space_error_mentions_subtract(self, o3_inputs):
        x, y = o3_inputs
        other_space = O3Space(f"{self.dim}x0e")
        other = type(x)(other_space, y.array, self.layout)
        with pytest.raises(ValueError, match="subtract"):
            x - other

    def test_add_sub_non_array_raises(self, o3_inputs):
        x, _ = o3_inputs
        with pytest.raises(ValueError):
            x + x.array
        with pytest.raises(ValueError):
            x - x.array

    @pytest.mark.xfail
    def test_radd(self, o3_inputs):
        x, _ = o3_inputs
        z = 2 + x
        assert isinstance(z, O3Array) and z.space == self.space
        assert jnp.all(z.array == 2 + x.array)

    @pytest.mark.xfail
    def test_illegal_mul(self, o3_inputs):
        x, y = o3_inputs
        z = x * y
        assert isinstance(z, O3Array) and z.space == self.space
        assert jnp.all(z.array == x.array * y.array)

    def test_rmul(self, o3_inputs):
        x, _ = o3_inputs
        z = 2 * x
        assert isinstance(z, O3Array) and z.space == self.space
        assert jnp.all(z.array == 2 * x.array)

    def test_batched_rmul(self, o3_inputs):
        x, _ = o3_inputs
        shape = list(self.shape)
        shape[x.feature_axis] = 1
        scalars = jnp.ones(tuple(shape))
        scalars *= jnp.arange(scalars.size).reshape(*shape)
        result = (scalars * x).array
        expect = scalars * x.array
        assert jnp.all(expect == result)

    @pytest.mark.xfail
    def test_illegal_rmul(self, o3_inputs):
        x, y = o3_inputs
        z = x * y

    @pytest.mark.xfail
    def test_shorter_feature_axis(self, inputs):
        x, y = inputs
        z = O3Array(self.space, x[:, :-2])
        assert not z

    def test_getitem(self, o3_inputs):
        _, y = o3_inputs
        idx = jnp.array([0, 1, 2])
        yi = y[idx]
        assert isinstance(yi, O3Array) and yi.space == self.space
        assert jnp.all(yi.array == y.array[idx])

    def test_getitem_numpy_index_off_feature_axis(self, o3_inputs):
        _, y = o3_inputs
        # A plain numpy (not jax) array index doesn't touch the feature
        # axis, so this must index normally rather than raising from an
        # ambiguous elementwise `==`/`!=`/`in` comparison against it.
        idx = np.array([0, 1, 2])
        yi = y[idx]
        assert isinstance(yi, O3Array) and yi.space == self.space
        assert jnp.all(yi.array == y.array[idx])

    def test_getitem_feature_axis_raises(self, o3_inputs):
        x, _ = o3_inputs
        axis = x.feature_axis % x.ndim
        # Same-length reversal: doesn't trip the shape/dim check in
        # Array.__init__, so it can only be caught by an explicit guard.
        key = [slice(None)] * x.ndim
        key[axis] = slice(None, None, -1)
        with pytest.raises(ValueError):
            x[tuple(key)]

    def test_getitem_multiaxis_boolean_mask_over_feature_axis_raises(self, o3_inputs):
        x, _ = o3_inputs
        # A boolean mask whose rank matches more than one axis (here: the
        # leading batch axis together with the feature axis) must still be
        # caught by the feature-axis guard, even though it is a single
        # index-tuple element rather than one element per axis.
        axis = x.feature_axis % x.ndim
        mask = jnp.zeros(x.shape[: axis + 1], dtype=bool)
        mask = mask.at[(0,) * mask.ndim].set(True)
        mask = mask.at[(1,) + (0,) * (mask.ndim - 1)].set(True)
        with pytest.raises(ValueError):
            x[mask]

    def test_jit_return(self, o3_inputs):
        x, y = o3_inputs

        @jax.jit
        def fn(a, b):
            return a + b

        z = fn(x, y)
        assert isinstance(z, O3Array) and z.space == self.space
        assert z.layout == x.layout
        assert jnp.all(z.array == x.array + y.array)


class TestO3ArrayLeading(_TestO3Array):
    layout = "LEADING_CHANNELS"


class TestO3ArrayTrailing(_TestO3Array):
    layout = "TRAILING_CHANNELS"


def test_array_repr_interpolates_layout():
    space = O3Space("0e+1o")
    # LEADING_CHANNELS is a non-default layout (project default is
    # TRAILING_CHANNELS), so Array.__repr__ appends the layout suffix.
    x = O3Array(space, jnp.zeros((2, 32, 4)), "LEADING_CHANNELS")
    # Call Array.__repr__ directly: O3Array gets its own dataclass-generated
    # __repr__, which shadows the inherited one for plain repr(x).
    r = Array.__repr__(x)
    assert "{self.layout}" not in r
    assert str(x.layout) in r


def test_rmul_lower_rank_scalar_trailing_channels():
    space = O3Space("0e+1o")
    x = O3Array(space, jnp.ones((128, 4, 32)), "TRAILING_CHANNELS")
    scalar = jnp.arange(32.0)  # per-channel scalar, fewer dims than x.array
    result = scalar * x
    assert isinstance(result, O3Array)
    assert jnp.all(result.array == scalar * x.array)
