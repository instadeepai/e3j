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

import e3nn_jax as e3nn
import jax
import jax.numpy as np
import jax.random as random
import numpy.testing
import pytest
from jax.experimental import sparse

import e3j
from e3j.core.tensor_product import TensorProduct
from e3j.utils.sparse import narrow_index_dtype

np.set_printoptions(edgeitems=12, precision=3)


def assert_allclose(expect, result, rtol=5e-6, atol=5e-6, debug: int = 1):
    """Show more on assertion errors to help diagnose addressing errors."""
    try:
        numpy.testing.assert_allclose(expect, result, rtol=rtol, atol=atol)
    except AssertionError as err:
        if debug >= 1:
            print("expect == result\n", abs(expect - result) < atol)
        if debug >= 2:
            print("expect\n", expect)
            print("result\n", result)
        raise err


@pytest.mark.xfail
def test_BCOO_is_mocked():  # noqa
    """Check that in-bound checks are performed."""
    coef = np.ones(3)
    index = np.array([[0, 1], [1, 2], [7, 8]])
    shape = (7, 8)
    matrix = sparse.BCOO((coef, index), shape=shape)  # noqa: unused-variable


class _TestTensorProduct:
    """Base class for tensor product tests."""

    out: str | None
    in1: str
    in2: str
    batch_size: int = 32
    _seed: int = 42

    @property
    def key(self) -> jax.Array:
        return random.key(self._seed)

    @pytest.fixture
    def inputs(self):
        key = jax.random.key(123)
        key1, key2 = jax.random.split(key)
        in1, in2 = e3nn.Irreps(self.in1), e3nn.Irreps(self.in2)
        x1 = jax.random.normal(key1, (self.batch_size, in1.dim))
        x2 = jax.random.normal(key2, (self.batch_size, in2.dim))
        return (x1, x2)

    @pytest.fixture(scope="class")
    def e3nn_module(self):
        """Return e3nn module acting on raw jax inputs."""
        irreps_out = e3nn.Irreps(self.out) if self.out is not None else self.out

        def tensor_product(x1, x2):
            x_out = e3nn.tensor_product(
                e3nn.IrrepsArray(self.in1, x1),
                e3nn.IrrepsArray(self.in2, x2),
                filter_ir_out=irreps_out,
                irrep_normalization="none",
            )
            return x_out.array

        return tensor_product

    @pytest.fixture(scope="class")
    def e3j_module(self):
        """Return our module acting on raw jax inputs."""
        return TensorProduct((self.in1, self.in2), self.out, sort=True)

    @pytest.fixture(scope="class")
    def rotations(self, e3j_module) -> list[np.ndarray]:
        """Prepare output and input rotations."""
        rotation = e3nn.rand_matrix(self.key)
        lhs, rhs = e3j_module.source
        target = e3j_module.target
        rotation_in1 = lhs.action(rotation)
        rotation_in2 = rhs.action(rotation)
        rotation_out = target.action(rotation)
        return [rotation_out, rotation_in1, rotation_in2]

    def assert_zero(self, data, tol: float = 5e-6):
        norm = float(np.sqrt(np.sum(data**2))) / data.size
        assert norm < tol

    # --- Test functions ---

    def test_irrep_order(self, e3j_module):
        source = e3j_module.source
        target_e3j = e3j_module.target
        target_to_e3nn = e3nn.tensor_product(
            source[0]._to_e3nn(),
            source[1]._to_e3nn(),
            filter_ir_out=self.out,
            regroup_output=True,
        )
        try:
            assert e3j_module.target._to_e3nn() == target_to_e3nn
        except Exception as err:
            print("e3j:", target_e3j, "\te3nn:", target_to_e3nn)
            raise err

    def test_e3nn(self, inputs, e3j_module, e3nn_module):
        result = e3j_module(*inputs)
        expect = e3nn_module(*inputs)
        assert_allclose(expect, result, rtol=1e-5, atol=1e-5)

    def test_equivariance(self, inputs, e3j_module, rotations):
        """Check that module commutes with SO3."""
        x1, x2 = inputs
        gfx = e3j_module(x1, x2) @ rotations[0]
        fgx = e3j_module(x1 @ rotations[1], x2 @ rotations[2])
        self.assert_zero(gfx - fgx)

    def test_index_dtype(self, e3j_module):
        """Check that CG indices are narrowed to the smallest fitting dtype."""
        coef = e3j_module.coef
        expected_dtype = narrow_index_dtype(coef.shape)
        assert coef.indices.dtype == expected_dtype

    def test_jittable(self, inputs, e3j_module):
        """Check that module can be compiled.

        May fail if some BCOO manipulations are not excluded from the
        JIT context with `jax.ensure_compile_time_eval()`.
        """
        x1, x2 = inputs
        in1, in2, tgt = self.in1, self.in2, self.out
        # Delay computation of coefficients to catch tracer errors
        tp = jax.jit(lambda x, y: TensorProduct((in1, in2), tgt, sort=True)(x, y))
        z = tp(x1, x2)
        assert z.shape[-1] == e3j_module.target.dim


class TestTensorProduct_1_1(_TestTensorProduct):

    out = None
    in1 = "1o"
    in2 = "1o"


class TestTensorProduct_2_2(_TestTensorProduct):

    out = None
    in1 = "2e"
    in2 = "2e"


class TestTensorProduct_012_123(_TestTensorProduct):

    out = None
    in1 = "3x0e + 2x1o + 2e"
    in2 = "1o + 2x2e + 3e"


class TestTensorProduct_out012_012_012(_TestTensorProduct):

    out = "0e + 1o + 2e"
    in1 = "2x0e + 2x1o + 2x2e"
    in2 = "4x0e + 1o + 2x2e"


class _TestTensorProductLayout(_TestTensorProduct):
    """Test tensor product with non-default layouts."""

    n_channels: int = 8

    def test_layout_consistency(self, e3j_module):
        """TensorProduct commutes with LEADING <-> TRAILING layout cast."""
        tp_leading = TensorProduct(
            (self.in1, self.in2),
            self.out,
            sort=True,
            layout="LEADING_CHANNELS",
        )
        tp_trailing = TensorProduct(
            (self.in1, self.in2),
            self.out,
            sort=True,
            layout="TRAILING_CHANNELS",
        )
        K = self.n_channels
        in1 = e3nn.Irreps(self.in1)
        in2 = e3nn.Irreps(self.in2)
        key1, key2 = random.split(random.key(456))
        x1_lc = random.normal(key1, (self.batch_size, K, in1.dim))
        x2 = random.normal(key2, (self.batch_size, in2.dim))

        z_lc = tp_leading(x1_lc, x2)
        assert z_lc.shape == (self.batch_size, K, e3j_module.target.dim)

        x1_tc = np.swapaxes(x1_lc, -1, -2)
        z_tc = tp_trailing(x1_tc, x2)
        assert z_tc.shape == (self.batch_size, e3j_module.target.dim, K)

        assert_allclose(np.swapaxes(z_lc, -1, -2), z_tc)


class TestLayout_1_1(_TestTensorProductLayout):
    out = None
    in1 = "1o"
    in2 = "1o"


class TestLayout_012_012(_TestTensorProductLayout):
    out = "0e + 1o + 2e"
    in1 = "2x0e + 2x1o + 2x2e"
    in2 = "4x0e + 1o + 2x2e"


@pytest.mark.e3j_ops
class TestTensorProductFused(_TestTensorProduct):

    out = "0e + 1o + 2e"
    in1 = "2x0e + 2x1o + 2x2e"
    in2 = "4x0e + 1o + 2x2e"

    @pytest.fixture(scope="class")
    def e3j_module(self):
        with e3j.config.use(tensor_product="FUSED_CUDA"):
            return TensorProduct((self.in1, self.in2), self.out, sort=True)

    def test_e3nn(self, inputs, e3j_module, e3nn_module):
        # precision is slightly lower on GPU
        result = e3j_module(*inputs)
        expect = e3nn_module(*inputs)
        assert_allclose(expect, result, rtol=2e-3, atol=2e-3)

    def test_jit_pack_jax(self, inputs, e3j_module, e3nn_module):
        """Coef.pack_jax() must succeed inside JIT via _fused_eval."""
        x1, x2 = inputs
        in1, in2, tgt = self.in1, self.in2, self.out

        with e3j.config.use(tensor_product="FUSED_CUDA"):
            f = jax.jit(lambda x, y: TensorProduct((in1, in2), tgt, sort=True)(x, y))
            z = f(x1, x2)

        expect = e3nn_module(x1, x2)
        assert_allclose(expect, z, rtol=2e-3, atol=2e-3)

    def test_jit_pack_jax_backward(self, inputs, e3nn_module):
        """Coef.pack_jax() must succeed in the backward pass under JIT."""
        x1, x2 = inputs
        in1, in2, tgt = self.in1, self.in2, self.out

        with e3j.config.use(tensor_product="FUSED_CUDA"):

            @jax.jit
            def grad_fused(x, y):
                def f(x, y):
                    return np.sum(TensorProduct((in1, in2), tgt, sort=True)(x, y))

                return jax.grad(f, argnums=(0, 1))(x, y)

            dx, dy = grad_fused(x1, x2)

        def f_ref(x, y):
            return np.sum(e3nn_module(x, y))

        dx_ref, dy_ref = jax.grad(f_ref, argnums=(0, 1))(x1, x2)
        assert_allclose(dx_ref, dx, rtol=2e-3, atol=2e-3)
        assert_allclose(dy_ref, dy, rtol=2e-3, atol=2e-3)

    @pytest.mark.parametrize("batched", ["x", "y"])
    def test_vmap_single_operand(self, inputs, e3nn_module, batched):
        """vmap with only one equivariant operand batched (jacrev's pattern)."""
        x1, x2 = inputs
        in1, in2, tgt = self.in1, self.in2, self.out

        with e3j.config.use(tensor_product="FUSED_CUDA"):
            tp = TensorProduct((in1, in2), tgt, sort=True)
            if batched == "x":
                stack = np.stack([x1, x1 * 2.0, x1 * 3.0], axis=0)
                result = jax.vmap(tp, in_axes=(0, None))(stack, x2)
                expect = np.stack([e3nn_module(s, x2) for s in stack])
            else:
                stack = np.stack([x2, x2 * 2.0, x2 * 3.0], axis=0)
                result = jax.vmap(tp, in_axes=(None, 0))(x1, stack)
                expect = np.stack([e3nn_module(x1, s) for s in stack])

        assert_allclose(expect, result, rtol=2e-3, atol=2e-3)

    def test_jacrev_of_grad(self, inputs, e3nn_module):
        """jacrev(grad(...)) routes through the new vmap-of-bwd path."""
        x1, x2 = inputs
        in1, in2, tgt = self.in1, self.in2, self.out

        def loss_ref(x, y):
            return np.sum(e3nn_module(x, y))

        with e3j.config.use(tensor_product="FUSED_CUDA"):

            def loss_fused(x, y):
                return np.sum(TensorProduct((in1, in2), tgt, sort=True)(x, y))

            d2_fused = jax.jacrev(jax.grad(loss_fused, argnums=0), argnums=1)(x1, x2)

        d2_ref = jax.jacrev(jax.grad(loss_ref, argnums=0), argnums=1)(x1, x2)
        assert_allclose(d2_ref, d2_fused, rtol=2e-3, atol=2e-3)
