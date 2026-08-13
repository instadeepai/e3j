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

import e3nn_jax as e3nn
import jax
import jax.numpy as np
import pytest
from jax import random

from e3j.core.harmonics import Harmonics
from e3j.utils.safe import safe_norm


class _TestHarmonics:
    """Base class for harmonics polynomial tests."""

    out: int | str
    normalization: str = "integral"
    batch_size: int = 4096
    _seed: int = 42

    @property
    def key(self) -> jax.Array:
        return random.key(self._seed)

    @pytest.fixture
    def inputs(self, normalized=True):
        rs = random.normal(
            random.key(123),
            (self.batch_size, 3),
        )
        if normalized:
            norm = safe_norm(rs, axis=-1, eps=1e-8)
            rs_norm = rs / norm[:, None]
            return rs_norm
        return rs

    @pytest.fixture
    def small_inputs(self, normalized=True):
        rs = random.normal(
            random.key(123),
            (64, 3),
        )
        if normalized:
            norm = safe_norm(rs, axis=-1, eps=1e-8)
            rs_norm = rs / norm[:, None]
            return rs_norm
        return rs

    @pytest.fixture
    def tangents(self, inputs):
        r = inputs
        v = random.normal(random.key(789), (self.batch_size, 3))
        r_cross_v = [
            r[:, 1] * v[:, 2] - r[:, 2] * v[:, 1],
            r[:, 2] * v[:, 0] - r[:, 0] * v[:, 2],
            r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0],
        ]
        return np.stack(r_cross_v, axis=-1)

    @pytest.fixture(scope="class")
    def e3j_module(self):
        """Our polynomial evaluation."""
        return Harmonics(self.out, normalize=True, normalization=self.normalization)

    @pytest.fixture(scope="class")
    def e3nn_module(self):
        """Wrapper around `e3nn.spherical_harmonics`."""
        if isinstance(self.out, int):
            out = e3nn.Irreps([(1, (l, (-1) ** l)) for l in range(self.out + 1)])
        else:
            out = e3nn.Irreps(self.out)

        def e3nn_harmonics(x: np.ndarray) -> np.ndarray:
            return e3nn.spherical_harmonics(
                out, x, True, normalization=self.normalization
            ).array

        return e3nn_harmonics

    @pytest.fixture(scope="class")
    def rotations(self, e3j_module) -> list[np.ndarray]:
        """Prepare output and input rotations."""
        rotation_in = e3nn.rand_matrix(self.key)
        rotation_out = e3j_module.target.action(rotation_in)
        return [rotation_in, rotation_out]

    def assert_proportional(self, expect, result, tol: int = 1e-5):
        ratio = result / (expect + 1e-8)
        avg_ratio = np.mean(ratio, axis=0)
        norm = safe_norm(ratio - avg_ratio, eps=0.0) / ratio.size
        try:
            assert norm < tol
        except AssertionError as err:
            print(f"{self.__class__.__name__}.assert_proportional\n", avg_ratio)
            raise err

    def assert_close(self, expect, result, tol: float = 3e-4):
        """Assert the two outputs are equal value-for-value (not just proportional).

        The tolerance accommodates TPU float32 matmul precision: e3j and e3nn
        agree to ~1e-6 in float32 on CPU, but the ladder recursion at high ``l``
        accumulates ~1e-4 worst-case error on TPU (even at the ``highest`` matmul
        precision pinned in ``pytest.ini``).
        """
        max_abs = float(np.max(np.abs(result - expect)))
        try:
            assert max_abs < tol
        except AssertionError as err:
            print(f"{self.__class__.__name__}.assert_close max_abs={max_abs}")
            raise err

    def assert_zero(self, data, tol: int = 1e-6):
        norm = float(np.sqrt(np.sum(data**2))) / data.size
        try:
            assert norm < tol
        except BaseException as err:
            print(self.__class__.__name__ + ".assert_zero")
            print("error.std", np.std(data, axis=0), sep="\n")
            raise err

    def expected_s2_norm(self, l: int) -> float:
        """Expected `sqrt(integral_S2(Y_lm^2))` for `self.normalization`.

        Mirrors e3nn's documented relationship between conventions: "integral"
        has unit S2-integral by definition, "component" scales it by `4*pi`,
        and "norm" scales it by `4*pi / (2l + 1)`.
        """
        if self.normalization == "integral":
            return 1.0
        if self.normalization == "component":
            return math.sqrt(4 * math.pi)
        if self.normalization == "norm":
            return math.sqrt(4 * math.pi / (2 * l + 1))
        raise NotImplementedError(self.normalization)

    # --- Test functions ---

    def test_jvp(self, inputs, tangents, e3j_module, e3nn_module):
        expect = jax.jvp(e3nn_module, [inputs], [tangents])[1]
        result = jax.jvp(e3j_module, [inputs], [tangents])[1]
        self.assert_close(expect, result)

    def test_jacrev_equal_jacfwd(self, small_inputs, e3j_module):
        fwd = jax.jacfwd(e3j_module)(small_inputs)
        rev = jax.jacrev(e3j_module)(small_inputs)
        self.assert_zero(fwd - rev)

    def test_normalization(self, inputs, e3j_module, tol=2e-4):
        """Check Monte-Carlo expectation on the sphere."""
        N = self.batch_size
        results = e3j_module(inputs)
        s2_mass = 4 * np.pi
        s2_expectation = (1 / N) * np.sum(np.abs(results) ** 2, axis=0)
        s2_integral = s2_expectation * s2_mass
        s2_norm = np.sqrt(s2_integral)
        expected = np.concatenate(
            [
                np.full(2 * irrep_out.l + 1, self.expected_s2_norm(irrep_out.l))
                for _, irrep_out in e3j_module.target
            ]
        )
        self.assert_zero(expected - s2_norm, tol=5e-3)

    def test_e3nn(self, inputs, e3j_module, e3nn_module):
        """Check that e3j and e3nn modules give the same output."""
        result = e3j_module(inputs)
        expect = e3nn_module(inputs)
        return self.assert_close(expect, result)

    def test_default_normalization(self, inputs):
        """Check that leaving `normalization` unset agrees with e3nn_jax's default."""
        out = (
            e3nn.Irreps([(1, (l, (-1) ** l)) for l in range(self.out + 1)])
            if isinstance(self.out, int)
            else e3nn.Irreps(self.out)
        )
        result = Harmonics(self.out, normalize=True)(inputs)
        expect = e3nn.spherical_harmonics(out, inputs, True).array
        self.assert_close(expect, result)

    def test_grad_jit(self, inputs, e3j_module):
        sum_P = jax.jit(lambda r: np.sum(e3j_module(r)))
        dP = jax.grad(sum_P)
        dPx = dP(inputs)
        assert dPx.shape == (inputs.shape[0], 3)

    def test_jit_grad(self, inputs, e3j_module):
        dP = jax.grad(lambda r: np.sum(e3j_module(r)))
        jit_dP = jax.jit(dP)
        dPx = jit_dP(inputs)
        assert dPx.shape == (inputs.shape[0], 3)

    def test_equivariance(self, inputs, e3j_module, rotations):
        """Check that module commutes with SO3.

        The harmonics and the Wigner-D (``target.action``) both live in the e3nn
        (y, z, x) frame, so equivariance holds directly: rotating the output by
        ``D(R)`` equals evaluating the harmonics on the rotated inputs.
        """
        gfx = e3j_module(inputs) @ rotations[1]
        fgx = e3j_module(inputs @ rotations[0])
        self.assert_zero(gfx - fgx)

    def test_e3nn_equivariance(self, inputs, e3nn_module, rotations):
        """Check that e3nn module commutes with SO3."""
        gfx = e3nn_module(inputs) @ rotations[1]
        fgx = e3nn_module(inputs @ rotations[0])
        self.assert_zero(gfx - fgx)


class TestHarmonicsP(_TestHarmonics):

    out = "1o"


class TestHarmonicsSP(_TestHarmonics):

    out = "0e + 1o"


class TestHarmonics5(_TestHarmonics):

    out = "0e + 1o + 2e + 3o + 4e + 5o"


class TestHarmonics5Component(_TestHarmonics):

    out = "0e + 1o + 2e + 3o + 4e + 5o"
    normalization = "component"


class TestHarmonics5Norm(_TestHarmonics):

    out = "0e + 1o + 2e + 3o + 4e + 5o"
    normalization = "norm"
