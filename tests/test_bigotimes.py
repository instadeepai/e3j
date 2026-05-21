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

from typing import Literal

import e3nn_jax as e3nn
import jax
import jax.numpy as np
import jax.random as random
import pytest

from e3j.core.bigotimes import Bigotimes
from e3j.spaces import O3Space
from e3j.utils.irreps import irrep_range

np.set_printoptions(edgeitems=6, precision=3)


class _TestBigotimesPullback:
    """Check that Bigotimes.pullback coincides with composition."""

    ins: list[str]
    upstream: Bigotimes
    downstream: Bigotimes

    batch_size: int = 32
    _seed: int = 42

    @property
    def key(self) -> jax.Array:
        return random.key(self._seed)

    @pytest.fixture
    def inputs(self):
        keys = random.split(self.key, len(self.ins))
        nb = self.batch_size
        reps_in = [e3nn.Irreps(in_k) for in_k in self.ins]
        return tuple(random.normal(key, (nb, rk.dim)) for key, rk in zip(keys, reps_in))

    @pytest.fixture
    def pullback(self, upstream, downstream):
        return downstream.pullback(upstream)

    @pytest.fixture
    def sequential(self, upstream, downstream):

        def composed(*xs):
            r = upstream.arity
            y = upstream(*xs[:r])
            return downstream(y, *xs[r:])

        return composed

    def assert_zero(self, data, tol: int = 1e-6):
        norm = float(np.sqrt(np.sum(data**2))) / data.size
        try:
            assert norm < tol
        except BaseException as err:
            print(self.__class__.__name__ + ".assert_zero")
            print("error.std", np.std(data, axis=0), sep="\n")
            raise err

    def test_pullback_eval(self, inputs, sequential, pullback):
        expect = sequential(*inputs)
        result = pullback(*inputs)
        self.assert_zero(expect - result)


class _TestBigotimes:
    """Base class for multilinear product tests."""

    out: int | str | None
    ins: list[str]
    l_max: int | None = None
    aggregation_method: Literal["scatter"] | Literal["dense"] = "scatter"
    batch_size: int = 32
    _seed: int = 123

    @property
    def key(self) -> jax.Array:
        return random.key(self._seed)

    @pytest.fixture
    def inputs(self):
        keys = random.split(self.key, len(self.ins))
        reps = (e3nn.Irreps(in_k) for in_k in self.ins)
        return tuple(
            random.normal(k, (self.batch_size, in_k.dim)) for k, in_k in zip(keys, reps)
        )

    @pytest.fixture(scope="class")
    def e3nn_module(self):
        """Return e3nn module acting on raw jax inputs."""
        # e3nn output Irreps
        irreps_out = e3nn.Irreps(self.out) if self.out is not None else self.out
        if self.l_max is None:
            filters = [None] * (len(self.ins) - 2) + [irreps_out]
        else:
            reps = irrep_range(self.l_max, pseudotensors=True)
            filters = [reps] * (len(self.ins) - 1)

        out = e3nn.tensor_product(
            self.ins[0],
            self.ins[1],
            filter_ir_out=filters[0],
        )
        for in_k, filter_k in zip(self.ins[2:], filters[1:]):
            out = e3nn.tensor_product(
                out,
                in_k,
                filter_ir_out=filter_k,
            )

        print("e3nn output:", out)

        def bigotimes(*xs):
            x_out = e3nn.tensor_product(
                e3nn.IrrepsArray(self.ins[0], xs[0]),
                e3nn.IrrepsArray(self.ins[1], xs[1]),
                filter_ir_out=filters[0],
                irrep_normalization="none",
            )
            for in_k, x_k, filter_k in zip(self.ins[2:], xs[2:], filters[1:]):
                x_out = e3nn.tensor_product(
                    x_out,
                    e3nn.IrrepsArray(in_k, x_k),
                    filter_ir_out=filter_k,
                    irrep_normalization="none",
                )
            return x_out.array

        return bigotimes

    @pytest.fixture(scope="class")
    def e3j_module(self):
        """Return our module acting on raw jax inputs."""
        tp = Bigotimes(self.ins, self.out, l_max=self.l_max)
        print("\n")
        print("e3j output", tp.target)
        print("nnz ratio:", tp.nnz_ratio)
        return tp

    @pytest.fixture
    def rotations(self, e3j_module) -> list[np.ndarray]:
        """Prepare output and input rotations."""
        rots = []
        rotation = e3nn.rand_matrix(self.key)
        rot_0 = e3j_module.target.action(rotation)
        rots.append(rot_0)
        for in_k in self.ins:
            rot_k = O3Space(in_k).action(rotation)
            rots.append(rot_k)
        print([rot.shape for rot in rots])
        return rots

    def assert_zero(self, data, tol: int = 1e-6):
        norm = float(np.sqrt(np.sum(data**2))) / data.size
        try:
            assert norm < tol
        except BaseException as err:
            print("error.std", np.std(data, axis=0), sep="\n")
            raise err

    # --- Test functions ---

    def test_e3nn(self, inputs, e3j_module, e3nn_module):
        """Check that e3j and e3nn modules give the same output."""
        result = e3j_module(*inputs)
        expect = e3nn_module(*inputs)
        return self.assert_zero(result - expect)

    def test_equivariance(self, inputs, e3j_module, rotations):
        """Check that module commutes with SO3."""
        gfx = e3j_module(*inputs) @ rotations[0]
        fgx = e3j_module(*[xk @ rot_k for xk, rot_k in zip(inputs, rotations[1:])])
        self.assert_zero(gfx - fgx)

    def test_infer_target(self, e3j_module):
        """Check that Bigotimes.infer_target agrees with actual target"""
        target = Bigotimes.infer_target(self.ins, self.out, self.l_max)
        assert target == e3j_module.target


# --- Pullback test cases ---


class TestBigotimesPullback_p3(_TestBigotimesPullback):
    """TestBigotimesPullback on p**3"""

    ins = ["1o", "1o", "1o"]
    batch_size = 32

    @pytest.fixture(scope="class")
    def upstream(self):
        return Bigotimes(["1o", "1o"], None, sort=False)

    @pytest.fixture(scope="class")
    def downstream(self):
        return Bigotimes(["0e + 1e + 2e", "1o"], None, sort=False)


class TestBigotimesPullback_ReducibleInputs(_TestBigotimesPullback):
    """TestBigotimesPullback on (s+p)**3"""

    ins = ["0e + 1o", "0e + 1o", "0e + 1o"]

    @pytest.fixture(scope="class")
    def upstream(self):
        return Bigotimes(["0e + 1o", "0e + 1o"], None, sort=False)

    @pytest.fixture(scope="class")
    def downstream(self, upstream):
        return Bigotimes([upstream.target, "0e + 1o"], None, sort=False)


# --- Equivariance test cases ---


class TestBigotimes_1_1_1(_TestBigotimes):

    out = None
    ins = ["1o", "1o", "1o"]

    @pytest.mark.tolerated
    def test_e3nn(self, inputs, e3j_module, e3nn_module):
        result = e3j_module(*inputs)
        expect = e3nn_module(*inputs)
        # TODO: strange e-4 deviation at indices 1, 2, 3
        return self.assert_zero(result - expect, 1e-5)


class TestBigotimes_1_2_1(_TestBigotimes):

    out = None
    ins = ["1o", "2e", "1o"]


class TestBigotimes_1_2_1_dense(_TestBigotimes):

    out = None
    ins = ["1o", "2e", "1o"]
    aggregation_method = "dense"


class TestBigotimes_01_012_01(_TestBigotimes):

    out = "0e + 0o + 1o + 1e + 2e + 2o"
    ins = ["0e+1o", "0e+1o+2e", "0e+1o"]
    l_max = 3

    @pytest.mark.skip("output filter incompatible with e3nn_module")
    def test_e3nn(self, inputs, e3j_module, e3nn_module):
        result = e3j_module(*inputs)
        expect = e3nn_module(*inputs)
        return self.assert_zero(result - expect)

    @pytest.mark.tolerated
    def test_equivariance(self, inputs, e3j_module, rotations):
        """Check that module commutes with SO3."""
        gfx = e3j_module(*inputs) @ rotations[0]
        fgx = e3j_module(*[xk @ rot_k for xk, rot_k in zip(inputs, rotations[1:])])
        self.assert_zero(gfx - fgx, tol=3.5e-6)


class TestBigotimes_1_2_1_2_lmax2(_TestBigotimes):

    out = None
    ins = ["1o", "2e", "1o", "2e"]
    l_max = 2
    batch_size = 8

    @pytest.mark.skip("l_max argument incompatible with e3nn")
    def test_e3nn(self, inputs, e3j_module, e3nn_module):
        result = e3j_module(*inputs)
        expect = e3nn_module(*inputs)
        return self.assert_zero(result - expect, tol=1e-7)
