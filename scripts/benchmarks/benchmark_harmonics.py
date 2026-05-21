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

from __future__ import annotations

import functools

import e3nn_jax as e3nn
import e3x
import jax
import jax.numpy as np
import jax.random as random

from e3j.core.harmonics import Harmonics, Y
from e3j.spaces import O3Space

from .e3_benchmark import E3Benchmark

L_MAX = 9
BATCH_SIZE = 2048
ITERATIONS = 1000
PROFILE = False


class HarmonicsBenchmark(E3Benchmark):

    keys = ("e3x", "e3nn", "e3j", "e3j_float32")  # "e3j_float32"

    def __init__(
        self,
        l_max: int,
        batch_size: int = 1024,
        n_it: int = 1000,
    ):
        self.l_max = l_max
        self.source = O3Space("1o")
        self.target = O3Space([(1, (l, (-1) ** l)) for l in range(self.l_max + 1)])
        self.batch_size = batch_size
        self.n_it = n_it
        self.n_samp = 10
        self.rng = random.key(123)

    def e3j_inputs(self) -> tuple[jax.Array]:
        shape = (self.n_samp, self.batch_size, 3)
        return (random.normal(self.rng, shape),)

    def e3j_float32_inputs(self) -> tuple[jax.Array]:
        return self.e3j_inputs()

    def e3nn_inputs(self) -> tuple[jax.Array]:
        return self.e3j_inputs()

    def e3x_inputs(self) -> tuple[jax.Array]:
        return self.e3j_inputs()

    @property
    def dim_in(self):
        return self.source.dim

    @functools.cached_property
    def e3x(self):
        return functools.partial(
            e3x.so3.spherical_harmonics,
            max_degree=self.l_max,
            r_is_normalized=True,
        )

    @functools.cached_property
    def e3nn(self):
        sh = functools.partial(
            e3nn.spherical_harmonics,
            self.target._to_e3nn(),
            normalize=False,
        )
        return lambda r: sh(r).array

    @functools.cached_property
    def e3j(self):
        return Harmonics(self.l_max, real=True)

    @functools.cached_property
    def e3j_float32(self):
        return self.e3j.polynomial.C()

    def grad(self) -> HarmonicsGradBenchmark:
        """Benchmark the gradients (vjp) of polynomials."""
        return HarmonicsGradBenchmark(self)

    def hessian(self) -> HarmonicsGradBenchmark:
        """Benchmark the hessian (vjp) of polynomials."""
        return HarmonicsHessianBenchmark(self)

    def __str__(self):
        return f"Ylm(r) for l <= {self.l_max}"


class HarmonicsGradBenchmark(HarmonicsBenchmark):

    keys = ("e3j", "e3nn", "e3x")

    def __init__(self, primitive: HarmonicsBenchmark, batch_size=None):
        self.primitive = primitive
        self.l_max = primitive.l_max
        self.batch_size = primitive.batch_size
        self.n_it = primitive.n_it
        self.n_samp = primitive.n_samp
        self.rng = primitive.rng
        # fix: differentiate polynomial beforehand
        self.primitive.e3j.polynomial.diff
        self.source = primitive.source
        self.target = 3 * primitive.target

    @functools.cached_property
    def e3j(self):
        return jax.grad(lambda r: np.sum(self.primitive.e3j(r)))

    @functools.cached_property
    def e3j_float32(self):
        return jax.grad(lambda r: np.sum(self.primitive.e3j_float32(r).real))

    @functools.cached_property
    def e3x(self) -> jax.Array:
        return jax.grad(lambda r: np.sum(self.primitive.e3x(r)))

    @functools.cached_property
    def e3nn(self) -> jax.Array:
        return jax.grad(lambda r: np.sum(self.primitive.e3nn(r)))

    def __str__(self) -> str:
        return "∇." + str(self.primitive)


class HarmonicsHessianBenchmark(HarmonicsBenchmark):

    keys = ("e3j", "e3nn")
    # e3x: ~1min compilation, 1.8s / jitted run, OOM

    def __init__(self, primitive: HarmonicsBenchmark, batch_size=None):
        self.primitive = primitive
        self.l_max = primitive.l_max
        self.batch_size = primitive.batch_size
        self.n_it = primitive.n_it
        self.rng = primitive.rng

    @functools.cached_property
    def e3j(self):
        return jax.hessian(lambda r: np.sum(self.primitive.e3j(r)))

    @functools.cached_property
    def e3j_float32(self):
        return jax.hessian(lambda r: np.sum(self.primitive.e3j_float32(r).real))

    @functools.cached_property
    def e3x(self) -> jax.Array:
        return jax.vmap(jax.hessian(lambda r: np.sum(self.primitive.e3x(r))), 0)

    @functools.cached_property
    def e3nn(self) -> jax.Array:
        return jax.vmap(jax.hessian(lambda r: np.sum(self.primitive.e3nn(r))), 0)

    def __str__(self) -> str:
        return "∇.∇." + str(self.primitive)


if __name__ == "__main__":
    benchmark = HarmonicsBenchmark(
        l_max=L_MAX,
        batch_size=BATCH_SIZE,
        n_it=ITERATIONS,
    )
    benchmark.run()
    # benchmark.trace("tmp/harmonics")

    """
    # Ylm.diff
    benchmark_grad = benchmark.grad()
    benchmark_grad.run()
    benchmark_grad.trace("tmp/harmonics-grad")

    # Ylm.diff.diff
    benchmark_hessian = benchmark.hessian()
    benchmark_hessian.run()
    """
