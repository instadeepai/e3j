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

from collections import defaultdict
from functools import cached_property, partial
from typing import Any, TypeAlias

import e3nn_jax as e3nn
import jax
from jax import random

from e3j.spaces import O3Space

from .backends import torch
from .utils import timed_loop

# interpret `(mul, l_max)` as `mul x (0 + ... + l_max)`
E3Rep: TypeAlias = str | tuple[int, int]

N_SAMP = 2


class E3Benchmark:
    """Base class for E3-libraries benchmarks.

    Subclass for any equivariant module by implementing:

    * `self.e3?(*inputs)`

    where `e3?` stands for any of the library keys `e3nn/e3x/e3j`.
    Note that `self.e3?` may equivalently be returned by a cached property.

    The default `self.e3?_inputs()` will return raw/wrapped JAX arrays given
    the attributes, and may be overriden if needed.

    * `source (str | tuple[str])` : input representation(s)
    * `batch_size (int)`
    * `n_it (int)` : number of iterations

    See `self.run()` for more details on the benchmarking process.
    """

    keys: tuple[str, ...] = ("e3nn", "e3x", "e3j")

    is_jax: bool = True

    def __init__(
        self,
        source: E3Rep | tuple[E3Rep, ...],
        target: E3Rep,
        batch_size: int = 1024,
        n_it: int = 100,
    ):
        """Define input representations and shapes."""
        try:
            self.source = tuple(self._parse_rep(src) for src in source)
        except ValueError:
            self.source = (self._parse_rep(source),)
        self.target = self._parse_rep(target)
        self.batch_size = batch_size
        self.n_it = n_it
        self.rng = random.key(1234)
        self.n_samp = N_SAMP

    def items(self):
        """Yield library names and implementations."""
        return ((e3, getattr(self, e3)) for e3 in self.keys)

    def metadata(self) -> dict[str, Any]:
        """Return run-specific metadata, e.g. l_max and batch_size."""
        out = {"batch_size": self.batch_size, "out": str(self.target)}
        for i, src_i in enumerate(self.source):
            out[f"in{i+1}"] = str(src_i)
        return out

    def run(
        self,
        jit: bool = True,
        write: bool = True,
        debug: int = 0,
    ) -> dict[str, float]:
        """Loop over timed implementations."""
        metrics_out = defaultdict(lambda: {})
        if write:
            self._print_header()

        is_torch = not (self.is_jax)

        for e3, impl in self.items():
            # wrap/transform raw JAX arrays for e3nn/e3x
            inputs = getattr(self, e3 + "_inputs")()
            # compile and rename implementation
            if jit and is_torch:
                impl = torch.compile(impl)
            elif jit:
                impl = jax.jit(impl)
            impl.__name__ = e3
            metrics = timed_loop(
                impl, self.n_it, write, is_torch=is_torch, debug=debug
            )(*inputs)
            for k, mk in metrics.items():
                metrics_out[k][e3] = mk
            del inputs
            (jax.clear_caches() if not is_torch else torch.cuda.empty_cache())

        if write:
            print("")

        return metrics_out

    def trace(self, logdir: str = "tmp/tensorboard", jit=True, warmup=10):
        """Loop over profiled implementations."""
        # prepare inputs outside of profiling context
        inputs_all = []
        for e3, _impl in self.items():
            inputs = getattr(self, e3 + "_inputs")()
            inputs_all.append(tuple(x[0] for x in inputs))
            del inputs
            jax.clear_caches()

        # compile implementations before profiling
        items = [(e3, jax.jit(impl) if jit else impl) for e3, impl in self.items()]
        for _ in range(warmup):
            for (_e3, impl), xs in zip(items, inputs_all):
                y = impl(*xs)

        with jax.profiler.trace(logdir):
            for (_e3, impl), xs in zip(items, inputs_all):
                print(f"profiling {_e3} on inputs", *(x.shape for x in xs))
                y = impl(*xs)
                try:
                    jax.block_until_ready(y)
                except AttributeError:
                    # y: e3nn.IrrepsArray
                    y.array.block_until_ready()

    @property
    def dim_in(self):
        return sum(src.dim for src in self.e3j.source)

    @property
    def dim_out(self):
        return self.e3j.target.dim

    def num_features(self, argnum: int = 0):
        """Retrieve the multiplicity of one input, constant by e3x constraints."""
        muls = [mul for mul, _ in self.source[argnum]]
        assert min(muls) == max(muls)
        return muls[0]

    @property
    def in_features(self):
        return tuple(self.num_features(i) for i in range(len(self.source)))

    @property
    def out_features(self):
        """Retrieve the output multiplicity, constant by e3x constraints."""
        muls = [mul for mul, _ in self.target]
        assert min(muls) == max(muls)
        return muls[0]

    @property
    def dim_out(self):
        return self.e3j.target.dim

    @property
    def dim_in(self):
        source = self.e3j.source
        if isinstance(source, (e3nn.Irreps, O3Space)):
            return source.dim
        return sum(s.dim for s in source)

    def e3nn_inputs(self) -> tuple[e3nn.IrrepsArray, ...]:
        """Wrap raw JAX inputs into `e3nn.IrrepsArray`."""
        inputs = self.e3j_inputs()
        source = tuple(src._to_e3nn() for src in self.source)
        return tuple(e3nn.IrrepsArray(src, x) for src, x in zip(source, inputs))

    def e3x_inputs(self) -> tuple[jax.Array, ...]:
        """Reshape JAX inputs to `(..., 1, (2l+1)**2, mul)` for e3x."""
        inputs = self.e3j_inputs()
        n_samp, nb = self.n_samp, self.batch_size
        xs = []
        for i, x_raw in enumerate(inputs):
            shape = (n_samp, nb, 1, -1, self.num_features(i))
            xs.append(x_raw.reshape(shape))
        return tuple(xs)

    def e3j_inputs(self) -> tuple[jax.Array, ...]:
        """Prepare raw JAX inputs for e3j."""
        shapes = [(self.n_samp, self.batch_size, src.dim) for src in self.source]
        return tuple(random.normal(self.rng, shape) for shape in shapes)

    def _print_header(self):
        """Print header for CLI/text output."""
        inputs = self.e3j_inputs()
        print("=" * 6 + f" {self} " + 6 * "=")
        print("")
        print(f"on: {jax.devices()}", end="\t")
        print("inputs:", *(inpt.shape[1:] for inpt in inputs), sep=" ")
        print("")

    def _parse_rep(self, rep: E3Rep) -> O3Space:
        if isinstance(rep, str):
            return O3Space(rep)
        elif isinstance(rep, tuple) and len(rep) == 2:
            mul, l_max = rep
            return O3Space([(mul, (l, (-1) ** l)) for l in range(l_max + 1)])
        raise ValueError(f"Expected `str` or `tuple[int, int]`, got {rep}")


class E3FlaxBenchmark(E3Benchmark):
    """Base class for E3-equivariant flax modules benchmarks.

    Subclass this module by implementing

    * `self.e3?_module()` as a cached property

    The inherited `self.e3?_params` and `self.e3?_inputs` methods
    should do the rest of the job.

    Note
    ----
    We use `cached_property` to initialize flax parameters instead
    of setting attributes inside `__init__`, to avoid any errors in
    case one of the E3-modules is not defined (therefore excluded
    from from the library keys).
    """

    # --- Module calls ---

    @cached_property
    def e3nn(self) -> jax.Array:
        return partial(self.e3nn_module.apply, self.e3nn_params)

    @cached_property
    def e3x(self) -> jax.Array:
        return partial(
            self.e3x_module.apply,
            self.e3x_params,
        )

    @cached_property
    def e3j(self) -> jax.Array:
        return partial(
            self.e3j_module.apply,
            self.e3j_params,
        )

    # --- Parameters ---

    @cached_property
    def e3nn_params(self):
        inputs = self.e3nn_inputs()
        return self.e3nn_module.init(self.rng, *(x[0] for x in inputs))

    @cached_property
    def e3x_params(self):
        inputs = self.e3x_inputs()
        return self.e3x_module.init(self.rng, *(x[0] for x in inputs))

    @cached_property
    def e3j_params(self):
        inputs = self.e3j_inputs()
        return self.e3j_module.init(self.rng, *(x[0] for x in inputs))
