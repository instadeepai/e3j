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

from functools import cached_property

import e3nn_jax as e3nn
import e3x

from e3j.linen.linear import Linear

from .e3_benchmark import E3FlaxBenchmark

L_MAX = 4
BATCH_SIZE = 1024
ITERATIONS = 100


class LinearBenchmark(E3FlaxBenchmark):

    keys = ("e3x", "e3nn", "e3j")

    @cached_property
    def e3nn_module(self):
        return e3nn.flax.Linear(
            irreps_in=self.source[0]._to_e3nn(),
            irreps_out=self.target._to_e3nn(),
        )

    @cached_property
    def e3j_module(self):
        return Linear(str(self.source[0]), str(self.target))

    @cached_property
    def e3x_module(self):
        return e3x.nn.Dense(self.out_features, use_bias=False)

    def __repr__(self):
        return f"Linear: {self.source[0]} → {self.target}"


if __name__ == "__main__":

    benchmark = LinearBenchmark(
        "128x0e + 128x1o + 128x2e + 128x3o + 128x4e",
        "128x0e + 128x1o + 128x2e + 128x3o + 128x4e",
        BATCH_SIZE,
        ITERATIONS,
    )
    benchmark.run()
