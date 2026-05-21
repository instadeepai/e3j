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

from e3j.core.bigotimes import Bigotimes

from .e3_benchmark import E3Benchmark


class BigotimesBenchmark(E3Benchmark):

    keys = ("e3j",)

    @cached_property
    def e3j(self):
        return Bigotimes(
            self.target,
            self.source,
        )


if __name__ == "__main__":

    benchmark = BigotimesBenchmark(
        ("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e"),
        "0e + 1o + 2e",
        batch_size=10000,
        n_it=50,
    )

    benchmark.run()
