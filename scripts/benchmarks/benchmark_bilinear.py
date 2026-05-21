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

import itertools
from functools import cached_property

from .backends import cue, cuet
from .backends import e3nn_torch as e3nn_t
from .backends import openeq, torch
from .e3_benchmark import E3Benchmark


class TorchBilinearBenchmark(E3Benchmark):

    keys = ("e3nn_torch", "torch_openeq", "torch_cuet")

    is_jax = False

    @cached_property
    def e3nn_torch(self):
        X_ir, Y_ir = (e3nn_t.o3.Irreps(str(src)) for src in self.source)
        Z_ir = e3nn_t.o3.Irreps(str(self.target))
        return e3nn_t.o3.FullyConnectedTensorProduct(
            X_ir,
            Y_ir,
            Z_ir,
            internal_weights=False,
            shared_weights=True,
        ).cuda()

    def e3nn_torch_inputs(self):
        gen = torch.Generator(device="cuda")
        dim_X, dim_Y = (src.dim for src in self.source)
        nb = self.batch_size
        ns = self.n_samp
        X = torch.rand(ns, nb, dim_X, device="cuda", generator=gen)
        Y = torch.rand(ns, nb, dim_Y, device="cuda", generator=gen)
        W = torch.rand(ns, self.e3nn_torch.weight_numel, device="cuda", generator=gen)
        return (X, Y, W)

    @cached_property
    def torch_openeq(self):
        X_ir, Y_ir = (e3nn_t.o3.Irreps(str(src)) for src in self.source)
        Z_ir = e3nn_t.o3.Irreps(str(self.target))
        instructions = [
            (ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight)
            for ins in self.e3nn_torch.instructions
        ]
        problem = openeq.TPProblem(
            X_ir, Y_ir, Z_ir, instructions, internal_weights=False
        )
        return openeq.TensorProduct(problem, torch_op=True)

    def torch_openeq_inputs(self):
        return self.e3nn_torch_inputs()

    @cached_property
    def torch_cuet(self):
        source = tuple(cue.Irreps("O3", str(src)) for src in self.source)
        target = cue.Irreps("O3", str(self.target))
        return cuet.FullyConnectedTensorProduct(
            *source,
            target,
            internal_weights=False,
            shared_weights=True,
            layout=cue.ir_mul,
        ).cuda()

    def torch_cuet_inputs(self):
        (X, Y, W) = self.e3nn_torch_inputs()
        return (X, Y, W.reshape(self.n_samp, 1, -1))


if __name__ == "__main__":
    benchmark = TorchBilinearBenchmark(
        ("8x0e + 8x1o + 8x2e", "8x0e + 8x1o + 8x2e"),
        "0e + 1o + 2e",
    )
    benchmark.run(jit=False)
