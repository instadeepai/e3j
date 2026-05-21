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
import os

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
except ModuleNotFoundError:
    pass

try:
    import e3nn as e3nn_t
    import torch

except ModuleNotFoundError:

    class torch:
        Tensor = type


import e3nn_jax as e3nn
import e3x
import jax
import jax.numpy as np
import numpy

from e3j.core.tensor_product import TensorProduct
from e3j.utils import config

from .e3_benchmark import E3Benchmark

L_MAX = 4
MUL = 32
ITERATIONS = 100
BATCH_SIZE = 8192  # 8192  # 131072  # 262144 #16384  # 8192 #4096 #2048
PROFILE = False

# Note: export JAX_PLATFORMS='cpu' if torch claims GPU memory.
BACKEND = (
    os.environ.get("E3J_BACKEND")
    if os.environ.get("E3J_BACKEND") in ("jax", "torch")
    else "jax"
)

DEVICE = "cuda"  # if torch.cuda.is_available() else "cpu"

OUT = "0e + 1o + 2e + 3o"  # + 4e + 5o + 6e"
IN1 = "0e + 1o + 2e + 3o"
IN2 = "0e + 1o + 2e + 3o"

UNROLL = (1, 8, 8)

IrrepsArray = e3nn.IrrepsArray


def as_tensor(x: np.ndarray) -> torch.Tensor:
    """Cast jax.Array to torch.Tensor."""
    return torch.as_tensor(numpy.asarray(x).copy())


class TensorProductBenchmark(E3Benchmark):
    keys = (
        # "e3nn", "e3nn_f32",
        "e3j_ops",
        "cuequivariance_jax",
    )

    def __init__(
        self,
        source: E3Rep | tuple[E3Rep, ...],
        target: E3Rep,
        mul: int,
        batch_size: int = 1024,
        n_it: int = 100,
    ):
        super().__init__(source, target, batch_size, n_it)
        self.mul = mul

    @functools.cached_property
    def e3j(self):
        with config.use(tensor_product="SPARSE", aggregation="SCATTER"):
            source = self.mul * self.source[0], self.source[1]
            return TensorProduct(
                source,
                self.target,
                layout="E3NN",
            )

    @functools.cached_property
    def e3j_ops_leading(self):
        with config.use(tensor_product="FUSED"):
            return TensorProduct(
                self.source, self.target, unroll=UNROLL, layout="LEADING_CHANNELS"
            )

    @functools.cached_property
    def e3j_ops(self):
        with config.use(tensor_product="FUSED"):
            return TensorProduct(
                self.source,
                self.target,
                unroll=UNROLL,
                layout="TRAILING_CHANNELS",
            )

    def e3j_inputs(self):
        key = jax.random.key(123)
        ns, nb = self.n_samp, self.batch_size
        mul = self.mul
        dim_x, dim_y = (src.dim for src in self.source)
        x = jax.random.normal(key, (ns, nb, mul * dim_x))
        y = jax.random.normal(key, (ns, nb, dim_y))
        return (x, y)

    def e3j_ops_leading_inputs(self):
        key = jax.random.key(123)
        ns, nb = self.n_samp, self.batch_size
        mul = self.mul
        dim_x, dim_y = (src.dim for src in self.e3j_ops_leading.source)
        x = jax.random.normal(key, (ns, nb, mul, dim_x))
        y = jax.random.normal(key, (ns, nb, dim_y))
        return (x, y)

    def e3j_ops_inputs(self):
        key = jax.random.key(123)
        ns, nb = self.n_samp, self.batch_size
        mul = self.mul
        dim_x, dim_y = (src.dim for src in self.e3j_ops.source)
        x = jax.random.normal(key, (ns, nb, dim_x, mul))
        y = jax.random.normal(key, (ns, nb, dim_y))
        return (x, y)

    @functools.cached_property
    def cuequivariance_jax(self):
        import cuequivariance as cue
        import cuequivariance_jax as cuex
        import jax.numpy as jnp

        with cue.assume(layout=cue.ir_mul):
            src0 = cue.Irreps("O3", str(self.mul * self.source[0]))
            src1 = cue.Irreps("O3", str(self.source[1]))
            filter = cue.Irreps("O3", str(self.target))
            descriptor = cue.descriptors.full_tensor_product(src0, src1, filter)

            def call(x: jax.Array, y: jax.Array) -> jax.Array:
                with cue.assume(layout=cue.ir_mul):
                    x_rep = cuex.RepArray(src0, x)
                    y_rep = cuex.RepArray(src1, y)
                    result = cuex.equivariant_polynomial(
                        descriptor,
                        [x_rep, y_rep],
                        jax.ShapeDtypeStruct((x.shape[0], -1), jnp.float32),
                        method="uniform_1d",
                    )
                    result = result[0] if isinstance(result, list) else result
                    return result.array

            return call

    def cuequivariance_jax_inputs(self):
        return self.e3j_inputs()

    @functools.cached_property
    def e3x_module(self):
        return e3x.nn.Tensor(
            max_degree=self.target.l_max,
            include_pseudotensors=False,
        )

    @functools.cached_property
    def e3x(self):
        inputs = self.e3x_inputs()
        module = self.e3x_module
        params = module.init(self.rng, *(x[0] for x in inputs))
        return functools.partial(module.apply, params)

    @functools.cached_property
    def e3nn(self):
        return functools.partial(
            e3nn.tensor_product,
            filter_ir_out=self.target._to_e3nn(),
            irrep_normalization="component",
        )

    def e3nn_inputs(self):
        key = jax.random.key(123)
        ns, nb = self.n_samp, self.batch_size
        mul = self.mul
        source = tuple(src._to_e3nn() for src in self.source)
        dim_x, dim_y = (src.dim for src in source)
        x = jax.random.normal(key, (ns, nb, mul * dim_x))
        y = jax.random.normal(key, (ns, nb, dim_y))
        return (
            e3nn.IrrepsArray(mul * source[0], x),
            e3nn.IrrepsArray(source[1], y),
        )

    @functools.cached_property
    def e3nn_f32(self):
        def tensor_product(
            x: e3nn.IrrepsArray, y: e3nn.IrrepsArray
        ) -> e3nn.IrrepsArray:
            with jax.default_matmul_precision("float32"):
                return functools.partial(
                    e3nn.tensor_product,
                    filter_ir_out=self.target._to_e3nn(),
                    irrep_normalization="component",
                )(x, y)

        return tensor_product

    def e3nn_f32_inputs(self):
        return self.e3nn_inputs()

    def __str__(self):
        return f"{self.source[0]} x {self.source[1]} -> {self.target}"

    @property
    def nnz_and_dim_out(self):
        tp_core = self.e3j
        nnz_and_dim_out_dict = {
            "dim_out": tp_core.target.dim,
            "dim_in": sum(src.dim for src in tp_core.source),
            "nnz": tp_core.nnz,
            "nnz_ratio": tp_core.nnz_ratio,
        }
        return nnz_and_dim_out_dict

    @property
    def e3x_dim_out(self):
        xs = tuple(x[0] for x in self.e3x_inputs())
        out = self.e3x(*xs)
        return out.shape[-2]


class TorchTensorProductBenchmark(TensorProductBenchmark):

    keys = ("e3nn_torch", "torch_cuet")

    is_jax = False

    @functools.cached_property
    def torch_cuet(self):
        src = tuple(cue.Irreps("O3", str(s)) for s in self.source)
        descriptor = cue.descriptors.full_tensor_product(*src)
        tp = cuet.EquivariantTensorProduct(
            descriptor,
            layout=cue.ir_mul,
            device=DEVICE,
            use_fallback=False,
        )
        return tp

    def torch_cuet_inputs(self):
        return self._torch_inputs()

    @functools.cached_property
    def e3nn_torch(self):
        return e3nn_t.o3.FullTensorProduct(
            irreps_in1=str(self.source[0]),
            irreps_in2=str(self.source[1]),
        ).to(DEVICE)

    def _torch_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(as_tensor(x).to(DEVICE) for x in self.e3j_inputs())

    def e3nn_torch_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._torch_inputs()

    def __str__(self):
        return f"{self.source[0]} ⊗  {self.source[1]} → {self.target}"

    @property
    def nnz_and_dim_out(self):
        tp_core = TensorProduct(self.source, self.target)
        nnz_and_dim_out_dict = {
            "dim_out": tp_core.shape[0],
            "nnz": tp_core.nnz,
            "nnz_ratio": tp_core.nnz_ratio,
            "source": tuple(str(si) for si in tp_core.source),
            "target": str(tp_core.target),
        }
        return nnz_and_dim_out_dict

    @property
    def dim_in(self):
        return self.source[0].dim + self.source[1].dim

    @property
    def dim_out(self):
        return self.target.dim

    @property
    def e3x_dim_out(self):
        xs = tuple(x[0] for x in self.e3x_inputs())
        out = self.e3x(*xs)
        return out.shape[-2]


class TensorProductGradBenchmark(TensorProductBenchmark):

    keys = (
        "e3nn",
        "e3nn_f32",
        "e3j_ops",
        "cuequivariance_jax",
    )

    @staticmethod
    def differentiate(f):
        """Return df: (x, y) -> ((dx, dy), z) from binary f: (x, y) -> z."""

        def f_aux(x, y):
            z = f(x, y)
            return (
                np.sum(z) if isinstance(z, jax.Array) else np.sum(z.array),
                z,
            )

        df_aux = jax.grad(f_aux, argnums=(0, 1), has_aux=True)

        def df(x, y):
            (dx, dy), z = df_aux(x, y)
            return (dx, dy, z)

        if hasattr(f, "source") and hasattr(f, "target"):
            df.source = f.source
            df.target = (*f.source, f.target)

        return df

    @functools.cached_property
    def e3j(self):
        primitive = super().e3j
        return self.differentiate(primitive)

    @functools.cached_property
    def e3j_ops_leading(self):
        primitive = super().e3j_ops_leading
        return self.differentiate(primitive)

    @functools.cached_property
    def e3j_ops(self):
        primitive = super().e3j_ops
        return self.differentiate(primitive)

    @functools.cached_property
    def e3nn(self):
        primitive = super().e3nn
        return self.differentiate(primitive)

    @functools.cached_property
    def e3nn_f32(self):
        primitive = super().e3nn_f32
        df = jax.grad(
            lambda x, y: np.sum(primitive(x, y).array),
            argnums=(0, 1),
        )
        return df

    def e3nn_inputs(self):
        return super().e3nn_inputs()

    def e3nn_f32_inputs(self):
        return super().e3nn_inputs()

    def e3j_inputs(self):
        return super().e3j_inputs()

    def e3j_ops_leading_inputs(self):
        return super().e3j_ops_leading_inputs()

    def e3j_ops_inputs(self):
        return super().e3j_ops_inputs()

    @property
    def dim_in(self):
        return super().dim_in

    @property
    def dim_out(self):
        return super().dim_in + super().dim_out

    def __str__(self):
        return "Bwd " + super().__str__()


if __name__ == "__main__":
    Benchmark = (
        TensorProductBenchmark if BACKEND == "jax" else TorchTensorProductBenchmark
    )
    benchmark = Benchmark(
        (IN1, IN2),
        OUT,
        mul=MUL,
        batch_size=BATCH_SIZE,
        n_it=ITERATIONS,
    )
    benchmark_grad = TensorProductGradBenchmark(
        (IN1, IN2),
        OUT,
        mul=MUL,
        batch_size=BATCH_SIZE,
        n_it=ITERATIONS,
    )
    # benchmark_grad.trace("tmp/tensor-product")
    # benchmark.run(jit=True)  # (BACKEND != "torch"))
    benchmark_grad.run(jit=True)
    # benchmark.trace("tmp/tensor_product")
