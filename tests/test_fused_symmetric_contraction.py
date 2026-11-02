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

"""Every fused SymmetricContraction, through the core and linen modules, against `UNFUSED`."""

import jax
import jax.numpy as jnp
import numpy.testing as testing
import pytest
from conftest import skip_unless_hopper, skip_unless_tpu

from e3j.core.symmetric_contraction import SymmetricContraction
from e3j.linen.symmetric_contraction import SymmetricContraction as LinenModule
from e3j.spaces import O3Space
from e3j.utils.config import Config
from e3j.utils.options import SymmetricContraction as Algorithm

TPU = Algorithm.FUSED_MOSAIC_TPU
GPU = Algorithm.FUSED_MOSAIC_GPU
MARKS = {TPU: pytest.mark.mosaic_tpu, GPU: pytest.mark.mosaic_gpu}
REQUIRE = {TPU: skip_unless_tpu, GPU: skip_unless_hopper}
TOLERANCE = {TPU: 2e-3, GPU: 1e-3}

# (source, correlation, target)
CASES = [
    ("0e + 1o", 2, "0e + 1o + 2e"),
    ("0e + 1o + 2e", 2, "0e + 1o + 2e"),
    ("0e + 1o", 3, "0e + 1o"),
    ("0e + 1o + 2e + 3o", 3, "0e + 1o"),
]
# (batch, mul, num_species): unaligned batches and sub-128 mul exercise padding
TPU_SHAPES = [(256, 128, 4), (16, 64, 2), (37, 256, 8)]
# (source, correlation, target, num_species, mul, batch)
GPU_CASES = [
    ("0e+1o+2e", 1, "0e+1o", 4, 128, 37),
    ("0e+1o+2e", 2, "0e+1o", 4, 128, 300),
    ("0e+1o+2e", 3, "0e+1o", 4, 128, 64),
    ("0e+1o+2e+3o", 3, "0e+1o+2e", 5, 256, 40),
    # every power is pure `0e`: its blocks sit side by side in `sigma`
    ("0e+1o+2e+3o", 3, "0e", 3, 128, 64),
    ("1x0e+2x1o+1x2e", 2, "0e+1o", 3, 128, 96),
    ("0e+1o", 4, "0e+1o", 2, 128, 17),
    ("0e+1o+2e+3o", 3, "0e+1o", 3, 128, 50),
]


def _param(algorithm, *values, marks=True):
    case_id = "-".join([algorithm.name.lower(), *(str(v) for v in values)])
    return pytest.param(
        algorithm,
        *values,
        marks=MARKS[algorithm] if marks else (),
        id=case_id.replace(" ", ""),
    )


# (algorithm, source, correlation, target, num_species, mul, batch)
DISPATCH_CASES = [
    _param(TPU, *case, num_species, mul, batch)
    for case in CASES
    for batch, mul, num_species in TPU_SHAPES
] + [_param(GPU, *case) for case in GPU_CASES]

# (algorithm, source, correlation, target, mul): GPU channels are multiples of 128
SECOND_ORDER_CASES = [_param(TPU, *case, 64) for case in CASES] + [
    _param(GPU, *case, 128) for case in CASES
]


@pytest.fixture
def require_backend(algorithm):
    REQUIRE[algorithm]()


def _core(source, correlation, target, num_species, mul, algorithm, layout=None):
    return SymmetricContraction(
        source,
        correlation,
        target,
        num_species,
        mul,
        layout or "TRAILING_CHANNELS",
        config=Config(symmetric_contraction=algorithm),
    )


def _inputs(core, batch, seed=0):
    """`(features, indices, weight)` for `core`, weights at unit scale."""
    kx, ki, kw = jax.random.split(jax.random.key(seed), 3)
    features = jax.random.normal(kx, (batch, core.source.dim, core.num_channels))
    indices = jax.random.choice(ki, core.num_species, (batch,))
    shape = (core.num_species, core.num_channels)
    weight = [
        jax.random.normal(k, (*shape, *b.shape))
        for k, b in zip(
            jax.random.split(kw, len(core.linear_blocks)), core.linear_blocks
        )
    ]
    return features, indices, weight


@pytest.mark.usefixtures("require_backend")
@pytest.mark.parametrize(
    ("algorithm", "source", "correlation", "target", "num_species", "mul", "batch"),
    DISPATCH_CASES,
)
def test_core_dispatch_matches_unfused(
    algorithm, source, correlation, target, num_species, mul, batch
):
    """Outputs and first-order gradients match the unfused core."""
    structure = (source, correlation, target, num_species, mul)
    features, indices, weight = _inputs(_core(*structure, Algorithm.UNFUSED), batch)

    def run(algorithm):
        core = _core(*structure, algorithm)
        loss = lambda f, w: jnp.sum(core(f, indices, w) ** 2)
        return core(features, indices, weight), jax.grad(loss, (0, 1))(features, weight)

    tol = TOLERANCE[algorithm]
    result, grads = run(algorithm)
    expect, grads_ref = run(Algorithm.UNFUSED)
    testing.assert_allclose(result, expect, rtol=tol, atol=tol)
    for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(grads_ref)):
        testing.assert_allclose(a, b, rtol=tol, atol=tol)


@pytest.mark.usefixtures("require_backend")
@pytest.mark.parametrize(
    ("algorithm", "source", "correlation", "target", "mul"), SECOND_ORDER_CASES
)
def test_core_second_order_matches_unfused(algorithm, source, correlation, target, mul):
    """Weight gradient of a feature-gradient loss, as in force training."""
    structure = (source, correlation, target, 3, mul)
    features, indices, weight = _inputs(_core(*structure, Algorithm.UNFUSED), 40)

    def run(algorithm):
        core = _core(*structure, algorithm)
        energy = lambda f, w: jnp.sum(core(f, indices, w) ** 2)
        forces = jax.grad(energy)
        return jax.grad(lambda w: jnp.sum(forces(features, w) ** 2))(weight)

    # fourth power of unit-scale weights: compare relative to the largest entry
    tol = TOLERANCE[algorithm]
    for a, b in zip(run(algorithm), run(Algorithm.UNFUSED)):
        scale = jnp.abs(b).max()
        testing.assert_allclose(a / scale, b / scale, rtol=tol, atol=tol)


@pytest.mark.usefixtures("require_backend")
@pytest.mark.parametrize(("algorithm", "mul"), [_param(TPU, 32), _param(GPU, 128)])
def test_core_batch_axes(algorithm, mul):
    structure = ("0e + 1o", 2, "0e + 1o", 2, mul)
    features, indices, weight = _inputs(_core(*structure, Algorithm.UNFUSED), 24)
    features = features.reshape(2, 12, *features.shape[1:])
    indices = indices.reshape(2, 12)
    result = _core(*structure, algorithm)(features, indices, weight)
    expect = _core(*structure, Algorithm.UNFUSED)(features, indices, weight)
    assert result.shape == expect.shape
    tol = TOLERANCE[algorithm]
    testing.assert_allclose(result, expect, rtol=tol, atol=tol)


@pytest.mark.usefixtures("require_backend")
@pytest.mark.parametrize(("algorithm", "mul"), [_param(TPU, 64), _param(GPU, 128)])
def test_linen_module_config(algorithm, mul):
    """The linen module carries the option through its config."""
    source, target = "0e + 1o + 2e", "0e + 1o"
    feats = jax.random.normal(jax.random.key(0), (32, O3Space(source).dim, mul))
    index = jax.random.choice(jax.random.key(1), 4, (32,))

    def module(algorithm):
        return LinenModule(
            source,
            3,
            target,
            4,
            mul,
            "TRAILING_CHANNELS",
            config=Config(symmetric_contraction=algorithm),
        )

    variables = module(Algorithm.UNFUSED).init(jax.random.key(2), feats, index)
    result = module(algorithm).apply(variables, feats, index)
    expect = module(Algorithm.UNFUSED).apply(variables, feats, index)
    tol = TOLERANCE[algorithm]
    testing.assert_allclose(result, expect, rtol=tol, atol=tol)


@pytest.mark.parametrize("algorithm", [_param(a, marks=False) for a in MARKS])
def test_leading_channels_raises(algorithm):
    core = _core("0e + 1o", 2, "0e + 1o", 2, 8, algorithm, "LEADING_CHANNELS")
    features = jnp.zeros((4, 8, O3Space("0e + 1o").dim))
    weight = [jnp.zeros((2, 8, *b.shape)) for b in core.linear_blocks]
    with pytest.raises(NotImplementedError, match="TRAILING_CHANNELS"):
        core(features, jnp.zeros(4, dtype=jnp.int32), weight)
