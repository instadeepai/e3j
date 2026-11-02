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

"""The fused Mosaic GPU SymmetricContraction kernel must match `UNFUSED`.

Every comparison runs through the public module, so it exercises the whole
chain: every power's Clebsch-Gordan contraction, the block assembly plan that
stands in for `Permutation`, and the final index-wise linear mix. The kernel's
tuning knobs are pinned on its params (`_PinnedSymmetricContraction`), and
every result is compared against `UNFUSED` (and its autodiff, once and twice:
training on forces).

The kernel needs a Hopper (sm_90) or newer CUDA device to run, so the whole
module skips elsewhere: Pallas interpret mode has no discharge rule for Mosaic
GPU's async copies, so there is no interpret-mode tier to fall back on.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import skip_unless_hopper

from e3j.core.symmetric_contraction import SymmetricContraction
from e3j.pallas_ops.symmetric_contraction import (
    PallasMosaicGPUSymmetricContractionConfig,
)
from e3j.pallas_ops.symmetric_contraction.mosaic_gpu.primitives import (
    chain_supported,
    double_backward_supported,
    symmetric_contraction_pallas_mosaic_gpu,
)
from e3j.utils import config, options

pytestmark = pytest.mark.mosaic_gpu

RTOL = 1e-3
ATOL = 1e-3


@pytest.fixture(autouse=True)
def _require_hopper():
    skip_unless_hopper()


class _PinnedSymmetricContraction(SymmetricContraction):
    """The module with its Mosaic GPU kernel's tuning knobs pinned: the
    module itself leaves them at the kernel's defaults."""

    def __init__(self, *args, mgpu_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.mgpu_config = mgpu_config

    @functools.cached_property
    def mosaic_gpu_params(self):
        params = super().mosaic_gpu_params
        return dataclasses.replace(params, config=self.mgpu_config)


def _core(
    source_irreps,
    correlation,
    keep_irrep_out,
    num_species,
    num_channels,
    strategy=options.SymmetricContraction.FUSED_MOSAIC_GPU,
    **mgpu_config,
):
    with config.use(symmetric_contraction=strategy):
        return _PinnedSymmetricContraction(
            source_irreps=source_irreps,
            correlation=correlation,
            keep_irrep_out=keep_irrep_out,
            num_species=num_species,
            num_channels=num_channels,
            layout="TRAILING_CHANNELS",
            mgpu_config=PallasMosaicGPUSymmetricContractionConfig(**mgpu_config),
        )


def _weights(blocks, num_species, num_channels, seed):
    key = jax.random.key(seed)
    return [
        jax.random.normal(
            jax.random.fold_in(key, i),
            (num_species, num_channels, *block.shape),
            dtype=jnp.float32,
        )
        for i, block in enumerate(blocks)
    ]


def _case(
    source_irreps,
    correlation,
    keep_irrep_out,
    num_species,
    num_channels,
    batch,
    index=None,
    **mgpu_config,
):
    """`(fused, plain, feat, index, weights)`: the fused module with
    `mgpu_config`, its `UNFUSED` twin, and random operands."""
    structure = (source_irreps, correlation, keep_irrep_out, num_species, num_channels)
    fused = _core(*structure, **mgpu_config)
    plain = _core(*structure, options.SymmetricContraction.UNFUSED)
    weights = _weights(fused.linear_blocks, num_species, num_channels, seed=1)
    feat = jax.random.normal(
        jax.random.key(2), (batch, fused.source.dim, num_channels), dtype=jnp.float32
    )
    if index is None:
        index = jax.random.randint(
            jax.random.key(3), (batch,), 0, num_species, dtype=jnp.int32
        )
    return fused, plain, feat, jnp.asarray(index), weights


CONFIGS = [
    # (source_irreps, correlation, keep_irrep_out, num_species, num_channels, batch)
    ("0e+1o+2e", 1, "0e+1o", 4, 128, 37),
    ("0e+1o+2e", 2, "0e+1o", 4, 128, 300),
    ("0e+1o+2e", 3, "0e+1o", 4, 128, 64),
    ("0e+1o+2e+3o", 3, "0e+1o+2e", 5, 256, 40),
    # Every power is pure `0e`: its blocks sit side by side in `sigma`.
    ("0e+1o+2e+3o", 3, "0e", 3, 128, 64),
    ("1x0e+2x1o+1x2e", 2, "0e+1o", 3, 128, 96),
    ("0e+1o", 4, "0e+1o", 2, 128, 17),
]
CONFIG_IDS = [f"corr{c}_ns{ns}_nc{nc}" for _, c, _, ns, nc, _ in CONFIGS]


def _assert_matches(fused, plain, feat, index, weights):
    got = jax.jit(fused)(feat, index, weights)
    expected = jax.jit(plain)(feat, index, weights)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", CONFIGS, ids=CONFIG_IDS)
def test_matches_unfused(case):
    _assert_matches(*_case(*case))


def test_class_dispatch_unsupported_channels_raises():
    """A channel count that is not a positive multiple of 128 must be refused."""
    fused = _core("0e+1o", 2, "0e+1o", 4, 64)
    weights = _weights(fused.linear_blocks, 4, 64, seed=0)
    feat = jnp.zeros((8, fused.source.dim, 64), dtype=jnp.float32)
    index = jnp.zeros((8,), jnp.int32)

    with pytest.raises(NotImplementedError, match="channels"):
        fused(feat, index, weights)


@pytest.mark.parametrize(
    "mgpu_config",
    [{"warpgroups": 1}, {"warpgroups": 3, "pipeline_depth": 2}, {"pipeline_depth": 1}],
    ids=["one_warpgroup", "three_warpgroups", "depth_one"],
)
def test_chain_launch_matches_unfused(mgpu_config):
    """The chain forward's warpgroups and depth change nothing but speed."""
    _assert_matches(*_case("0e+1o+2e+3o", 3, "0e+1o", 4, 128, 301, **mgpu_config))


def test_tables_over_shared_memory_run_reference():
    """89 species of 29 folded weight rows (1.3 MB per table) do not fit in
    shared memory: the whole call runs the unfused reference."""
    case = _case("0e+1o+2e+3o", 3, "0e+1o", 89, 128, 300)
    assert not chain_supported(case[0].mosaic_gpu_params, 300, 128, "fwd")
    _assert_matches(*case)
    _assert_grads_match(*case)


def test_tables_over_shared_memory_without_reference_raises():
    fused, _, feat, index, weights = _case("0e+1o+2e+3o", 3, "0e+1o", 89, 128, 300)
    with pytest.raises(NotImplementedError, match="no `reference`"):
        symmetric_contraction_pallas_mosaic_gpu(
            feat, index, weights, fused.mosaic_gpu_params
        )


# --- Gradients --------------------------------------------------------------
#
# The chain backward (`bwd`), checked against `UNFUSED`'s autodiff.

GRAD_RTOL = 1e-3
GRAD_ATOL = 1e-3


def _assert_grads_match(fused, plain, feat, index, weights):
    cotangent = jax.random.normal(
        jax.random.key(4), jax.eval_shape(plain, feat, index, weights).shape
    )

    def grads(module):
        loss = lambda f, w: jnp.sum(module(f, index, w) * cotangent)  # noqa: E731
        return jax.jit(jax.grad(loss, argnums=(0, 1)))(feat, weights)

    got, expected = grads(fused), grads(plain)
    np.testing.assert_allclose(got[0], expected[0], rtol=GRAD_RTOL, atol=GRAD_ATOL)
    for g, e in zip(got[1], expected[1]):
        np.testing.assert_allclose(g, e, rtol=GRAD_RTOL, atol=GRAD_ATOL)


@pytest.mark.parametrize("case", CONFIGS, ids=CONFIG_IDS)
def test_grad_matches_unfused(case):
    _assert_grads_match(*_case(*case))


def _assert_feature_grads_match(fused, plain, feat, index, weights):
    cotangent = jax.random.normal(
        jax.random.key(4), jax.eval_shape(plain, feat, index, weights).shape
    )

    def grad(module):
        loss = lambda f: jnp.sum(module(f, index, weights) * cotangent)  # noqa: E731
        return jax.jit(jax.grad(loss))(feat)

    np.testing.assert_allclose(grad(fused), grad(plain), rtol=GRAD_RTOL, atol=GRAD_ATOL)


@pytest.mark.parametrize("case", CONFIGS, ids=CONFIG_IDS)
def test_grad_features_only_matches_unfused(case):
    """Forces at fixed weights (e.g. molecular dynamics): the weights are not
    differentiated, and the backward computes `dx` alone."""
    _assert_feature_grads_match(*_case(*case))


def test_chain_tables_over_one_copy_match_unfused():
    """Two channel blocks, and a folded weight table of 6 x 49 rows: more
    than one tensor-map copy moves (`common.table_copies`)."""
    case = _case(
        "0e+1o+2e+3o",
        3,
        "0e+1o+2e",
        6,
        256,
        40,
        channel_block=128,
    )
    _assert_matches(*case)
    _assert_grads_match(*case)


def test_chain_long_walks_match_unfused():
    """Enough rows that each warpgroup of the chain kernels walks tens of
    them, and that the last ones' species blocks (`common.SCALAR_BLOCK`) run
    past the end of the batch."""
    case = _case("0e+1o", 2, "0e+1o", 3, 128, 9001)
    _assert_matches(*case)
    _assert_grads_match(*case)


def test_degenerate_block_is_zeroed():
    """No power of `0e + 1o` up to correlation 2 holds a `2o`: its block has
    no weight, and the kernels zero its span of their one output."""
    case = _case("0e+1o", 2, "0e+1o+2o", 3, 128, 50)
    assert [b.shape[1] for b in case[0].linear_blocks] == [3, 3, 0]
    _assert_matches(*case)
    _assert_grads_match(*case)


def test_float32_operands_under_x64():
    """`jax_enable_x64` must not change what the kernels trace to: it makes
    Python ints int64, and the grid index they split is int32."""
    with jax.enable_x64():
        case = _case("0e+1o+2e+3o", 3, "0e+1o", 3, 128, 50)
        _assert_matches(*case)
        _assert_grads_match(*case)


# --- Second derivatives: training on forces ---------------------------------
#
# A loss on forces, `dE/dx`, differentiated in the inputs and the weights:
# through the chain backward's own derivative (`bwd.chain_double_backward`)
# where it runs, and through the unfused reference's elsewhere.


def _assert_close_to_scale(got, expected):
    """`got` within the gradient tolerances of `expected`, both scaled by
    `expected`'s largest entry: second derivatives span many magnitudes."""
    scale = float(np.max(np.abs(expected), initial=0.0)) or 1.0
    np.testing.assert_allclose(
        np.asarray(got) / scale,
        np.asarray(expected) / scale,
        rtol=GRAD_RTOL,
        atol=GRAD_ATOL,
    )


def _assert_force_grads_match(fused, plain, feat, index, weights, inner=(0,)):
    """The gradients in `(x, W)` of `E + <dE/dx, c_x>` (`+ <dE/dW, c_w>` when
    `inner` holds the weights too), `E = <tanh(s y), c_y>`: nonlinear in `y`,
    so the backward's cotangent depends on the inputs and weights as well."""
    y = jax.jit(plain)(feat, index, weights)
    s = 1.0 / float(jnp.std(y))
    c_y = jax.random.normal(jax.random.key(5), y.shape)
    c_x = jax.random.normal(jax.random.key(6), feat.shape)
    c_w = [
        jax.random.normal(jax.random.fold_in(jax.random.key(7), i), w.shape)
        for i, w in enumerate(weights)
    ]

    def grads(module):
        def energy(f, w):
            return jnp.sum(jnp.tanh(s * module(f, index, w)) * c_y)

        def loss(f, w):
            e, inner_grads = jax.value_and_grad(energy, argnums=inner)(f, w)
            total = e + jnp.sum(inner_grads[0] * c_x)
            if 1 in inner:
                total += sum(jnp.sum(g * c) for g, c in zip(inner_grads[1], c_w))
            return total

        return jax.jit(jax.grad(loss, argnums=(0, 1)))(feat, weights)

    got, expected = grads(fused), grads(plain)
    _assert_close_to_scale(got[0], expected[0])
    for g, e in zip(got[1], expected[1]):
        _assert_close_to_scale(g, e)


@pytest.mark.parametrize("case", CONFIGS, ids=CONFIG_IDS)
def test_force_grad_matches_unfused(case):
    """The chain backward, differentiated by the double backward kernel."""
    fused, *rest = _case(*case)
    _, _, _, num_species, channels, batch = case
    params = fused.mosaic_gpu_params
    assert double_backward_supported(params, batch, channels)
    _assert_force_grads_match(fused, *rest)


@pytest.mark.parametrize("case", CONFIGS[1:4:2], ids=CONFIG_IDS[1:4:2])
def test_force_and_weight_grad_matches_unfused(case):
    """An inner gradient in the weights too: its cotangent goes through a
    forward and a first backward with it for weights."""
    _assert_force_grads_match(*_case(*case), inner=(0, 1))


@pytest.mark.parametrize("case", CONFIGS[1:4:2], ids=CONFIG_IDS[1:4:2])
def test_hessian_at_fixed_weights_matches_unfused(case):
    """`d<dE/dx, c>/dx` with the weights closed over: the double backward
    leaves out their cotangent tables."""
    fused, plain, feat, index, weights = _case(*case)
    c_x = jax.random.normal(jax.random.key(6), feat.shape)

    def hvp(module):
        energy = lambda f: jnp.sum(module(f, index, weights) ** 2)  # noqa: E731
        loss = lambda f: jnp.sum(jax.grad(energy)(f) * c_x)  # noqa: E731
        return jax.jit(jax.grad(loss))(feat)

    _assert_close_to_scale(hvp(fused), hvp(plain))


def test_force_grad_where_double_backward_does_not_fit_matches_unfused():
    """Two warpgroups 8 rows deep fit the chain backward's rows, not the
    double backward's (a row of `u` more): the reference differentiates the
    chain backward instead."""
    case = ("0e+1o+2e+3o", 2, "0e", 7, 128, 301)
    fused, *rest = _case(*case, warpgroups=2, pipeline_depth=8)
    params = fused.mosaic_gpu_params
    assert chain_supported(params, 301, 128, "bwd")
    assert not double_backward_supported(params, 301, 128)
    _assert_force_grads_match(fused, *rest)


# 73 species fit the forward's resident table, not the backward's weight
# cotangent tables besides it. Forces at fixed weights need no cotangent
# tables, so their backward fits.
BACKWARD_OVER_SHARED_MEMORY = ("0e+1o", 2, "0e+1o+2e", 73, 128, 300)


def test_backward_tables_over_shared_memory_run_reference():
    """The forward runs fused, and the reference differentiates it."""
    case = _case(*BACKWARD_OVER_SHARED_MEMORY)
    params = case[0].mosaic_gpu_params
    assert chain_supported(params, 300, 128, "fwd")
    assert not chain_supported(params, 300, 128, "bwd")
    assert chain_supported(params, 300, 128, "bwd", with_dw=False)
    _assert_matches(*case)
    _assert_grads_match(*case)
    _assert_feature_grads_match(*case)
    _assert_force_grads_match(*case, inner=(0, 1))


def test_force_grad_degenerate_block_matches_unfused():
    _assert_force_grads_match(*_case("0e+1o", 2, "0e+1o+2o", 3, 128, 50), inner=(0, 1))


def test_second_derivative_without_reference_raises():
    """Called without a `reference`, a chain backward whose double backward
    does not fit refuses to be differentiated rather than fail inside JAX."""
    fused, _, feat, index, weights = _case(
        "0e+1o+2e+3o", 2, "0e", 7, 128, 301, warpgroups=2, pipeline_depth=8
    )
    params = fused.mosaic_gpu_params
    assert not double_backward_supported(params, 301, 128)

    def energy(f):
        y = symmetric_contraction_pallas_mosaic_gpu(f, index, weights, params)
        return jnp.sum(y**2)

    hvp = jax.grad(lambda f: jnp.sum(jax.grad(energy)(f)))
    with pytest.raises(NotImplementedError, match="no `reference`"):
        jax.eval_shape(hvp, feat)


def test_backward_over_shared_memory_without_reference_raises_on_weights():
    """Without a `reference`, the forward and forces at fixed weights still
    run fused: only the gradient in the weights refuses."""
    fused, _, feat, index, weights = _case(*BACKWARD_OVER_SHARED_MEMORY)
    params = fused.mosaic_gpu_params

    def energy(f, w):
        y = symmetric_contraction_pallas_mosaic_gpu(f, index, w, params)
        return jnp.sum(y**2)

    jax.eval_shape(energy, feat, weights)
    jax.eval_shape(jax.grad(energy), feat, weights)
    with pytest.raises(NotImplementedError, match="no `reference`"):
        jax.eval_shape(jax.grad(energy, argnums=1), feat, weights)


def _stack_lanes(v, feat, index, weights, num_species):
    """`v` distinct lanes of each operand."""
    feats = jnp.stack([feat * (1 + 0.1 * i) for i in range(v)])
    indices = jnp.stack([(index + i) % num_species for i in range(v)])
    ws = [jnp.stack([w * (1 - 0.05 * i) for i in range(v)]) for w in weights]
    return feats, indices, ws


def _assert_vmapped_force_grads_match(fused, plain, args, in_axes):
    """`(E, dE/dx, dE/dW)` and the gradient of `E + <dE/dx, c>` in `(x, W)`,
    vmapped over `in_axes`: the forward, backward and double backward all
    run with the vmap axis folded in."""
    c_x = jax.random.normal(jax.random.key(6), args[0].shape[-3:])

    def run(module):
        def energy(f, index, w):
            return jnp.sum(jnp.sin(module(f, index, w)))

        def loss(f, index, w):
            e, (dx, dw) = jax.value_and_grad(energy, argnums=(0, 2))(f, index, w)
            return e + jnp.sum(dx * c_x), (e, dx, dw)

        grads = jax.grad(loss, argnums=(0, 2), has_aux=True)
        return jax.jit(jax.vmap(grads, in_axes))(*args)

    got, expected = run(fused), run(plain)
    for g, e in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        assert g.shape == e.shape
        _assert_close_to_scale(g, e)


@pytest.mark.parametrize(
    "in_axes",
    [(0, None, None), (None, None, 0), (None, 0, None), (0, 0, 0)],
    ids=["x", "weights", "index", "all"],
)
def test_vmap_matches_unfused(in_axes):
    """Under `jax.vmap`, any batched subset of `(x, index, weights)`."""
    case = ("0e+1o+2e", 2, "0e+1o", 4, 128, 37)
    fused, plain, feat, index, weights = _case(*case)
    lanes = _stack_lanes(3, feat, index, weights, case[3])
    args = [
        b if axis == 0 else a
        for a, b, axis in zip((feat, index, weights), lanes, in_axes)
    ]
    _assert_vmapped_force_grads_match(fused, plain, args, in_axes)


def test_vmap_folded_tables_over_shared_memory_match_unfused():
    """Per-lane weights whose folded tables do not all fit in shared memory
    run in chunks of lanes."""
    case = ("0e+1o+2e+3o", 3, "0e+1o+2e", 5, 256, 40)
    v = 8
    fused, plain, feat, index, weights = _case(*case)
    params = fused.mosaic_gpu_params
    folded = dataclasses.replace(params, num_species=v * params.num_species)
    assert not chain_supported(folded, case[-1], case[-2], "fwd")
    args = _stack_lanes(v, feat, index, weights, case[3])
    _assert_vmapped_force_grads_match(fused, plain, args, (0, 0, 0))


def test_vmap_weights_chunks_backward_with_addend():
    """The backward adding `dx1`'s second-order cotangent stages it in every
    row, so it fits fewer lanes per launch than the plain backward."""
    case = ("0e+1o+2e", 3, "0e+1o", 5, 128, 100)
    fused, plain, feat, index, weights = _case(*case)
    _, _, ws = _stack_lanes(5, feat, index, weights, case[3])
    # The unfused reference's batched matmuls run in TF32 by default.
    with jax.default_matmul_precision("highest"):
        _assert_vmapped_force_grads_match(
            fused, plain, (feat, index, ws), (None, None, 0)
        )


@pytest.mark.multi_devices
def test_vmap_execute_multi_devices():
    """Under a mesh, the vmap axis is split across `jax.devices()`."""
    case = ("0e+1o+2e", 2, "0e+1o", 4, 128, 37)
    fused, plain, feat, index, weights = _case(*case)
    args = _stack_lanes(2 * jax.device_count(), feat, index, weights, case[3])
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("batch",))
    with jax.sharding.set_mesh(mesh):
        _assert_vmapped_force_grads_match(fused, plain, args, (0, 0, 0))
