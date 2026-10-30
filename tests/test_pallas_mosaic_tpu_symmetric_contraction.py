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

"""Numerical correctness tests for the fused Pallas Mosaic TPU symmetric contraction."""

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import pytest
from conftest import skip_unless_tpu
from jax.core import ShapedArray
from jax.experimental.topologies import get_topology_desc
from jax.sharding import Mesh

from e3j.core.symmetric_contraction import (
    SymmetricContraction as CoreSymmetricContraction,
)
from e3j.pallas_ops.symmetric_contraction import symmetric_contraction_pallas_mosaic_tpu
from e3j.pallas_ops.symmetric_contraction.common.plan import build_plan, fold_weights
from e3j.pallas_ops.symmetric_contraction.mosaic_tpu.fwd import (
    block_size,
    symmetric_contraction_reference,
)
from e3j.pallas_ops.symmetric_contraction.mosaic_tpu.params import (
    BATCH_BLOCK_SIZE_CANDIDATES,
)
from e3j.utils.config import Config
from e3j.utils.options import SymmetricContraction as SymmetricContractionOption

RTOL = 2e-3
ATOL = 2e-3

UNFUSED = Config(symmetric_contraction=SymmetricContractionOption.UNFUSED)

# (source, correlation, target)
CASES = [
    ("0e + 1o", 2, "0e + 1o + 2e"),
    ("0e + 1o + 2e", 2, "0e + 1o + 2e"),
    ("0e + 1o", 3, "0e + 1o"),
    ("0e + 1o + 2e + 3o", 3, "0e + 1o"),
]
CASE_IDS = [f"{src}__corr{corr}__{tgt}" for src, corr, tgt in CASES]

# (batch, mul, num_species): unaligned batches and sub-128 mul exercise padding
SHAPES = [
    (256, 128, 4),
    (16, 64, 2),
    (37, 256, 8),
]
SHAPE_IDS = [f"b{b}_m{m}_s{s}" for b, m, s in SHAPES]

mosaic_tpu = pytest.mark.mosaic_tpu


@pytest.fixture
def require_tpu():
    skip_unless_tpu()


def _core(source, correlation, target, mul, num_species):
    layout = "TRAILING_CHANNELS"
    return CoreSymmetricContraction(
        source, correlation, target, num_species, mul, layout, config=UNFUSED
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


def _kernel_inputs(source, correlation, target, batch, mul, num_species):
    """`(params, features, weights, indices)`: the non-degenerate blocks'
    weights, the kernel's operands."""
    core = _core(source, correlation, target, mul, num_species)
    features, indices, weight = _inputs(core, batch)
    return core.mosaic_tpu_params, features, core._kernel_weights(weight), indices


def _reference(x, species, weights, params):
    """The pure-JAX reference on the kernel's own operands."""
    w = fold_weights(weights, build_plan(params))
    return symmetric_contraction_reference(x, w, species, params)


def test_params_are_hashable():
    first = _core("0e + 1o", 3, "0e + 1o", 8, 2).mosaic_tpu_params
    second = _core("0e + 1o", 3, "0e + 1o", 8, 2).mosaic_tpu_params
    other = _core("0e + 1o", 2, "0e + 1o", 8, 2).mosaic_tpu_params
    assert first == second and hash(first) == hash(second)
    assert first != other


def test_entry_point_rejects_mismatched_operands():
    params, x, w, species = _kernel_inputs("0e + 1o", 2, "0e + 1o", 16, 8, 2)

    def call(x=x, species=species, w=w):
        symmetric_contraction_pallas_mosaic_tpu(x, species, w, params)

    # one more input multiplicity than the block has: was silently ignored
    wide = w[0].shape[:-1] + (w[0].shape[-1] + 1,)
    with pytest.raises(ValueError, match=r"weights\[0\] must be"):
        call(w=[jnp.zeros(wide), *w[1:]])
    with pytest.raises(ValueError, match="block weights"):
        call(w=w[:-1])
    with pytest.raises(ValueError, match="index must be"):
        call(species=species[:-1])
    with pytest.raises(ValueError, match="integer"):
        call(species=species.astype(jnp.float32))
    with pytest.raises(ValueError, match="coordinate width"):
        call(x=x[:, :-1])
    with pytest.raises(ValueError, match="x1 is"):
        call(w=[a.astype(jnp.bfloat16) for a in w])
    with pytest.raises(ValueError, match="float32"):
        call(x=x.astype(jnp.bfloat16), w=[a.astype(jnp.bfloat16) for a in w])


def test_block_size_leaves_room_for_reserved_buffers():
    # 368 rows of 128 f32 channels per batch row: 32 rows fit the 6 MiB budget
    assert block_size(368, BATCH_BLOCK_SIZE_CANDIDATES) == 32
    assert block_size(368, BATCH_BLOCK_SIZE_CANDIDATES, 6 << 20) == 32
    assert block_size(368, BATCH_BLOCK_SIZE_CANDIDATES, 10 << 20) == 16
    # the smallest block when the reserved buffers alone fill the VMEM
    assert block_size(368, BATCH_BLOCK_SIZE_CANDIDATES, 16 << 20) == 8


def test_params_reject_blocks_under_8_rows():
    params = _core("0e + 1o", 2, "0e + 1o", 8, 2).mosaic_tpu_params
    with pytest.raises(ValueError):
        dataclasses.replace(params, fwd_batch_block_size=4)
    with pytest.raises(ValueError):
        dataclasses.replace(params, bwd_batch_block_size=2)


# (case, shape, block): every case and shape at the default blocks, one
# unaligned batch at explicit forward and backward blocks, and many species
KERNEL_CASES = (
    [
        pytest.param(case, shape, None, id=f"{case_id}-{shape_id}")
        for case, case_id in zip(CASES, CASE_IDS)
        for shape, shape_id in zip(SHAPES, SHAPE_IDS)
    ]
    + [
        pytest.param(
            ("0e + 1o + 2e", 3, "0e + 1o"), (100, 128, 3), block, id=f"bbs{block}"
        )
        for block in (8, 16)
    ]
    + [
        # the weight cotangent sums of 89 species and 256 channels overflow v4's
        # VMEM: the backward runs once per 128 channels
        pytest.param(
            ("0e + 1o + 2e + 3o", 3, "0e + 1o + 2e + 3o"),
            (64, 256, 89),
            None,
            id="split-channels",
        )
    ]
)


@mosaic_tpu
@pytest.mark.usefixtures("require_tpu")
@pytest.mark.parametrize(("case", "shape", "block"), KERNEL_CASES)
def test_kernel_matches_reference(case, shape, block):
    """Output and both cotangents of the kernel match the pure-JAX reference."""
    batch, mul, num_species = shape
    params, x, w, species = _kernel_inputs(*case, batch, mul, num_species)
    if block is not None:
        params = dataclasses.replace(
            params, fwd_batch_block_size=block, bwd_batch_block_size=block
        )

    ct = jax.random.normal(jax.random.key(7), (batch, params.out_dim, mul))

    # jitted: XLA may then place the operands in CMEM, which eager runs never see
    @jax.jit
    def run(x, w, ct):
        out, vjp = jax.vjp(
            lambda x, w: symmetric_contraction_pallas_mosaic_tpu(x, species, w, params),
            x,
            w,
        )
        return out, *vjp(ct)

    out, dx, dw = run(x, w, ct)
    expect, vjp_ref = jax.vjp(lambda x, w: _reference(x, species, w, params), x, w)
    dx_ref, dw_ref = vjp_ref(ct)

    assert out.shape == (batch, params.out_dim, mul)
    assert dx.shape == x.shape
    assert [d.shape for d in dw] == [v.shape for v in w]
    testing.assert_allclose(out, expect, rtol=RTOL, atol=ATOL)
    testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)
    for d, d_ref in zip(dw, dw_ref):
        testing.assert_allclose(d, d_ref, rtol=RTOL, atol=ATOL)


@mosaic_tpu
@pytest.mark.usefixtures("require_tpu")
def test_kernel_dx_only_matches_reference():
    """At fixed weights the backward writes no `dw`, and `dx` (and its own
    derivative, as for forces) still matches the reference."""
    params, x, w, species = _kernel_inputs("0e + 1o + 2e", 3, "0e + 1o", 100, 128, 3)

    def energy(f, x):
        return jnp.sum(jnp.sin(f(x, species, w, params)))

    dx = jax.grad(energy, argnums=1)(symmetric_contraction_pallas_mosaic_tpu, x)
    dx_ref = jax.grad(energy, argnums=1)(_reference, x)
    testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)

    def force_loss(f, x):
        return jnp.sum(jax.grad(energy, argnums=1)(f, x) ** 2)

    ddx = jax.grad(force_loss, argnums=1)(symmetric_contraction_pallas_mosaic_tpu, x)
    ddx_ref = jax.grad(force_loss, argnums=1)(_reference, x)
    testing.assert_allclose(ddx, ddx_ref, rtol=RTOL, atol=ATOL)

    hlo = (
        jax.jit(jax.grad(energy, argnums=1), static_argnums=0)
        .lower(symmetric_contraction_pallas_mosaic_tpu, x)
        .as_text()
    )
    assert "symmetric_contraction_mtpu_bwd_dx" in hlo


@mosaic_tpu
@pytest.mark.usefixtures("require_tpu")
@pytest.mark.parametrize(
    "in_axes",
    [(0, None, None), (None, None, 0), (None, 0, None), (0, 0, 0)],
    ids=["x", "weights", "species", "all"],
)
def test_kernel_vmap_matches_reference(in_axes):
    """Under `jax.vmap` (jitted), the output and both cotangents match the
    reference for any batched subset of `(x, species, weights)`."""
    params, x, w, species = _kernel_inputs("0e + 1o", 2, "0e + 1o + 2e", 37, 256, 3)
    v = 3
    xs = jnp.stack([x * (i + 1) for i in range(v)])
    ss = jnp.stack([jnp.roll(species, i) for i in range(v)])
    ws = [jnp.stack([a * (1 - 0.3 * i) for i in range(v)]) for a in w]
    args = [
        b if axis == 0 else a
        for a, b, axis in zip((x, species, w), (xs, ss, ws), in_axes)
    ]

    def run(f):
        def energy(x, species, w):
            return jnp.sum(jnp.sin(f(x, species, w, params)))

        grad = jax.value_and_grad(energy, argnums=(0, 2))
        return jax.vmap(grad, in_axes=in_axes)(*args)

    got = jax.jit(lambda: run(symmetric_contraction_pallas_mosaic_tpu))()
    expect = run(_reference)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expect)):
        assert a.shape == b.shape
        testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


def _vmapped_grad(f, params, in_axes=(0, 0, 0)):
    """`(x, species, w) -> (energy, (dx, dw))` of `f`, vmapped over `in_axes`."""

    def energy(x, species, w):
        return jnp.sum(jnp.sin(f(x, species, w, params)))

    return jax.vmap(jax.value_and_grad(energy, argnums=(0, 2)), in_axes=in_axes)


def _virtual_tpu_mesh() -> Mesh:
    """A compile-only `v4-8` mesh, or skip if `libtpu` is absent."""
    try:
        topology = get_topology_desc("v4-8", "tpu")
    except Exception as exc:  # noqa: BLE001 - any failure means no usable topology
        pytest.skip(f"virtual TPU topology unavailable (needs libtpu): {exc!r}")
    return Mesh(np.asarray(topology.devices), ("batch",))


def test_vmap_precompiles_device_free(batch=64, mul=128, num_species=3):
    """Under a mesh, the vmap axis is sharded across the devices: the
    forward and backward compile against a virtual topology (no live TPU),
    and every device runs the kernels on its own `V / devices` lanes, the
    batched weights tiled per lane."""
    mesh = _virtual_tpu_mesh()
    lanes = 2  # per device
    v = lanes * mesh.size
    params, x, w, species = _kernel_inputs(
        "0e + 1o", 2, "0e + 1o + 2e", batch, mul, num_species
    )
    args = (
        ShapedArray((v, *x.shape), x.dtype),
        ShapedArray((v, *species.shape), species.dtype),
        [ShapedArray((v, *a.shape), a.dtype) for a in w],
    )
    run = _vmapped_grad(symmetric_contraction_pallas_mosaic_tpu, params)
    with jax.sharding.set_mesh(mesh):
        hlo = jax.jit(run).lower(*args).compile().as_text()

    assert re.search(r"num_partitions=%d\b" % mesh.size, hlo)
    rows = lanes * batch
    fwd = r"= f32\[%d,%d,128\]\S* custom-call\(" % (rows, params.out_dim)
    bwd = r"= \(f32\[%d,%d,128\]\S*, f32\[1,%d,\d+,128\]" % (
        rows,
        x.shape[1],
        lanes * num_species,
    )
    for pattern in (fwd, bwd):
        assert re.search(pattern + r".*tpu_custom_call", hlo), pattern


def test_nested_vmap_shards_the_divisible_axis_device_free(batch=64, mul=128):
    """With an inner vmap axis the devices do not divide, the outer one is
    sharded instead."""
    mesh = _virtual_tpu_mesh()
    params, x, w, species = _kernel_inputs("0e + 1o", 2, "0e + 1o", batch, mul, 2)
    v = (mesh.size, 3)
    args = (
        ShapedArray((*v, *x.shape), x.dtype),
        ShapedArray((*v, *species.shape), species.dtype),
        [ShapedArray((*v, *a.shape), a.dtype) for a in w],
    )
    run = _vmapped_grad(symmetric_contraction_pallas_mosaic_tpu, params)
    with jax.sharding.set_mesh(mesh):
        hlo = jax.jit(jax.vmap(run)).lower(*args).compile().as_text()
    assert re.search(r"num_partitions=%d\b" % mesh.size, hlo)
    rows = v[1] * batch
    pattern = r"= f32\[%d,%d,128\]\S* custom-call\(.*tpu_custom_call"
    assert re.search(pattern % (rows, params.out_dim), hlo)


def test_vmap_indivisible_by_devices_raises():
    """A vmap axis the devices do not divide is not sharded, and no outer
    vmap layer shards it either."""
    mesh = _virtual_tpu_mesh()
    params, x, w, species = _kernel_inputs("0e + 1o", 2, "0e + 1o", 16, 128, 2)
    args = (
        ShapedArray((1, *x.shape), x.dtype),
        ShapedArray((1, *species.shape), species.dtype),
        [ShapedArray((1, *a.shape), a.dtype) for a in w],
    )
    run = _vmapped_grad(symmetric_contraction_pallas_mosaic_tpu, params)
    with jax.sharding.set_mesh(mesh):
        with pytest.raises(NotImplementedError, match="shard_map"):
            jax.jit(run).lower(*args)


@mosaic_tpu
@pytest.mark.multi_devices
@pytest.mark.usefixtures("require_tpu")
@pytest.mark.parametrize("in_axes", [(0, None, None), (0, 0, 0)], ids=["x", "all"])
def test_vmap_execute_multi_devices(in_axes, batch=37, mul=256, num_species=3):
    """Real sharded execution across `jax.devices()`: the output and both
    cotangents match the reference."""
    v = 2 * jax.device_count()
    mesh = Mesh(np.asarray(jax.devices()), ("batch",))
    params, x, w, species = _kernel_inputs(
        "0e + 1o", 2, "0e + 1o + 2e", batch, mul, num_species
    )
    xs = jnp.stack([x * (1 + 0.1 * i) for i in range(v)])
    ss = jnp.stack([jnp.roll(species, i) for i in range(v)])
    ws = [jnp.stack([a * (1 - 0.05 * i) for i in range(v)]) for a in w]
    args = [
        b if axis == 0 else a
        for a, b, axis in zip((x, species, w), (xs, ss, ws), in_axes)
    ]

    run = _vmapped_grad(symmetric_contraction_pallas_mosaic_tpu, params, in_axes)
    with jax.sharding.set_mesh(mesh):
        got = jax.jit(run)(*args)
    expect = _vmapped_grad(_reference, params, in_axes)(*args)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expect)):
        assert a.shape == b.shape
        testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
