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

"""Numerical correctness tests for the Pallas Mosaic TPU tensor product."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import pytest
from jax.core import ShapedArray
from jax.experimental.topologies import get_topology_desc
from jax.sharding import Mesh

from e3j.core.tensor_product import TensorProduct
from e3j.pallas_ops.tensor_product.mosaic_tpu import tensor_product_pallas_mosaic_tpu
from e3j.pallas_ops.tensor_product.mosaic_tpu.params import (
    PallasMosaicTPUTensorProductParams,
)
from e3j.utils import config, options

pytestmark = pytest.mark.mosaic_tpu

RTOL = 1e-4
ATOL = 1e-4

BATCH_SIZES = [256, 1024, 4096]
CHANNELS = [32, 64, 128]

CASES = [
    ("0e + 1o", "0e + 1o", "0e + 1o + 2e"),
    ("2x0e + 1o + 2e", "0e + 2x1o + 2e", "0e + 1o + 2e"),
    ("0e + 1o + 2e + 3o", "0e + 1o + 2e", "0e + 1o + 2e + 3o"),
]
CASE_IDS = [f"{a}__{b}__{c}" for a, b, c in CASES]


def parametrize_non_shard(func):
    """Decorator bundling CASES, BATCH_SIZES, and CHANNELS parametrization."""
    func = pytest.mark.parametrize("channels", CHANNELS)(func)
    func = pytest.mark.parametrize("batch_size", BATCH_SIZES)(func)
    func = pytest.mark.parametrize(("in1", "in2", "out"), CASES, ids=CASE_IDS)(func)
    return func


@pytest.fixture(autouse=True)
def _require_tpu():
    """Skip (rather than error) when not running on a TPU backend."""
    if jax.default_backend() != "tpu":
        pytest.skip("pallas_mosaic_tpu tests require a TPU backend")


def _build_params(
    in1: str, in2: str, out: str | None, mode: options.TPMode = options.TPMode.OUTER
):
    """Build TRAILING_CHANNELS kernel params from a validated `TensorProduct`."""
    tp = TensorProduct((in1, in2), out, sort=True)
    params = PallasMosaicTPUTensorProductParams(
        indices=np.asarray(tp.coef.indices),
        values=np.asarray(tp.coef.data),
        layout=options.Layout.TRAILING_CHANNELS,
        mode=mode,
        x_space=tp.source[0],
        y_space=tp.source[1],
        z_space=tp.target,
    )
    return params


def _reference_trailing_outer(x, y, params):
    """TRAILING_CHANNELS OUTER: o[b, oi, k] = Σ coef · x[b, xi, k] · y[b, yi]."""
    indices = jnp.asarray(params.indices)
    values = jnp.asarray(params.values, dtype=x.dtype)
    oi, xi, yi = indices[:, 0], indices[:, 1], indices[:, 2]
    contrib = (
        values[None, :, None] * x[:, xi, :] * y[:, yi, None]
    )  # (batch, nnz, channels)
    out = jnp.zeros((x.shape[0], params.z_space.dim, x.shape[2]), x.dtype)
    return out.at[:, oi, :].add(contrib)


def _reference_trailing_map(x, y, params):
    """TRAILING_CHANNELS MAP: o[b, oi, k] = Σ coef · x[b, xi, k] · y[b, yi, k]."""
    indices = jnp.asarray(params.indices)
    values = jnp.asarray(params.values, dtype=x.dtype)
    oi, xi, yi = indices[:, 0], indices[:, 1], indices[:, 2]
    contrib = (
        values[None, :, None] * x[:, xi, :] * y[:, yi, :]
    )  # (batch, nnz, channels)
    out = jnp.zeros((x.shape[0], params.z_space.dim, x.shape[2]), x.dtype)
    return out.at[:, oi, :].add(contrib)


def _inputs_trailing_outer(params, batch_size: int, channels: int, seed: int = 0):
    """TRAILING_CHANNELS OUTER inputs: x (batch, x_dim, channels), y (batch, y_dim)."""
    kx, ky = jax.random.split(jax.random.key(seed))
    x = jax.random.normal(kx, (batch_size, params.x_space.dim, channels))
    y = jax.random.normal(ky, (batch_size, params.y_space.dim))
    return x, y


def _inputs_trailing_map(params, batch_size: int, channels: int, seed: int = 0):
    """TRAILING_CHANNELS MAP inputs: x (batch, x_dim, channels), y (batch, y_dim, channels)."""
    kx, ky = jax.random.split(jax.random.key(seed))
    x = jax.random.normal(kx, (batch_size, params.x_space.dim, channels))
    y = jax.random.normal(ky, (batch_size, params.y_space.dim, channels))
    return x, y


@parametrize_non_shard
def test_forward_trailing_channel_outer_matches_reference(
    in1, in2, out, batch_size, channels
):
    """TRAILING_CHANNELS OUTER: inputs are (batch, dim, channels) and (batch, dim)."""
    params = _build_params(in1, in2, out)
    x, y = _inputs_trailing_outer(params, batch_size, channels)

    result = tensor_product_pallas_mosaic_tpu(x, y, params)
    expect = _reference_trailing_outer(x, y, params)

    assert result.shape == (batch_size, params.z_space.dim, channels)
    testing.assert_allclose(result, expect, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_backward_trailing_channel_outer_matches_reference(
    in1, in2, out, batch_size, channels
):
    """TRAILING_CHANNELS OUTER backward: dx (batch, x_dim, channels), dy (batch, y_dim)."""
    params = _build_params(in1, in2, out)
    x, y = _inputs_trailing_outer(params, batch_size, channels)

    do = jax.random.normal(
        jax.random.key(1), (batch_size, params.z_space.dim, channels), dtype=x.dtype
    )

    _, vjp = jax.vjp(lambda x, y: tensor_product_pallas_mosaic_tpu(x, y, params), x, y)
    dx, dy = vjp(do)

    _, vjp_ref = jax.vjp(lambda x, y: _reference_trailing_outer(x, y, params), x, y)
    dx_ref, dy_ref = vjp_ref(do)

    assert dx.shape == (batch_size, params.x_space.dim, channels)
    assert dy.shape == (batch_size, params.y_space.dim)
    testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)
    testing.assert_allclose(dy, dy_ref, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_forward_trailing_channel_map_matches_reference(
    in1, in2, out, batch_size, channels
):
    """TRAILING_CHANNELS MAP: inputs are (batch, dim, channels) and (batch, dim, channels)."""
    params = _build_params(in1, in2, out, options.TPMode.MAP)
    x, y = _inputs_trailing_map(params, batch_size, channels)

    result = tensor_product_pallas_mosaic_tpu(x, y, params)
    expect = _reference_trailing_map(x, y, params)

    assert result.shape == (batch_size, params.z_space.dim, channels)
    testing.assert_allclose(result, expect, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_backward_trailing_channel_map_matches_reference(
    in1, in2, out, batch_size, channels
):
    """TRAILING_CHANNELS MAP backward: dx (batch, x_dim, channels), dy (batch, y_dim, channels)."""
    params = _build_params(in1, in2, out, options.TPMode.MAP)
    x, y = _inputs_trailing_map(params, batch_size, channels)

    do = jax.random.normal(
        jax.random.key(1), (batch_size, params.z_space.dim, channels), dtype=x.dtype
    )

    _, vjp = jax.vjp(lambda x, y: tensor_product_pallas_mosaic_tpu(x, y, params), x, y)
    dx, dy = vjp(do)

    _, vjp_ref = jax.vjp(lambda x, y: _reference_trailing_map(x, y, params), x, y)
    dx_ref, dy_ref = vjp_ref(do)

    assert dx.shape == (batch_size, params.x_space.dim, channels)
    assert dy.shape == (batch_size, params.y_space.dim, channels)
    testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)
    testing.assert_allclose(dy, dy_ref, rtol=RTOL, atol=ATOL)


# --- TensorProduct class dispatch (MOSAIC_TPU config) ---
#
# These exercise the public `TensorProduct.__call__` path rather than calling
# `tensor_product_pallas_mosaic_tpu` directly: with `tensor_product` set to
# `MOSAIC_TPU` the class must route through `mtpu_eval`. References are built
# from the same `tp` (via `tp.mtpu_params()`) so coefficients stay consistent.


def _mtpu_tp(in1: str, in2: str, out: str | None, mode: options.TPMode):
    """Build a TRAILING_CHANNELS `TensorProduct` wired to the Mosaic TPU path."""
    tp = TensorProduct(
        (in1, in2),
        out,
        layout=options.Layout.TRAILING_CHANNELS,
        mode=mode,
        sort=True,
    )
    assert tp.is_mtpu, "TensorProduct should dispatch to the Mosaic TPU kernel"
    return tp


@parametrize_non_shard
def test_class_dispatch_trailing_outer_forward(in1, in2, out, batch_size, channels):
    """`TensorProduct(...)(x, y)` with MOSAIC_TPU matches the OUTER reference."""
    with config.use(tensor_product=options.TensorProduct.FUSED_MOSAIC_TPU):
        tp = _mtpu_tp(in1, in2, out, options.TPMode.OUTER)
        params = tp._mtpu_params()
        x, y = _inputs_trailing_outer(params, batch_size, channels)
        result = tp(x, y)

    expect = _reference_trailing_outer(x, y, params)
    assert result.shape == (batch_size, params.z_space.dim, channels)
    testing.assert_allclose(result, expect, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_class_dispatch_trailing_outer_backward(in1, in2, out, batch_size, channels):
    """Backward through `TensorProduct.__call__` matches the OUTER reference."""
    with config.use(tensor_product=options.TensorProduct.FUSED_MOSAIC_TPU):
        tp = _mtpu_tp(in1, in2, out, options.TPMode.OUTER)
        params = tp._mtpu_params()
        x, y = _inputs_trailing_outer(params, batch_size, channels)
        do = jax.random.normal(
            jax.random.key(1), (batch_size, params.z_space.dim, channels), dtype=x.dtype
        )
        _, vjp = jax.vjp(tp, x, y)
        dx, dy = vjp(do)

    _, vjp_ref = jax.vjp(lambda x, y: _reference_trailing_outer(x, y, params), x, y)
    dx_ref, dy_ref = vjp_ref(do)

    assert dx.shape == (batch_size, params.x_space.dim, channels)
    assert dy.shape == (batch_size, params.y_space.dim)
    testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)
    testing.assert_allclose(dy, dy_ref, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_class_dispatch_trailing_map_forward(in1, in2, out, batch_size, channels):
    """`TensorProduct(...)(x, y)` with MOSAIC_TPU matches the MAP reference."""
    with config.use(tensor_product=options.TensorProduct.FUSED_MOSAIC_TPU):
        tp = _mtpu_tp(in1, in2, out, options.TPMode.MAP)
        params = tp._mtpu_params()
        x, y = _inputs_trailing_map(params, batch_size, channels)
        result = tp(x, y)

    expect = _reference_trailing_map(x, y, params)
    assert result.shape == (batch_size, params.z_space.dim, channels)
    testing.assert_allclose(result, expect, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_class_dispatch_trailing_map_backward(in1, in2, out, batch_size, channels):
    """Backward through `TensorProduct.__call__` matches the MAP reference."""
    with config.use(tensor_product=options.TensorProduct.FUSED_MOSAIC_TPU):
        tp = _mtpu_tp(in1, in2, out, options.TPMode.MAP)
        params = tp._mtpu_params()
        x, y = _inputs_trailing_map(params, batch_size, channels)
        do = jax.random.normal(
            jax.random.key(1), (batch_size, params.z_space.dim, channels), dtype=x.dtype
        )
        _, vjp = jax.vjp(tp, x, y)
        dx, dy = vjp(do)

    _, vjp_ref = jax.vjp(lambda x, y: _reference_trailing_map(x, y, params), x, y)
    dx_ref, dy_ref = vjp_ref(do)

    assert dx.shape == (batch_size, params.x_space.dim, channels)
    assert dy.shape == (batch_size, params.y_space.dim, channels)
    testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)
    testing.assert_allclose(dy, dy_ref, rtol=RTOL, atol=ATOL)


SHARD_CASES = [
    ("0e + 1o", "0e + 1o", "0e + 1o + 2e"),
    ("2x0e + 1o + 2e", "0e + 2x1o + 2e", "0e + 1o + 2e"),
    ("0e + 1o + 2e + 3o", "0e + 1o + 2e", "0e + 1o + 2e + 3o"),
]
SHARD_CASES_IDS = [f"{a}__{b}__{c}" for a, b, c in SHARD_CASES]


@pytest.mark.parametrize(("in1", "in2", "out"), SHARD_CASES, ids=SHARD_CASES_IDS)
def test_vmap_matches_loop(in1, in2, out, batch: int = 128, channels: int = 128):
    """jax.vmap over the fused TP matches a per-item loop in both value and
    gradient Exercises the data-parallel custom_vmap
    rule (no mesh -> identity shard_map)."""
    v = 4
    kx, ky = jax.random.split(jax.random.key(0))

    params = _build_params(in1, in2, out)
    xs = jax.random.normal(kx, (v, batch, params.x_space.dim, channels))
    ys = jax.random.normal(ky, (v, batch, params.y_space.dim))

    def tp(x, y):
        return tensor_product_pallas_mosaic_tpu(x, y, params)

    out_vmap = jax.jit(jax.vmap(tp))(xs, ys)
    expected = jnp.stack([tp(xs[i], ys[i]) for i in range(v)])
    testing.assert_allclose(out_vmap, expected, rtol=RTOL, atol=ATOL)

    loss = lambda x, y: jnp.sum(jax.vmap(tp)(x, y))  # noqa: E731
    dx, dy = jax.jit(jax.grad(loss, argnums=(0, 1)))(xs, ys)
    grads = [
        jax.grad(lambda x, y: jnp.sum(tp(x, y)), argnums=(0, 1))(xs[i], ys[i])
        for i in range(v)
    ]
    exp_dx = jnp.stack([g[0] for g in grads])
    exp_dy = jnp.stack([g[1] for g in grads])
    testing.assert_allclose(dx, exp_dx, rtol=RTOL, atol=ATOL)
    testing.assert_allclose(dy, exp_dy, rtol=RTOL, atol=ATOL)


@pytest.mark.multi_devices
@pytest.mark.parametrize(("in1", "in2", "out"), SHARD_CASES, ids=SHARD_CASES_IDS)
def test_vmap_execute_multi_devices_channel_packing(
    in1, in2, out, batch: int = 128, channels: int = 32
):
    """Real sharded execution across `jax.devices()` matches a per-item loop.

    `channels=32` makes `k = 128 // channels == 4`, clearing both fwd's and bwd's
    `k >= 4` channel-packing gate (`channels=64` would give `k=2` and stay
    unpacked), so this exercises the packed lane layout through the
    data-parallel `custom_vmap` -> `shard_map` rule for real, rather than the
    device-free compile-only checks below.
    """
    v = jax.device_count()
    mesh = Mesh(np.asarray(jax.devices()), ("batch",))
    kx, ky = jax.random.split(jax.random.key(0))

    params = _build_params(in1, in2, out)
    xs = jax.random.normal(kx, (v, batch, params.x_space.dim, channels))
    ys = jax.random.normal(ky, (v, batch, params.y_space.dim))

    def tp(x, y):
        return tensor_product_pallas_mosaic_tpu(x, y, params)

    with jax.sharding.set_mesh(mesh):
        out_vmap = jax.jit(jax.vmap(tp))(xs, ys)
    expected = jnp.stack([tp(xs[i], ys[i]) for i in range(v)])
    testing.assert_allclose(out_vmap, expected, rtol=RTOL, atol=ATOL)

    loss = lambda x, y: jnp.sum(jax.vmap(tp)(x, y))  # noqa: E731
    with jax.sharding.set_mesh(mesh):
        dx, dy = jax.jit(jax.grad(loss, argnums=(0, 1)))(xs, ys)
    grads = [
        jax.grad(lambda x, y: jnp.sum(tp(x, y)), argnums=(0, 1))(xs[i], ys[i])
        for i in range(v)
    ]
    exp_dx = jnp.stack([g[0] for g in grads])
    exp_dy = jnp.stack([g[1] for g in grads])
    testing.assert_allclose(dx, exp_dx, rtol=RTOL, atol=ATOL)
    testing.assert_allclose(dy, exp_dy, rtol=RTOL, atol=ATOL)


def _num_partitions(compiled) -> int:
    match = re.search(r"num_partitions=(\d+)", compiled.as_text())
    assert match is not None, "num_partitions missing from compiled HLO"
    return int(match.group(1))


def _assert_input_sharded(compiled, batch, channels) -> None:
    x_operand = r"f32\[%d,\d+,%d\]" % (batch, channels)  # (batch, x_dim, channels)
    pattern = (
        r'custom_call_target="tpu_custom_call", operand_layout_constraints=\{'
        + x_operand
    )
    hlo = compiled.as_text()
    assert re.search(
        pattern, hlo
    ), f"expected a sharded tpu_custom_call matching {pattern!r}\n\n{hlo}"


def _assert_grad_sharded(compiled, batch, channels) -> None:
    dx = r"f32\[%d,\d+,%d\]" % (batch, channels)  # (batch, x_dim, channels)
    pattern = r"= \(" + dx + r'.+?custom_call_target="tpu_custom_call"'
    hlo = compiled.as_text()
    assert re.search(
        pattern, hlo
    ), f"expected a sharded backward tpu_custom_call matching {pattern!r}\n\n{hlo}"


def _virtual_tpu_mesh(axis: str = "batch") -> Mesh:
    """Build a compile-only `v4-8` mesh, or skip if `libtpu` is absent."""
    try:
        topology = get_topology_desc("v4-8", "tpu")
    except Exception as exc:  # noqa: BLE001 - any failure means no usable topology
        pytest.skip(f"virtual TPU topology unavailable (needs libtpu): {exc!r}")
    return Mesh(np.asarray(topology.devices), (axis,))


def _packed_shard_dims(batch: int, channels: int) -> tuple[int, int]:
    """(batch, channels) as seen by the sharded `tpu_custom_call`, once the fwd/bwd
    `packing` gate folds `k = 128 // channels` batch rows into the lane axis
    (`_FwdTrailingChannelKernel._pack_x` / `_BwdTrailingChannelKernel._pack_rows`).
    Mirrors the gate so callers can assert on either the packed or unpacked shape."""
    k = 128 // channels if channels < 128 and 128 % channels == 0 else 1
    if k < 4:  # below the `k >= 4` gate: packing stays off
        return batch, channels
    return batch // k, channels * k


@pytest.mark.parametrize(
    "channels", [32, 128], ids=["channels32_packed", "channels128_unpacked"]
)
@pytest.mark.parametrize(("in1", "in2", "out"), SHARD_CASES, ids=SHARD_CASES_IDS)
def test_vmap_precompiles_device_free(in1, in2, out, channels, batch: int = 128):
    """The dp-vmap forward and backward lower and compile against a virtual
    multi-device topology (no live TPU), sharding the merged batch axis across
    the whole mesh. The compiled HLO is checked to run the Mosaic forward and
    backward kernels on the per-shard batch size, including the packed lane
    layout when `channels=32` clears the channel-packing gate."""
    mesh = _virtual_tpu_mesh()
    v = mesh.size
    params = _build_params(in1, in2, out)
    x_dim, y_dim = params.x_space.dim, params.y_space.dim
    xa = ShapedArray((v, batch, x_dim, channels), jnp.float32)
    ya = ShapedArray((v, batch, y_dim), jnp.float32)

    def tp(x, y):
        return tensor_product_pallas_mosaic_tpu(x, y, params)

    with jax.sharding.set_mesh(mesh):
        fwd = jax.jit(jax.vmap(tp)).lower(xa, ya).compile()
        grad = (
            jax.jit(jax.grad(lambda x, y: jnp.sum(jax.vmap(tp)(x, y)), argnums=(0, 1)))
            .lower(xa, ya)
            .compile()
        )

    assert _num_partitions(fwd) == v
    assert _num_partitions(grad) == v
    shard_batch, shard_channels = _packed_shard_dims(batch, channels)
    _assert_input_sharded(fwd, shard_batch, shard_channels)
    _assert_grad_sharded(grad, shard_batch, shard_channels)
