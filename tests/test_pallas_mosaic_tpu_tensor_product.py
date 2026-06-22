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
MULS = [32, 64, 128]

CASES = [
    ("0e + 1o", "0e + 1o", "0e + 1o + 2e"),
    ("2x0e + 1o + 2e", "0e + 2x1o + 2e", "0e + 1o + 2e"),
    ("0e + 1o + 2e + 3o", "0e + 1o + 2e", "0e + 1o + 2e + 3o"),
]
CASE_IDS = [f"{a}__{b}__{c}" for a, b, c in CASES]


def parametrize_non_shard(func):
    """Decorator bundling CASES, BATCH_SIZES, and MULS parametrization."""
    func = pytest.mark.parametrize("mul", MULS)(func)
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
    contrib = values[None, :, None] * x[:, xi, :] * y[:, yi, None]  # (batch, nnz, mul)
    out = jnp.zeros((x.shape[0], params.z_space.dim, x.shape[2]), x.dtype)
    return out.at[:, oi, :].add(contrib)


def _reference_trailing_map(x, y, params):
    """TRAILING_CHANNELS MAP: o[b, oi, k] = Σ coef · x[b, xi, k] · y[b, yi, k]."""
    indices = jnp.asarray(params.indices)
    values = jnp.asarray(params.values, dtype=x.dtype)
    oi, xi, yi = indices[:, 0], indices[:, 1], indices[:, 2]
    contrib = values[None, :, None] * x[:, xi, :] * y[:, yi, :]  # (batch, nnz, mul)
    out = jnp.zeros((x.shape[0], params.z_space.dim, x.shape[2]), x.dtype)
    return out.at[:, oi, :].add(contrib)


def _inputs_trailing_outer(params, batch_size: int, mul: int, seed: int = 0):
    """TRAILING_CHANNELS OUTER inputs: x (batch, x_dim, mul), y (batch, y_dim)."""
    kx, ky = jax.random.split(jax.random.key(seed))
    x = jax.random.normal(kx, (batch_size, params.x_space.dim, mul))
    y = jax.random.normal(ky, (batch_size, params.y_space.dim))
    return x, y


def _inputs_trailing_map(params, batch_size: int, mul: int, seed: int = 0):
    """TRAILING_CHANNELS MAP inputs: x (batch, x_dim, mul), y (batch, y_dim, mul)."""
    kx, ky = jax.random.split(jax.random.key(seed))
    x = jax.random.normal(kx, (batch_size, params.x_space.dim, mul))
    y = jax.random.normal(ky, (batch_size, params.y_space.dim, mul))
    return x, y


@parametrize_non_shard
def test_forward_trailing_channel_outer_matches_reference(
    in1, in2, out, batch_size, mul
):
    """TRAILING_CHANNELS OUTER: inputs are (batch, dim, mul) and (batch, dim)."""
    params = _build_params(in1, in2, out)
    x, y = _inputs_trailing_outer(params, batch_size, mul)

    result = tensor_product_pallas_mosaic_tpu(x, y, params)
    expect = _reference_trailing_outer(x, y, params)

    assert result.shape == (batch_size, params.z_space.dim, mul)
    testing.assert_allclose(result, expect, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_backward_trailing_channel_outer_matches_reference(
    in1, in2, out, batch_size, mul
):
    """TRAILING_CHANNELS OUTER backward: dx (batch, x_dim, mul), dy (batch, y_dim)."""
    params = _build_params(in1, in2, out)
    x, y = _inputs_trailing_outer(params, batch_size, mul)

    do = jax.random.normal(
        jax.random.key(1), (batch_size, params.z_space.dim, mul), dtype=x.dtype
    )

    _, vjp = jax.vjp(lambda x, y: tensor_product_pallas_mosaic_tpu(x, y, params), x, y)
    dx, dy = vjp(do)

    _, vjp_ref = jax.vjp(lambda x, y: _reference_trailing_outer(x, y, params), x, y)
    dx_ref, dy_ref = vjp_ref(do)

    assert dx.shape == (batch_size, params.x_space.dim, mul)
    assert dy.shape == (batch_size, params.y_space.dim)
    testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)
    testing.assert_allclose(dy, dy_ref, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_forward_trailing_channel_map_matches_reference(in1, in2, out, batch_size, mul):
    """TRAILING_CHANNELS MAP: inputs are (batch, dim, mul) and (batch, dim, mul)."""
    params = _build_params(in1, in2, out, options.TPMode.MAP)
    x, y = _inputs_trailing_map(params, batch_size, mul)

    result = tensor_product_pallas_mosaic_tpu(x, y, params)
    expect = _reference_trailing_map(x, y, params)

    assert result.shape == (batch_size, params.z_space.dim, mul)
    testing.assert_allclose(result, expect, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_backward_trailing_channel_map_matches_reference(
    in1, in2, out, batch_size, mul
):
    """TRAILING_CHANNELS MAP backward: dx (batch, x_dim, mul), dy (batch, y_dim, mul)."""
    params = _build_params(in1, in2, out, options.TPMode.MAP)
    x, y = _inputs_trailing_map(params, batch_size, mul)

    do = jax.random.normal(
        jax.random.key(1), (batch_size, params.z_space.dim, mul), dtype=x.dtype
    )

    _, vjp = jax.vjp(lambda x, y: tensor_product_pallas_mosaic_tpu(x, y, params), x, y)
    dx, dy = vjp(do)

    _, vjp_ref = jax.vjp(lambda x, y: _reference_trailing_map(x, y, params), x, y)
    dx_ref, dy_ref = vjp_ref(do)

    assert dx.shape == (batch_size, params.x_space.dim, mul)
    assert dy.shape == (batch_size, params.y_space.dim, mul)
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
def test_class_dispatch_trailing_outer_forward(in1, in2, out, batch_size, mul):
    """`TensorProduct(...)(x, y)` with MOSAIC_TPU matches the OUTER reference."""
    with config.use(tensor_product=options.TensorProduct.FUSED_MOSAIC_TPU):
        tp = _mtpu_tp(in1, in2, out, options.TPMode.OUTER)
        params = tp._mtpu_params()
        x, y = _inputs_trailing_outer(params, batch_size, mul)
        result = tp(x, y)

    expect = _reference_trailing_outer(x, y, params)
    assert result.shape == (batch_size, params.z_space.dim, mul)
    testing.assert_allclose(result, expect, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_class_dispatch_trailing_outer_backward(in1, in2, out, batch_size, mul):
    """Backward through `TensorProduct.__call__` matches the OUTER reference."""
    with config.use(tensor_product=options.TensorProduct.FUSED_MOSAIC_TPU):
        tp = _mtpu_tp(in1, in2, out, options.TPMode.OUTER)
        params = tp._mtpu_params()
        x, y = _inputs_trailing_outer(params, batch_size, mul)
        do = jax.random.normal(
            jax.random.key(1), (batch_size, params.z_space.dim, mul), dtype=x.dtype
        )
        _, vjp = jax.vjp(tp, x, y)
        dx, dy = vjp(do)

    _, vjp_ref = jax.vjp(lambda x, y: _reference_trailing_outer(x, y, params), x, y)
    dx_ref, dy_ref = vjp_ref(do)

    assert dx.shape == (batch_size, params.x_space.dim, mul)
    assert dy.shape == (batch_size, params.y_space.dim)
    testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)
    testing.assert_allclose(dy, dy_ref, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_class_dispatch_trailing_map_forward(in1, in2, out, batch_size, mul):
    """`TensorProduct(...)(x, y)` with MOSAIC_TPU matches the MAP reference."""
    with config.use(tensor_product=options.TensorProduct.FUSED_MOSAIC_TPU):
        tp = _mtpu_tp(in1, in2, out, options.TPMode.MAP)
        params = tp._mtpu_params()
        x, y = _inputs_trailing_map(params, batch_size, mul)
        result = tp(x, y)

    expect = _reference_trailing_map(x, y, params)
    assert result.shape == (batch_size, params.z_space.dim, mul)
    testing.assert_allclose(result, expect, rtol=RTOL, atol=ATOL)


@parametrize_non_shard
def test_class_dispatch_trailing_map_backward(in1, in2, out, batch_size, mul):
    """Backward through `TensorProduct.__call__` matches the MAP reference."""
    with config.use(tensor_product=options.TensorProduct.FUSED_MOSAIC_TPU):
        tp = _mtpu_tp(in1, in2, out, options.TPMode.MAP)
        params = tp._mtpu_params()
        x, y = _inputs_trailing_map(params, batch_size, mul)
        do = jax.random.normal(
            jax.random.key(1), (batch_size, params.z_space.dim, mul), dtype=x.dtype
        )
        _, vjp = jax.vjp(tp, x, y)
        dx, dy = vjp(do)

    _, vjp_ref = jax.vjp(lambda x, y: _reference_trailing_map(x, y, params), x, y)
    dx_ref, dy_ref = vjp_ref(do)

    assert dx.shape == (batch_size, params.x_space.dim, mul)
    assert dy.shape == (batch_size, params.y_space.dim, mul)
    testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)
    testing.assert_allclose(dy, dy_ref, rtol=RTOL, atol=ATOL)


SHARD_CASES = [
    ("0e + 1o", "0e + 1o", "0e + 1o + 2e"),
    ("2x0e + 1o + 2e", "0e + 2x1o + 2e", "0e + 1o + 2e"),
    ("0e + 1o + 2e + 3o", "0e + 1o + 2e", "0e + 1o + 2e + 3o"),
]
SHARD_CASES_IDS = [f"{a}__{b}__{c}" for a, b, c in SHARD_CASES]


@pytest.mark.parametrize(("in1", "in2", "out"), SHARD_CASES, ids=SHARD_CASES_IDS)
def test_vmap_matches_loop(in1, in2, out, batch: int = 128, mul: int = 128):
    """jax.vmap over the fused TP matches a per-item loop in both value and
    gradient Exercises the data-parallel custom_vmap
    rule (no mesh -> identity shard_map)."""
    v = 4
    kx, ky = jax.random.split(jax.random.key(0))

    params = _build_params(in1, in2, out)
    xs = jax.random.normal(kx, (v, batch, params.x_space.dim, mul))
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


def _num_partitions(compiled) -> int:
    match = re.search(r"num_partitions=(\d+)", compiled.as_text())
    assert match is not None, "num_partitions missing from compiled HLO"
    return int(match.group(1))


def _assert_input_sharded(compiled, batch, mul) -> None:
    x_operand = r"f32\[%d,\d+,%d\]" % (batch, mul)  # (batch, x_dim, mul)
    pattern = (
        r'custom_call_target="tpu_custom_call", operand_layout_constraints=\{'
        + x_operand
    )
    hlo = compiled.as_text()
    assert re.search(
        pattern, hlo
    ), f"expected a sharded tpu_custom_call matching {pattern!r}\n\n{hlo}"


def _assert_grad_sharded(compiled, batch, mul) -> None:
    dx = r"f32\[%d,\d+,%d\]" % (batch, mul)  # (batch, x_dim, mul)
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


@pytest.mark.parametrize(("in1", "in2", "out"), SHARD_CASES, ids=SHARD_CASES_IDS)
def test_vmap_precompiles_device_free(in1, in2, out, batch: int = 128, mul: int = 128):
    """The dp-vmap forward and backward lower and compile against a virtual
    multi-device topology (no live TPU), sharding the merged batch axis across
    the whole mesh. The compiled HLO is checked to run the Mosaic forward and
    backward kernels on the per-shard batch size."""
    mesh = _virtual_tpu_mesh()
    v = mesh.size
    params = _build_params(in1, in2, out)
    x_dim, y_dim = params.x_space.dim, params.y_space.dim
    xa = ShapedArray((v, batch, x_dim, mul), jnp.float32)
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
    _assert_input_sharded(fwd, batch, mul)
    _assert_grad_sharded(grad, batch, mul)
