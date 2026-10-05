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

"""The fused Mosaic GPU tensor product must match a plain-JAX contraction.

Both modes are covered. `MAP` multiplies the two operands channel by channel;
`OUTER`'s `y` carries no channel axis and enters as a per-row scalar, which the
kernel reads straight out of SMEM -- so the two modes exercise different reads,
a different `dy` (a channel-summed one in `OUTER`), and a different `y` block.

The references here are written out in plain `jnp` from the same coefficients
rather than taken from `TensorProduct`'s own sparse evaluation, so a test
failure points at the kernel and not at a shared helper; the class-dispatch
tests at the end close that loop by comparing against the `UNFUSED` path.

Beyond values, the kernel's decomposition is what these check: the row blocking
and pipeline depth a CTA walks its rows with, the channel blocking that splits a
wide multiplicity over the grid, and the coordinates no Clebsch-Gordan path
reaches -- whose output and cotangent rows the kernels must still write, every
buffer they allocate being uninitialized.

The kernels need a Hopper (sm_90) or newer CUDA device to run, so the whole
module skips elsewhere: Pallas interpret mode has no discharge rule for Mosaic
GPU's async copies, so there is no interpret-mode tier to fall back on.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from e3j.core.tensor_product import TensorProduct

# Through the package rather than `...tensor_product.mosaic_gpu` directly: the
# package's import is guarded, so a jaxlib without Mosaic GPU support gives
# stubs and lets this module be collected -- and skipped by `_require_hopper`
# -- instead of failing collection and taking the rest of the run with it.
from e3j.pallas_ops.tensor_product import (
    PallasMosaicGPUTensorProductParams,
    PallasMosaicGPUTensorProductParamsBwdConfig,
    PallasMosaicGPUTensorProductParamsFwdConfig,
    tensor_product_pallas_mosaic_gpu,
)
from e3j.utils import config, options
from e3j.utils.options import Layout, MixingMode

pytestmark = pytest.mark.mosaic_gpu

RTOL = 1e-4
ATOL = 1e-4

#: `(x, y, target)` irreps. The last is the `l_max=3` full product the kernels
#: are tuned for (99 output coordinates, 353 coefficients); the second has
#: multiplicities above one on both inputs, and the third filters the target so
#: that some input coordinates feed no output at all.
CASES = [
    ("0e + 1o", "0e + 1o", "0e + 1o + 2e"),
    ("2x0e + 1o + 2e", "0e + 2x1o + 2e", "0e + 1o + 2e"),
    ("0e + 1o + 2e", "0e + 1o + 2e", "3o"),
    ("0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o"),
]
CASE_IDS = [f"{x}__{y}__{o}" for x, y, o in CASES]

MODES = [MixingMode.OUTER, MixingMode.MAP]
MODE_IDS = [mode.name for mode in MODES]


@pytest.fixture(autouse=True)
def _require_hopper():
    """Skip (rather than fail) when there is no Hopper-or-newer CUDA device."""
    if jax.default_backend() != "gpu":
        pytest.skip("mosaic_gpu tensor product tests require a CUDA backend")
    capability = getattr(jax.devices()[0], "compute_capability", None)
    if capability is None or tuple(int(p) for p in capability.split(".")) < (9, 0):
        pytest.skip(f"kernels target sm_90+, device is sm_{capability}")


def build_params(
    in1: str,
    in2: str,
    out: str | None,
    mode: MixingMode,
    **tuning,
) -> PallasMosaicGPUTensorProductParams:
    """TRAILING_CHANNELS kernel params from a `TensorProduct`'s coefficients."""
    tp = TensorProduct((in1, in2), out, sort=True)
    return PallasMosaicGPUTensorProductParams(
        indices=np.asarray(tp.coef.indices),
        values=np.asarray(tp.coef.data),
        layout=Layout.TRAILING_CHANNELS,
        mode=mode,
        x_space=tp.source[0],
        y_space=tp.source[1],
        z_space=tp.target,
        fwd_config=PallasMosaicGPUTensorProductParamsFwdConfig(**tuning),
        bwd_config=PallasMosaicGPUTensorProductParamsBwdConfig(**tuning),
    )


def reference(x: jax.Array, y: jax.Array, params) -> jax.Array:
    """`z[b, zi, c] = sum coef * x[b, xi, c] * y[b, yi(, c)]`, in plain JAX.

    The one expression covers both modes: `OUTER`'s `y` has no channel axis, so
    it is broadcast over one here -- which is exactly what the kernel refuses to
    do in memory.
    """
    indices = jnp.asarray(params.indices)
    values = jnp.asarray(params.values, dtype=x.dtype)
    zi, xi, yi = indices[:, 0], indices[:, 1], indices[:, 2]
    y_gathered = y[:, yi, None] if y.ndim == 2 else y[:, yi, :]
    contributions = values[None, :, None] * x[:, xi, :] * y_gathered
    zeros = jnp.zeros((x.shape[0], params.z_space.dim, x.shape[2]), x.dtype)
    return zeros.at[:, zi, :].add(contributions)


def operands(
    params, batch: int, channels: int, mode: MixingMode, seed: int = 0
) -> tuple[jax.Array, jax.Array]:
    """Random `(x, y)` of the shapes `mode` expects."""
    key_x, key_y = jax.random.split(jax.random.key(seed))
    x = jax.random.normal(key_x, (batch, params.x_space.dim, channels))
    y_shape = (
        (batch, params.y_space.dim)
        if mode is MixingMode.OUTER
        else (batch, params.y_space.dim, channels)
    )
    return x, jax.random.normal(key_y, y_shape)


def kernel_fn(params):
    """The op under test, jitted, with `params` closed over."""
    return jax.jit(lambda x, y: tensor_product_pallas_mosaic_gpu(x, y, params))


def channels_for(mode: MixingMode) -> list[int]:
    """Multiplicities the kernels accept in `mode`.

    Both modes take any multiple of 128 -- 256 and 384 need two and three
    channel blocks, which exercises the grid's channel axis and, in `OUTER`
    mode, the sum of `dy`'s per-block partials.
    """
    return [128, 256, 384]


# --------------------------------------------------------------- numerics ---


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
@pytest.mark.parametrize(("in1", "in2", "out"), CASES, ids=CASE_IDS)
def test_forward_matches_reference(in1, in2, out, mode):
    params = build_params(in1, in2, out, mode)
    for channels in channels_for(mode):
        x, y = operands(params, 256, channels, mode)

        got = kernel_fn(params)(x, y)
        expected = jax.jit(lambda x, y: reference(x, y, params))(x, y)

        assert got.shape == (256, params.z_space.dim, channels)
        np.testing.assert_allclose(
            got, expected, rtol=RTOL, atol=ATOL, err_msg=f"{channels=}"
        )


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
@pytest.mark.parametrize(("in1", "in2", "out"), CASES, ids=CASE_IDS)
def test_cotangents_match_reference(in1, in2, out, mode):
    """Both cotangents, against an output cotangent uncorrelated with them."""
    params = build_params(in1, in2, out, mode)
    for channels in channels_for(mode):
        x, y = operands(params, 256, channels, mode)
        cotangent = jax.random.normal(
            jax.random.key(7), (256, params.z_space.dim, channels)
        )

        _, pullback = jax.vjp(kernel_fn(params), x, y)
        _, pullback_ref = jax.vjp(jax.jit(lambda x, y: reference(x, y, params)), x, y)
        got = jax.jit(pullback)(cotangent)
        expected = jax.jit(pullback_ref)(cotangent)

        for name, a, b in zip(("dx", "dy"), got, expected):
            assert a.shape == b.shape, f"{name} shape, {channels=}"
            np.testing.assert_allclose(
                a, b, rtol=RTOL, atol=ATOL, err_msg=f"{name}, {channels=}"
            )


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_untouched_coordinates_are_zeroed(mode):
    """A coordinate no coefficient reaches still gets a written, zero row.

    Filtering the target down to `3o` leaves `x` and `y` coordinates that feed
    no kept output, so nothing accumulates into their cotangent rows. The
    kernel's buffers are uninitialized, so those rows are only zero because it
    writes them explicitly.
    """
    params = build_params("0e + 1o + 2e", "0e + 1o + 2e", "3o", mode)
    used_xis = {int(i) for i in params.indices[:, 1]}
    untouched = sorted(set(range(params.x_space.dim)) - used_xis)
    assert untouched, "this case is meant to leave x coordinates untouched"

    x, y = operands(params, 256, 128, mode)
    cotangent = jax.random.normal(jax.random.key(1), (256, params.z_space.dim, 128))
    _, pullback = jax.vjp(kernel_fn(params), x, y)
    dx, _dy = jax.jit(pullback)(cotangent)

    assert not np.any(np.asarray(dx)[:, untouched, :])


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_output_coordinates_with_no_path_are_written(mode):
    """An output coordinate no coefficient reaches must come back zero.

    The output buffer is not zero-initialized, so a coordinate the kernel skips
    keeps whatever the allocation held -- and on a fresh, zeroed allocation that
    reads as a pass. Running the kernel twice, the second time into memory the
    first call's (non-zero) output has been freed from, is what makes a skipped
    coordinate visible.

    A `TensorProduct`'s inferred target has a path for every coordinate, so the
    gap is built here by dropping one output index from the coefficients, as
    explicit coefficients may.
    """
    dense = build_params("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", mode)
    gap = 7
    keep = np.asarray(dense.indices)[:, 0] != gap
    params = PallasMosaicGPUTensorProductParams(
        indices=np.asarray(dense.indices)[keep],
        values=np.asarray(dense.values)[keep],
        layout=Layout.TRAILING_CHANNELS,
        mode=mode,
        x_space=dense.x_space,
        y_space=dense.y_space,
        z_space=dense.z_space,
    )
    assert gap not in {int(i) for i in params.indices[:, 0]}

    # Fill, then free, a buffer of the output's shape and value range, so the
    # allocator hands the kernel dirty memory rather than zeros.
    x, y = operands(params, 256, 128, mode)
    dirty = kernel_fn(dense)(x, y)
    assert np.any(np.asarray(dirty)[:, gap, :]), "the dense product writes there"
    del dirty

    got = kernel_fn(params)(x, y)
    expected = jax.jit(lambda x, y: reference(x, y, params))(x, y)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)
    assert not np.any(np.asarray(got)[:, gap, :])


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("batch", [1, 7, 129, 1001, 4096])
def test_batch_need_not_divide_the_row_block(batch, mode):
    """Any batch size works: the row block is narrowed to a divisor of it.

    A CTA walks a fixed number of rows, so a block that did not divide the
    batch would either read past the end or, clamped, have two CTAs write the
    same row. `resolve_rows_per_block` takes the default's greatest common
    divisor with the batch instead of padding the operands, which would cost a
    full copy of them.
    """
    params = build_params("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", mode)
    x, y = operands(params, batch, 128, mode)

    got = kernel_fn(params)(x, y)
    expected = jax.jit(lambda x, y: reference(x, y, params))(x, y)

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("channels", [128, 256])
@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_empty_batch_differentiates(mode, channels):
    """A product over zero rows -- an edgeless batch -- has zero cotangents.

    The backward has nothing to launch on then, and `OUTER`'s per-channel-block
    `dy` buffer has no row to slice.
    """
    params = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", mode)
    x, y = operands(params, 0, channels, mode)

    z, pullback = jax.vjp(kernel_fn(params), x, y)
    assert z.shape == (0, params.z_space.dim, channels)
    dx, dy = jax.jit(pullback)(jnp.zeros_like(z))
    assert dx.shape == x.shape and dy.shape == y.shape


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_more_row_blocks_than_a_cuda_grid_dimension_takes(mode):
    """A batch needing over 65535 CTAs must still launch.

    Only the first CUDA grid dimension takes more than 65535 blocks, so the
    launch is one flat axis that each CTA splits into its row block and channel
    block (`common.block_indices`). A two-dimensional grid put the row blocks
    on the second dimension and died at exactly 65536 with
    `cuLaunchKernelEx: invalid argument`. The spaces here are one coordinate
    wide so that 65537 rows stay cheap.
    """
    params = build_params("0e", "0e", "0e", mode)
    batch = 65_537
    x, y = operands(params, batch, 128, mode)

    got = kernel_fn(params)(x, y)
    expected = jax.jit(lambda x, y: reference(x, y, params))(x, y)

    assert got.shape == (batch, 1, 128)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_vmap_folds_the_mapped_axis_into_the_batch(mode):
    """A mapped axis of 65536 slices must launch, forward and backward.

    Left to `plgpu.kernel`'s batching rule, the mapped axis became a second grid
    axis, on the CUDA grid's second dimension, and 65536 slices failed with
    `CUDA_ERROR_INVALID_VALUE` -- the limit the flat grid exists to avoid.
    """
    params = build_params("0e", "0e", "0e", mode)
    n_slices = 65_536
    x, y = operands(params, n_slices, 128, mode)
    x, y = x[:, None], y[:, None]  # (n_slices, 1, ...): one row per slice
    cotangent = jax.random.normal(jax.random.key(5), (n_slices, 1, 1, 128))

    mapped = jax.vmap(lambda x, y: tensor_product_pallas_mosaic_gpu(x, y, params))
    reference_mapped = jax.vmap(lambda x, y: reference(x, y, params))
    got, pullback = jax.vjp(jax.jit(mapped), x, y)
    expected, pullback_ref = jax.vjp(jax.jit(reference_mapped), x, y)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)
    for a, b in zip(jax.jit(pullback)(cotangent), jax.jit(pullback_ref)(cotangent)):
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_vmap_over_one_operand_and_nested(mode):
    """An unmapped operand is shared by every slice, and vmaps nest."""
    params = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", mode)
    x, y = operands(params, 6 * 8, 128, mode)
    x = x.reshape((2, 3, 8) + x.shape[1:])
    y_shared = y[:8]
    op = lambda x, y: tensor_product_pallas_mosaic_gpu(x, y, params)  # noqa: E731
    ref = lambda x, y: reference(x, y, params)  # noqa: E731

    in_axes = (0, None)
    got = jax.jit(jax.vmap(jax.vmap(op, in_axes), in_axes))(x, y_shared)
    expected = jax.vmap(jax.vmap(ref, in_axes), in_axes)(x, y_shared)
    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)

    # `jacrev` is a vmap of the backward over the output's coordinates.
    x1, y1 = operands(params, 2, 128, mode)
    loss = lambda op: lambda y: op(x1, y).sum(axis=(0, 2))  # noqa: E731
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(loss(op)))(y1),
        jax.jacrev(loss(ref))(y1),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
@pytest.mark.parametrize(
    ("rows_per_block", "pipeline_depth"),
    [(1, 1), (1, 4), (2, 2), (8, 2), (8, 8), (32, 3)],
)
def test_row_blocking_agrees(rows_per_block, pipeline_depth, mode):
    """Every row block and pipeline depth must give the same answer.

    `pipeline_depth` sets the staging slots a CTA rotates through, so it is what
    the ordering between a body's reads and the pipeline's next fetch is keyed
    on: depth 1 leaves no overlap, a depth above the row count never wraps, and
    the ones in between wrap mid-walk. A slot refilled a step early would show
    up here as a wrong -- or hung -- run.
    """
    params = build_params(
        "0e + 1o + 2e",
        "0e + 1o + 2e",
        "0e + 1o + 2e",
        mode,
        rows_per_block=rows_per_block,
        pipeline_depth=pipeline_depth,
    )
    x, y = operands(params, 512, 128, mode)
    cotangent = jax.random.normal(jax.random.key(2), (512, params.z_space.dim, 128))

    default = build_params("0e + 1o + 2e", "0e + 1o + 2e", "0e + 1o + 2e", mode)
    for tuned, plain in (
        (kernel_fn(params)(x, y), kernel_fn(default)(x, y)),
        *zip(
            jax.jit(jax.vjp(kernel_fn(params), x, y)[1])(cotangent),
            jax.jit(jax.vjp(kernel_fn(default), x, y)[1])(cotangent),
        ),
    ):
        np.testing.assert_allclose(tuned, plain, rtol=RTOL, atol=ATOL)


def test_channel_block_override_agrees():
    """A 256-wide channel block must agree with the 128-wide default.

    The block width is how much of the channel axis one CTA covers; the grid's
    channel axis covers the rest. Splitting 512 channels two ways or four ways
    has to compute the same product.
    """
    kwargs = dict(in1="0e + 1o + 2e", in2="0e + 1o + 2e", out="0e + 1o + 2e")
    wide = build_params(**kwargs, mode=MixingMode.MAP, channel_block=256)
    default = build_params(**kwargs, mode=MixingMode.MAP)
    x, y = operands(wide, 256, 512, MixingMode.MAP)

    np.testing.assert_allclose(
        kernel_fn(wide)(x, y), kernel_fn(default)(x, y), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_float32_operands_under_x64(mode):
    """`jax_enable_x64` must not change what a float32 product traces to.

    It turns Python ints into int64 constants, and the grid index the kernels
    split into blocks is int32, so any arithmetic mixing the two without a cast
    fails the trace.
    """
    params = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", mode)
    with jax.enable_x64():
        x, y = (a.astype(jnp.float32) for a in operands(params, 64, 256, mode))
        cotangent = jax.random.normal(
            jax.random.key(3), (64, params.z_space.dim, 256), jnp.float32
        )
        got, pullback = jax.vjp(kernel_fn(params), x, y)
        _, pullback_ref = jax.vjp(lambda x, y: reference(x, y, params), x, y)
        expected = reference(x, y, params)
        np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)
        for a, b in zip(jax.jit(pullback)(cotangent), pullback_ref(cotangent)):
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("channels", [128, 256])
@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_second_order_matches_reference(mode, channels):
    """A gradient of a gradient must match the reference's.

    Reverse over reverse is what training on forces needs, and it only works
    because both of this op's rules recurse through the wrapped ops rather than
    the raw kernels (`primitives.py`). Contracted with random tangents rather
    than materialized: the full second derivative of this map is a large dense
    object, and a Hessian-vector product exercises the same rule.
    """
    params = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", mode)
    x, y = operands(params, 128, channels, mode)
    weight = jax.random.normal(jax.random.key(8), (128, params.z_space.dim, channels))
    tangent_x = jax.random.normal(jax.random.key(9), x.shape)
    tangent_y = jax.random.normal(jax.random.key(10), y.shape)

    def hessian_vector_product(product):
        def directional(x, y):
            gradients = jax.grad(lambda x, y: (product(x, y) * weight).sum(), (0, 1))
            return sum(
                (g * t).sum() for g, t in zip(gradients(x, y), (tangent_x, tangent_y))
            )

        return jax.jit(jax.grad(directional, (0, 1)))(x, y)

    got = hessian_vector_product(
        lambda x, y: tensor_product_pallas_mosaic_gpu(x, y, params)
    )
    expected = hessian_vector_product(lambda x, y: reference(x, y, params))

    for name, a, b in zip(("d2/dx", "d2/dy"), got, expected):
        assert a.shape == b.shape, f"{name} shape"
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=name)


def _assert_relatively_close(got, expected, names):
    """Each array is within `RTOL` of its reference, measured against its own
    largest entry: derivatives of these polynomial losses grow with the order,
    so an elementwise `atol` would be too tight on some and too loose on others."""
    for name, a, b in zip(names, got, expected):
        assert a.shape == b.shape, f"{name} shape"
        scale = max(float(jnp.abs(b).max()), 1.0)
        np.testing.assert_allclose(a, b, rtol=0, atol=RTOL * scale, err_msg=name)


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_higher_orders_match_reference(mode, order):
    """The rule on the backward is itself differentiable, to any order.

    The product is bilinear, so under a linear loss every derivative past the
    second vanishes and the check proves nothing; squaring the output keeps
    every order nonzero. Each level contracts the previous gradient with fresh
    random tangents and differentiates that scalar again, so every pass lands on
    `_ct_bwd` with nonzero cotangents on all of `(x, y, ct_z)`.
    """
    params = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", mode)
    x, y = operands(params, 128, 128, mode)
    weight = jax.random.normal(jax.random.key(11), (128, params.z_space.dim, 128))
    tangents = [
        (
            jax.random.normal(jax.random.key(12 + 2 * k), x.shape),
            jax.random.normal(jax.random.key(13 + 2 * k), y.shape),
        )
        for k in range(order - 1)
    ]

    def nth_derivative(product):
        def directional(f, tangent):
            gradients = jax.grad(f, (0, 1))
            return lambda x, y: sum(
                (g * t).sum() for g, t in zip(gradients(x, y), tangent)
            )

        f = lambda x, y: (weight * product(x, y) ** 2).sum()  # noqa: E731
        for tangent in tangents:
            f = directional(f, tangent)
        return jax.jit(jax.grad(f, (0, 1)))(x, y)

    got = nth_derivative(lambda x, y: tensor_product_pallas_mosaic_gpu(x, y, params))
    expected = nth_derivative(lambda x, y: reference(x, y, params))
    _assert_relatively_close(got, expected, (f"d{order}/dx", f"d{order}/dy"))


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_force_training_under_vmap(mode):
    """Second order over a vmapped batch, the shape of a force-training step.

    The forces are `x`'s gradient of an energy and the loss is their squared
    norm, so its gradient runs the backward's own rule under the mapped axis.
    """
    params = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", mode)
    x, y = operands(params, 3 * 32, 128, mode)
    x = x.reshape((3, 32) + x.shape[1:])
    y = y.reshape((3, 32) + y.shape[1:])

    def second_order(product):
        def loss(x, y):
            forces = jax.grad(lambda x: (product(x, y) ** 2).sum())(x)
            return (forces**2).sum()

        return jax.jit(jax.vmap(jax.grad(loss, (0, 1))))(x, y)

    got = second_order(lambda x, y: tensor_product_pallas_mosaic_gpu(x, y, params))
    expected = second_order(lambda x, y: reference(x, y, params))
    _assert_relatively_close(got, expected, ("d2/dx", "d2/dy"))


@pytest.mark.parametrize("y_irreps", ["32x0e", "9x1o + 0e", "4x2e"])
def test_outer_y_widths_around_the_padding_boundary(y_irreps):
    """`OUTER`'s `y` row is padded to 128 bytes, and its `dy` tile to 8 rows.

    Those two roundings differ, and `y_dim=32` needs no padding at all -- the
    branch `pad_y_rows` takes when the row is already wide enough. The `dy`
    buffer has to be wide enough for whichever rounding is larger.
    """
    params = build_params("0e + 1o", y_irreps, None, MixingMode.OUTER)
    x, y = operands(params, 256, 128, MixingMode.OUTER)
    cotangent = jax.random.normal(jax.random.key(11), (256, params.z_space.dim, 128))

    _, pullback = jax.vjp(kernel_fn(params), x, y)
    _, pullback_ref = jax.vjp(jax.jit(lambda x, y: reference(x, y, params)), x, y)

    np.testing.assert_allclose(
        kernel_fn(params)(x, y),
        jax.jit(lambda x, y: reference(x, y, params))(x, y),
        rtol=RTOL,
        atol=ATOL,
    )
    for name, a, b in zip(("dx", "dy"), pullback(cotangent), pullback_ref(cotangent)):
        assert a.shape == b.shape, f"{name} shape"
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=name)


# ---------------------------------------------------- unsupported shapes ---


@pytest.mark.parametrize(
    ("mode", "channels"),
    [
        (MixingMode.OUTER, 0),
        (MixingMode.OUTER, 1),
        (MixingMode.OUTER, 64),
        (MixingMode.OUTER, 192),
        (MixingMode.MAP, 0),
        (MixingMode.MAP, 1),
        (MixingMode.MAP, 32),
        (MixingMode.MAP, 192),
    ],
)
def test_unsupported_multiplicity_raises(mode, channels):
    """An unsupported multiplicity must be refused, not quietly miscomputed."""
    params = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", mode)
    x, y = operands(params, 256, channels, mode)

    with pytest.raises(NotImplementedError, match="channels"):
        kernel_fn(params)(x, y)


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_wide_input_coordinate_axis_raises(mode):
    """A space too wide to stage a row of must be refused by both passes.

    Both passes copy a whole `(1, dim, channel_block)` row of each operand into
    SMEM, and an async copy carries at most 256 elements along a dimension, so
    an input space above that needs coordinate blocking the kernels do not have.
    """
    for in1, in2, space in (("300x0e", "0e", "x_space"), ("0e", "300x0e", "y_space")):
        params = build_params(in1, in2, None, mode)
        x, y = operands(params, 128, 128, mode)
        with pytest.raises(NotImplementedError, match=space):
            kernel_fn(params)(x, y)


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_wide_output_coordinate_axis_raises(mode):
    """A wide output space is refused by both passes, though only one needs it.

    The forward never stages the output -- its rows go straight from registers
    to GMEM -- so the 256-coordinate bound is really the backward's, which
    stages a row of `ct_z`. Both passes take it anyway, so that a shape the op
    accepts is one it can also differentiate.
    """
    params = build_params("60x0e", "5x0e", None, mode)
    assert params.z_space.dim == 300
    x, y = operands(params, 128, 128, mode)

    with pytest.raises(NotImplementedError, match="z_space"):
        kernel_fn(params)(x, y)


def test_forward_mode_is_refused_clearly():
    """`jax.jvp` has no rule here, and must say so rather than crash inside."""
    params = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", MixingMode.OUTER)
    x, y = operands(params, 128, 128, MixingMode.OUTER)

    with pytest.raises(TypeError, match="forward-mode"):
        jax.jvp(
            lambda x, y: tensor_product_pallas_mosaic_gpu(x, y, params),
            (x, y),
            (jnp.ones_like(x), jnp.ones_like(y)),
        )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_non_float32_operands_are_refused(dtype):
    """The kernels are float32-only, and say which operand is not.

    `float64` is not in the list because JAX truncates it to `float32` unless
    `jax_enable_x64` is set, so the kernel never sees it here.
    """
    params = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", MixingMode.OUTER)
    x, y = operands(params, 256, 128, MixingMode.OUTER)

    with pytest.raises(NotImplementedError, match="float32-only"):
        kernel_fn(params)(x.astype(dtype), y.astype(dtype))


def test_mismatched_operands_are_refused():
    """A batch or channel count the two operands disagree on is a `ValueError`."""
    params = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", MixingMode.MAP)
    x, y = operands(params, 256, 128, MixingMode.MAP)

    with pytest.raises(ValueError, match="batch size"):
        kernel_fn(params)(x, y[:128])
    with pytest.raises(ValueError, match="channel count"):
        kernel_fn(params)(x, y[:, :, :64])
    with pytest.raises(ValueError, match="x_dim"):
        kernel_fn(params)(x[:, :2], y)


@pytest.mark.parametrize(
    ("knob", "value", "message"),
    [
        ("rows_per_block", 3, "must divide"),
        ("channel_block", 192, "multiple of"),
        ("channel_block", 384, "async copy"),
        ("channel_block", 64, "multiple of"),
    ],
)
def test_illegal_tuning_is_refused(knob, value, message):
    """A tuning value the decomposition cannot honour is a `ValueError`."""
    params = build_params(
        "0e + 1o", "0e + 1o", "0e + 1o + 2e", MixingMode.MAP, **{knob: value}
    )
    x, y = operands(params, 256, 384, MixingMode.MAP)

    with pytest.raises(ValueError, match=message):
        kernel_fn(params)(x, y)


def test_out_of_range_coefficients_are_refused():
    """An index outside its space must be refused when the params are built.

    The coefficients become static offsets into a staged SMEM row, with nothing
    to clamp them, so an out-of-range index is an out-of-bounds shared-memory
    read -- which takes the CUDA context down rather than raising. The check
    belongs where the spaces and the coefficients first meet.
    """
    tp = TensorProduct(("0e + 1o", "0e + 1o"), "0e + 1o + 2e", sort=True)
    spaces = dict(x_space=tp.source[0], y_space=tp.source[1], z_space=tp.target)
    for _column, axis in (("z", 0), ("x", 1), ("y", 2)):
        # Past every space here (the largest is 13 wide) but still inside the
        # coefficients' own index dtype, which is as narrow as `uint8`.
        indices = np.asarray(tp.coef.indices).copy()
        indices[0, axis] = 200
        with pytest.raises(ValueError, match=f"column {axis}"):
            PallasMosaicGPUTensorProductParams(
                indices=indices,
                values=np.asarray(tp.coef.data),
                layout=Layout.TRAILING_CHANNELS,
                mode=MixingMode.OUTER,
                **spaces,
            )


def test_pipeline_depth_over_the_smem_budget_is_refused():
    """A depth whose staging cannot fit must name the knob to turn.

    The backward stages a `ct_z` row per slot -- `z_dim * channel_block * 4`
    bytes, 50 KiB at this shape -- so a deep pipeline runs out of a CTA's
    shared memory. The platform refuses it too, but says nothing about which
    config field caused it or which pass failed.
    """
    params = build_params(
        "0e + 1o + 2e + 3o",
        "0e + 1o + 2e + 3o",
        "0e + 1o + 2e + 3o",
        MixingMode.OUTER,
        pipeline_depth=8,
        rows_per_block=8,
    )
    x, y = operands(params, 128, 128, MixingMode.OUTER)

    # The forward stages only `x`, so it still fits and still runs.
    z = kernel_fn(params)(x, y)
    cotangent = jax.random.normal(jax.random.key(12), z.shape)
    _, pullback = jax.vjp(kernel_fn(params), x, y)
    with pytest.raises(ValueError, match="pipeline_depth"):
        pullback(cotangent)


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_pipeline_depth_beyond_the_row_block_is_not_budgeted(mode):
    """The SMEM check must budget the slots the pipeline really allocates.

    `emit_pipeline` clamps its depth to the step count, and at the default one
    row per block that is one slot whatever `pipeline_depth` says. The full
    `l_max=3` product has 256 output coordinates, so two backward slots of its
    `ct_z` tile overflow the budget while the one actually allocated fits: the
    default config has to differentiate it.
    """
    params = build_params("0e + 1o + 2e + 3o", "0e + 1o + 2e + 3o", None, mode)
    assert params.z_space.dim == 256
    x, y = operands(params, 64, 128, mode)
    cotangent = jax.random.normal(jax.random.key(4), (64, 256, 128))

    _, pullback = jax.vjp(kernel_fn(params), x, y)
    _, pullback_ref = jax.vjp(jax.jit(lambda x, y: reference(x, y, params)), x, y)
    for a, b in zip(jax.jit(pullback)(cotangent), jax.jit(pullback_ref)(cotangent)):
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


def test_unsupported_mode_and_layout_raise():
    """`INNER` mode and the two channel-folded layouts are refused up front."""
    tp = TensorProduct(("0e + 1o", "0e + 1o"), "0e + 1o + 2e", sort=True)
    common = dict(
        indices=np.asarray(tp.coef.indices),
        values=np.asarray(tp.coef.data),
        x_space=tp.source[0],
        y_space=tp.source[1],
        z_space=tp.target,
    )
    with pytest.raises(NotImplementedError, match="OUTER"):
        PallasMosaicGPUTensorProductParams(
            layout=Layout.TRAILING_CHANNELS, mode=MixingMode.INNER, **common
        )
    for layout in (Layout.LEADING_CHANNELS, Layout.E3NN):
        with pytest.raises(NotImplementedError, match="TRAILING_CHANNELS"):
            PallasMosaicGPUTensorProductParams(
                layout=layout, mode=MixingMode.OUTER, **common
            )


def test_wrong_y_rank_raises():
    """Each mode wants its own `y` rank: 2-D in `OUTER`, 3-D in `MAP`."""
    outer = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", MixingMode.OUTER)
    mapped = build_params("0e + 1o", "0e + 1o", "0e + 1o + 2e", MixingMode.MAP)
    x, y_2d = operands(outer, 256, 128, MixingMode.OUTER)
    _, y_3d = operands(mapped, 256, 128, MixingMode.MAP)

    with pytest.raises(ValueError, match="batch, y_dim"):
        kernel_fn(outer)(x, y_3d)
    with pytest.raises(ValueError, match="batch, y_dim, channels"):
        kernel_fn(mapped)(x, y_2d)


# ------------------------------------------------------- class dispatch ----
#
# These go through the public `TensorProduct.__call__` rather than calling the
# op directly: with `tensor_product` set to `FUSED_MOSAIC_GPU` the class has to
# route through `_mgpu_eval`, and the answer has to match the `UNFUSED` path
# built from the same coefficients -- a reference independent of the `jnp` one
# above.


def _tp(in1: str, in2: str, out: str, mode: MixingMode) -> TensorProduct:
    return TensorProduct(
        (in1, in2),
        out,
        layout=Layout.TRAILING_CHANNELS,
        mode=mode,
        normalization="SQRT_DIM_OUT",
        sort=True,
    )


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_class_dispatch_matches_unfused(mode):
    irreps = "0e + 1o + 2e"
    with config.use(tensor_product=options.TensorProduct.FUSED_MOSAIC_GPU):
        fused = _tp(irreps, irreps, irreps, mode)
        x, y = operands(fused._mgpu_params(), 256, 128, mode)
        got = jax.jit(fused)(x, y)
        cotangent = jax.random.normal(jax.random.key(5), got.shape)
        got_cotangents = jax.jit(jax.vjp(jax.jit(fused), x, y)[1])(cotangent)

    with config.use(tensor_product=options.TensorProduct.UNFUSED):
        plain = _tp(irreps, irreps, irreps, mode)
        expected = jax.jit(plain)(x, y)
        expected_cotangents = jax.jit(jax.vjp(jax.jit(plain), x, y)[1])(cotangent)

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)
    for name, a, b in zip(("dx", "dy"), got_cotangents, expected_cotangents):
        assert a.shape == b.shape, f"{name} shape"
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=name)


def test_params_that_compare_equal_hash_equal():
    """`params` keys `custom_vjp`'s nondiff cache, so `==` must imply equal hashes.

    Equality compares the coefficients by value, so a different dtype or a
    signed zero must not change the hash either.
    """
    import dataclasses

    params = build_params("0e + 1o", "0e + 1o", "0e + 1o", MixingMode.OUTER)
    values = params.values.astype(np.float32)
    values[0] = -0.0
    signed_zero = values.copy()
    signed_zero[0] = 0.0
    variants = [
        dataclasses.replace(params, values=values),
        dataclasses.replace(params, values=values.astype(np.float64)),
        dataclasses.replace(params, values=signed_zero),
        dataclasses.replace(
            params, values=values, indices=params.indices.astype(np.int64)
        ),
        dataclasses.replace(
            params, values=values, indices=params.indices.astype(np.int32)
        ),
    ]
    for variant in variants[1:]:
        assert variant == variants[0]
        assert hash(variant) == hash(variants[0])
