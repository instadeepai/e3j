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

"""Shapes, validation and blocking shared by the Mosaic GPU tensor product kernels."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import mosaic_gpu as plgpu

from e3j.pallas_ops.tensor_product.mosaic_gpu.params import (
    PallasMosaicGPUTensorProductParams,
)

# `used_input_coordinates` is re-exported for the kernels.
from e3j.pallas_ops.utils.mosaic_gpu import (  # noqa: F401
    MAX_COPY_ELEMS_PER_AXIS,
    UNTILED_COPY_MIN_BYTES,
    WARPGROUP_LANES,
    pad_trailing_axis_for_untiled_copy,
    shared_memory_budget,
    used_input_coordinates,
)

#: Name of the flat grid axis. The grid stays one-dimensional because only the
#: first CUDA grid dimension accepts more than 65535 blocks; each CTA splits its
#: index into a row block and a channel block with :func:`block_indices`.
BLOCK_AXIS = "block"

#: Default `rows_per_block`. On an H100 at `l_max=3` with 128 channels, one CTA
#: per row already fills the device and was the best or tied setting; 4 to 16
#: rows were up to 10% slower.
DEFAULT_ROWS_PER_BLOCK = 1


class Shapes(NamedTuple):
    """Static shapes of one tensor product.

    Attributes:
        batch: Rows of the merged batch axis.
        x_dim: Coordinates of the `x` space.
        y_dim: Coordinates of the `y` space.
        y_dim_padded: In `OUTER` mode, `y_dim` rounded up to a legal untiled
            copy width; `y_dim` in `MAP` mode.
        z_dim: Coordinates of the output space.
        channels: Total channel count, before channel blocking.
        outer: True when `y` has no channel axis (`OUTER` mode).
    """

    batch: int
    x_dim: int
    y_dim: int
    y_dim_padded: int
    z_dim: int
    channels: int
    outer: bool


def validate_operands(
    x: jax.Array,
    y: jax.Array,
    params: PallasMosaicGPUTensorProductParams,
) -> Shapes:
    """Check the operands against `params` and the kernels' restrictions.

    `x` is `(batch, x_dim, channels)`; `y` is `(batch, y_dim, channels)` in
    `MAP` mode and `(batch, y_dim)` in `OUTER` mode. Operands must be float32,
    `channels` a multiple of `WARPGROUP_LANES` (128), and every coordinate axis,
    including the output's, must pass :func:`check_staged_axis`.

    Raises:
        ValueError: If the operands disagree with each other or with `params`.
        NotImplementedError: On a dtype, channel count or coordinate width the
            kernels do not support.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be (batch, x_dim, channels), got {x.shape}")
    batch, x_dim, channels = x.shape
    outer = params.outer

    if outer:
        if y.ndim != 2:
            raise ValueError(
                f"{params.mode} mode expects y of shape (batch, y_dim), "
                f"got {y.shape}"
            )
    elif y.ndim != 3:
        raise ValueError(
            f"{params.mode} mode expects y of shape (batch, y_dim, channels), "
            f"got {y.shape}"
        )
    elif y.shape[2] != channels:
        raise ValueError(
            f"x and y disagree on the channel count: {channels} vs {y.shape[2]}"
        )
    if y.shape[0] != batch:
        raise ValueError(f"x and y disagree on the batch size: {batch} vs {y.shape[0]}")

    y_dim = y.shape[1]
    if x_dim != params.x_space.dim:
        raise ValueError(
            f"x has x_dim={x_dim}, but x_space is {params.x_space} of dim "
            f"{params.x_space.dim}"
        )
    if y_dim != params.y_space.dim:
        raise ValueError(
            f"y has y_dim={y_dim}, but y_space is {params.y_space} of dim "
            f"{params.y_space.dim}"
        )
    for name, operand in (("x", x), ("y", y)):
        if operand.dtype != jnp.float32:
            raise NotImplementedError(
                f"the Mosaic GPU tensor product is float32-only, {name} is "
                f"{operand.dtype}"
            )

    if channels < WARPGROUP_LANES or channels % WARPGROUP_LANES:
        raise NotImplementedError(
            f"{params.mode} mode needs a positive multiple of {WARPGROUP_LANES} "
            f"channels, got {channels}: every SMEM value the kernel "
            f"materializes is a (channel_block,) vector, which Mosaic GPU only "
            f"vectorizes at that granularity"
        )

    elems = UNTILED_COPY_MIN_BYTES // y.dtype.itemsize
    y_dim_padded = -(-y_dim // elems) * elems if outer else y_dim
    check_staged_axis("x_space", x_dim)
    check_staged_axis("y_space", y_dim, staged=y_dim_padded)
    # Only the backward pass stages `z`, but the forward checks it too so that
    # every shape it accepts can also be differentiated.
    check_staged_axis("z_space", params.z_space.dim)
    return Shapes(
        batch=batch,
        x_dim=x_dim,
        y_dim=y_dim,
        y_dim_padded=y_dim_padded,
        z_dim=params.z_space.dim,
        channels=channels,
        outer=outer,
    )


def check_staged_axis(name: str, dim: int, staged: int | None = None) -> None:
    """Raise `NotImplementedError` if a staged axis exceeds `MAX_COPY_ELEMS_PER_AXIS`.

    Pass `staged` when the copied width differs from `dim`, as for `OUTER`'s
    padded `y`.
    """
    width = dim if staged is None else staged
    if width <= MAX_COPY_ELEMS_PER_AXIS:
        return
    padded = "" if staged is None or staged == dim else f" (staged as {staged})"
    raise NotImplementedError(
        f"{name} has {dim} coordinates{padded}, more than the "
        f"{MAX_COPY_ELEMS_PER_AXIS} an async copy can move along one dimension. "
        f"These kernels stage a whole row of each operand, and blocking over the "
        f"coordinate axis is not implemented"
    )


def resolve_channel_block(channels: int, channel_block: int | None) -> int:
    """Resolve `*_config.channel_block`; `None` means `WARPGROUP_LANES`.

    Raises:
        ValueError: If `channel_block` does not divide `channels`, is not a
            multiple of `WARPGROUP_LANES`, or exceeds
            `MAX_COPY_ELEMS_PER_AXIS`.
    """
    if channel_block is None:
        return WARPGROUP_LANES
    if channels % channel_block:
        raise ValueError(
            f"channel_block={channel_block} must divide channels={channels}"
        )
    if channel_block % WARPGROUP_LANES:
        raise ValueError(
            f"channel_block={channel_block} must be a multiple of "
            f"{WARPGROUP_LANES}, the granularity Mosaic GPU vectorizes at"
        )
    if channel_block > MAX_COPY_ELEMS_PER_AXIS:
        raise ValueError(
            f"channel_block={channel_block} is larger than the "
            f"{MAX_COPY_ELEMS_PER_AXIS} elements an async copy can move along "
            f"one axis"
        )
    return channel_block


def check_staged_smem(
    pass_name: str,
    coordinates: int,
    channel_block: int,
    pipeline_depth: int,
    itemsize: int,
    extra_bytes: int = 0,
) -> None:
    """Raise `ValueError` if the staged pipeline slots exceed per-CTA SMEM.

    One pipeline slot takes `coordinates * channel_block * itemsize` bytes,
    where `coordinates` is the total over all staged operands. `extra_bytes`
    covers anything else kept in SMEM. Barriers and alignment are left out, so
    the estimate is a lower bound and never rejects a plan that would compile.
    The error says which config field to lower.
    """
    budget = shared_memory_budget()
    staged = pipeline_depth * coordinates * channel_block * itemsize + extra_bytes
    if staged <= budget:
        return
    raise ValueError(
        f"the {pass_name} pass would stage {staged} bytes of shared memory per "
        f"CTA, but only {budget} are available ({pipeline_depth} pipeline slots "
        f"of {coordinates} coordinates at {channel_block} channels). Lower "
        f"pipeline_depth or channel_block in the pass's config"
    )


def block_indices(n_channel_blocks: int) -> tuple[jax.Array, jax.Array]:
    """Return this CTA's `(row block, channel block)` from the flat grid index.

    The channel block varies fastest, so consecutive CTAs write one contiguous
    stretch of the output. This was about 2% faster than the reverse order.
    """
    index = jax.lax.axis_index(BLOCK_AXIS)
    # `lax.rem` does not promote weak types, so without the cast x64 mode fails.
    return index // n_channel_blocks, jax.lax.rem(index, np.int32(n_channel_blocks))


def y_block_spec(
    shapes: Shapes,
    channel_block: int,
    first_row: jax.Array,
    block_index: jax.Array,
) -> plgpu.BlockSpec:
    """Return the pipeline block spec for one row of `y`.

    In `MAP` mode `y` is blocked like `x`. In `OUTER` mode each block is a single
    padded row of scalars.
    """
    if shapes.outer:
        return plgpu.BlockSpec(
            (None, shapes.y_dim_padded),
            lambda step: (first_row + step, 0),
        )
    return plgpu.BlockSpec(
        (None, shapes.y_dim, channel_block),
        lambda step: (first_row + step, 0, block_index),
    )


def resolve_rows_per_block(batch: int, rows_per_block: int | None) -> int:
    """Resolve `*_config.rows_per_block`, which must divide `batch`.

    `None` takes the gcd of `batch` and :data:`DEFAULT_ROWS_PER_BLOCK`.

    Raises:
        ValueError: If an explicit `rows_per_block` does not divide `batch`.
    """
    if rows_per_block is None:
        return math.gcd(batch, DEFAULT_ROWS_PER_BLOCK)
    if batch % rows_per_block:
        raise ValueError(
            f"rows_per_block={rows_per_block} must divide the batch size "
            f"{batch}, since each CTA processes exactly that many rows"
        )
    return rows_per_block


def pad_y_rows(y: jax.Array, shapes: Shapes) -> jax.Array:
    """Zero-pad `OUTER`'s `y` so a one-row transfer is a legal untiled copy."""
    if not shapes.outer or shapes.y_dim_padded == shapes.y_dim:
        return y
    return pad_trailing_axis_for_untiled_copy(y)
