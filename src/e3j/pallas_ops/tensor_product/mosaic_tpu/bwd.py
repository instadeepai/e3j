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

"""Backward pass of the tensor product using Pallas Mosaic TPU.

Supports the TRAILING_CHANNELS layout (MAP and OUTER modes).
"""

import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from e3j.pallas_ops.tensor_product.mosaic_tpu.params import (
    PallasMosaicTPUTensorProductParams,
    PallasMosaicTPUTensorProductParamsBwdConfig,
)
from e3j.pallas_ops.utils.coef import group_coef_by_xi
from e3j.pallas_ops.utils.mosaic_tpu import pad_for_tpu, select_batch_block_size
from e3j.utils import options


class BwdKernel(ABC):
    """Backward tensor-product kernel."""

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        dz: jax.Array,
        params: PallasMosaicTPUTensorProductParams,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the gradients `dx` and `dy` of the tensor product."""


class _BwdTrailingChannelKernel(BwdKernel):
    """TRAILING_CHANNELS backward tensor product on Mosaic TPU.

    Backpropagates `dz` through `z[zi] = sum_{(xi, yi, v)} v * y[yi] * x[xi]`
    (see `_FwdTrailingChannelKernel`). The same Clebsch-Gordan triples are
    regrouped by the *input* index `xi` (via `group_coef_by_xi`), so a single
    sweep over the triples produces both gradients::

        dx[xi] = sum_{(zi, yi, v)} v * y[yi] * dz[zi]
        dy[yi] = sum_{(zi, xi, v)} v * x[xi] * dz[zi]   (summed over xi)

    Products are elementwise over the trailing channel axis. `y` either
    broadcasts over the channel axis (OUTER) or is elementwise on it
    (MAP). That difference, plus how `dy` is laid out and reduced, is the only
    thing the two concrete subclasses change via the abstract `y_*` /
    `preload_y` / `write_dy` / `finalize_dy` hooks.

    Layout / shapes
    ---------------
    The TRAILING_CHANNELS layout puts the channel axis last, so the natural unit
    of work is a `(num_channels,)` vector lying along the TPU lane axis.

    Inputs from HBM after `pad_for_tpu`:

    - `x`: `(batch, x_dim, num_channels)`.
    - `y`: subclass-defined `(batch, y_dim)` for OUTER and `(batch, y_dim, num_channels)` for  MAP.
    - `dz`: passed in as `(batch, z_dim, num_channels)` but we expect the layout in memory to be
        `(z_dim, batch, num_channels)` because z is returned with this layout in fwd. So asume we can
        transpose it to have shape `(z_dim, batch, num_channels)` with a free bitcast.

    Outputs:

    - `dx`: `(batch, x_dim, num_channels)`
    - `dy`: subclass-defined (`dy_kernel_shape`) — OUTER `(batch, y_dim)`,
      MAP `(batch, y_dim, num_channels)`.

    `x`/`dz`/`y` are padded so `x_dim` / `num_channels` are tile multiples, the
    padding is sliced off the returned `dx` and `dy`.

    Algorithm
    ----------
    A single `pallas_call` runs over a megacore grid of `tpu_info.num_cores`
    with PARALLEL semantics, with all operands and outputs living in HBM.
    Inside each program, `emit_pipeline` walks a grid of `num_blocks` batch
    tiles of `batch_block_size` rows each. `core_axis=0` distributes tiles
    across cores, `batch_block_size` is chosen to fit the double-buffered VMEM
    working set, see `_vmem_bytes_per_batch_element`. `emit_pipeline`
    double-buffers the `dz` / `x` / `y` input tiles and the `dx` / `dy`
    output tiles so the next tile's DMA overlaps the current tile's compute.

    Per pipeline step (body), working in VMEM:
      1. Transpose the x tile from `(batch_block_size, x_dim, num_channels)` to `(x_dim, batch_block_size, num_channels)`
         so an `x[xi]` slice is a contiguous `(batch_block_size, num_channels)` operand
         making it the leading axis avoids a strided gather on every contribution.
      2. Preload each used `yi` into a `(batch_block_size, num_channels)` operand once
         (OUTER broadcasts the per-channel scalar across the channel axis, MAP slices it),
         so a `yi` reused across several `xi` is only materialized one time.
      3. Allocate fp32 `(batch_block_size, num_channels)` accumulators: one `dy_acc[yi]` per
         used `yi` (`yi_used = range(y_dim)`) and a `dx_rows` list with one
         entry per `xi`.
      4. For each `xi` and each contribution `(zi, yi, v)`, compute the shared
         term `vdz = v * dz[zi]` once, then accumulate `y[yi] * vdz` into the
         `xi` row of `dx` and `x[xi] * vdz` into `dy_acc[yi]`.
      5. Stack the `dx` rows to `(x_dim, batch_block_size, num_channels)` and transpose once
         back to the output's `(batch_block_size, x_dim, num_channels)` layout before storing.
      6. Store each `dy_acc[yi]` via `write_dy` — OUTER reduces over the channel axis to
         a per-channel scalar, MAP writes the full `(batch_block_size, num_channels)` vector.
    """

    # Only used for channel=32 as it was slowing down for channel=64
    _PACK_MIN_K = 4

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        dz: jax.Array,
        params: PallasMosaicTPUTensorProductParams,
    ) -> tuple[jax.Array, jax.Array]:
        bwd_config = params.bwd_config or PallasMosaicTPUTensorProductParamsBwdConfig()

        assert len(x.shape) == 3, f"x must be (batch, x_dim, channel), got {x.shape}"
        assert len(dz.shape) == 3, f"dz must be (batch, out_dim, channel), got {dz.shape}"
        self._validate_y(x, y)
        assert x.shape[0] == dz.shape[0], "Batch sizes must match for x and dz"
        assert x.shape[0] == y.shape[0], "Batch sizes must match for x and y"

        batch_size = x.shape[0]
        x_dim = x.shape[1]
        channels = x.shape[2]
        y_dim = y.shape[1]

        tpu_info = pltpu.get_tpu_info()
        num_kernel = tpu_info.num_cores

        n_batch_packed_in_channel = (
            128 // channels if channels < 128 and 128 % channels == 0 else 1
        )
        packing = (
            n_batch_packed_in_channel >= self._PACK_MIN_K
            and batch_size % n_batch_packed_in_channel == 0
            and (batch_size // n_batch_packed_in_channel) % num_kernel == 0
        )
        n_batch_packed_in_channel = n_batch_packed_in_channel if packing else 1
        if packing:
            packed_batch_size = batch_size // n_batch_packed_in_channel
            x = self._pack_rows(
                x, n_batch_packed_in_channel
            )  # (packed_batch_size, x_dim, k*channel == 128)
            dz = self._pack_rows(
                dz, n_batch_packed_in_channel
            )  # (packed_batch_size, out_dim, 128)
            y = self.pack_y(y, n_batch_packed_in_channel)

        x = pad_for_tpu(x)  # (kbatch, x_dim_padded, channel_padded)
        dz = pad_for_tpu(dz.transpose(1, 0, 2))  # (out_dim, kbatch, channel_padded)
        y = pad_for_tpu(y)  # OUTER: (kbatch, y_dim_p); MAP: (kbatch, y_dim_p, channel_p)

        # After packing the kernel runs on the shrunk batch ``packed_batch_size``.
        kernel_batch = x.shape[0]
        x_dim_padded = x.shape[1]
        z_dim = dz.shape[0]
        num_channels_padded = x.shape[2]
        dy_kernel_shape = self.dy_kernel_shape(y, kernel_batch)

        cg_by_xi = tuple(
            (xi, tuple((oi, yi, v) for oi, yi, v in contribs))
            for xi, contribs in group_coef_by_xi(params.indices, params.values).items()
        )
        yis = frozenset(yi for _, contribs in cg_by_xi for _, yi, _ in contribs)
        yi_used = range(y_dim)

        batch_block_size = select_batch_block_size(
            batch_size=kernel_batch,
            vmem_bytes_per_batch_element=self._vmem_bytes_per_batch_element(
                x_dim_padded,
                num_channels_padded,
                y,
                z_dim,
                len(yis),
                y_dim,
                x.dtype.itemsize,
            ),
            num_kernel=num_kernel,
            vmem_capacity_bytes=tpu_info.vmem_capacity_bytes,
            override=bwd_config.batch_block_size,
        )
        num_blocks = kernel_batch // batch_block_size

        def bwd_kernel(dz_ref, x_ref, y_ref, dx_ref, dy_ref):
            def body(dz_vmem, x_vmem_src, y_vmem, dx_vmem, dy_vmem):
                # dz already (out_dim, batch_block_size, channel); only x needs the in-kernel
                # transpose so the inner loop can index its leading dim for free.
                x_vmem = jax.lax.transpose(x_vmem_src[...], (1, 0, 2))
                pre_broadcast_y = self.preload_y(
                    y_vmem,
                    yis,
                    batch_block_size,
                    num_channels_padded,
                    n_batch_packed_in_channel,
                )
                zero = jnp.zeros(
                    (batch_block_size, num_channels_padded), jnp.float32
                )
                # Per-yi (batch_block_size, channel) accumulators. In OUTER they're reduced over
                # the channel axis before store; in MAP they're written out directly.
                dy_acc = {yi: zero for yi in yi_used}
                # Per-xi dx rows assembled in (x_dim, batch_block_size, channel) order, then
                # transposed once to the output's (batch_block_size, x_dim, channel) layout.
                dx_rows = [zero] * x_dim_padded

                for xi, contributions in cg_by_xi:
                    acc_dx = zero
                    x_slice = x_vmem[xi, :, :]
                    for oi, yi, v in contributions:
                        dz_slice = dz_vmem[oi, :, :]
                        vdz = v * dz_slice
                        acc_dx = acc_dx + pre_broadcast_y[yi] * vdz
                        dy_acc[yi] = dy_acc[yi] + x_slice * vdz
                    dx_rows[xi] = acc_dx

                dx_block = jnp.stack(dx_rows, axis=0)  # (x_dim, batch_block_size, channel)
                dx_vmem[...] = jax.lax.transpose(dx_block, (1, 0, 2)).astype(
                    dx_vmem.dtype
                )

                for yi in yi_used:
                    self.write_dy(dy_vmem, yi, dy_acc[yi], n_batch_packed_in_channel)

            pltpu.emit_pipeline(
                body,
                grid=(num_blocks,),
                in_specs=[
                    pl.BlockSpec(
                        block_shape=(
                            z_dim,
                            batch_block_size,
                            num_channels_padded,
                        ),
                        index_map=lambda i: (0, i, 0),
                    ),
                    pl.BlockSpec(
                        block_shape=(
                            batch_block_size,
                            x_dim_padded,
                            num_channels_padded,
                        ),
                        index_map=lambda i: (i, 0, 0),
                    ),
                    pl.BlockSpec(
                        block_shape=self.y_block_shape(y, batch_block_size),
                        index_map=self.y_index_map,
                    ),
                ],
                out_specs=[
                    pl.BlockSpec(
                        block_shape=(
                            batch_block_size,
                            x_dim_padded,
                            num_channels_padded,
                        ),
                        index_map=lambda i: (i, 0, 0),
                    ),
                    pl.BlockSpec(
                        block_shape=self.y_block_shape(y, batch_block_size),
                        index_map=self.y_index_map,
                    ),
                ],
                core_axis=0,
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
            )(dz_ref, x_ref, y_ref, dx_ref, dy_ref)

        dx, dy = pl.pallas_call(
            bwd_kernel,
            out_shape=[
                jax.ShapeDtypeStruct(
                    (kernel_batch, x_dim_padded, num_channels_padded), x.dtype
                ),
                jax.ShapeDtypeStruct(dy_kernel_shape, x.dtype),
            ],
            in_specs=[
                pl.BlockSpec(a.shape, memory_space=pltpu.HBM) for a in (dz, x, y)
            ],
            out_specs=[
                pl.BlockSpec(
                    (kernel_batch, x_dim_padded, num_channels_padded),
                    memory_space=pltpu.HBM,
                ),
                pl.BlockSpec(dy_kernel_shape, memory_space=pltpu.HBM),
            ],
            grid=(num_kernel,),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,)
            ),
            name="tensor_product_mtpu_trailing_channel_bwd",
        )(dz, x, y)

        dx = self._unpack_dx(dx, batch_size, x_dim, channels, n_batch_packed_in_channel)
        return dx, self.finalize_dy(
            dy, y_dim, channels, batch_size, n_batch_packed_in_channel
        )

    @staticmethod
    def _pack_rows(x: jax.Array, n_batch_packed_in_channel: int) -> jax.Array:
        """Fold ``n_batch_packed_in_channel`` batch rows into the lane axis."""
        batch_size, lm, channels = x.shape[0], x.shape[1], x.shape[2]
        assert batch_size % n_batch_packed_in_channel == 0
        packed_batch_size = batch_size // n_batch_packed_in_channel
        x = x.reshape(packed_batch_size, n_batch_packed_in_channel, lm, channels)
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(packed_batch_size, lm, n_batch_packed_in_channel * channels)

    def _unpack_dx(
        self,
        dx: jax.Array,
        batch: int,
        x_dim: int,
        channels: int,
        n_batch_packed_in_channel: int,
    ) -> jax.Array:
        """Reconstruct ``dx (batch, x_dim, channel)`` from the kernel output."""
        if n_batch_packed_in_channel == 1:
            return dx[:, :x_dim, :channels]
        packed_batch_size, x_dim_p = dx.shape[0], dx.shape[1]
        # (packed_batch_size, x_dim_p, 128) -> (packed_batch_size, x_dim_p, k, channel): split lane (free).
        dx = dx.reshape(packed_batch_size, x_dim_p, n_batch_packed_in_channel, channels)
        # Move k next to packed_batch_size then merge -> (batch, x_dim_p, channel). This crosses
        # x_dim so it is a genuine sub-128-lane relayout (the un-fold tax).
        dx = dx.transpose(0, 2, 1, 3).reshape(
            packed_batch_size * n_batch_packed_in_channel, x_dim_p, channels
        )
        return dx[:, :x_dim, :channels]

    def _vmem_bytes_per_batch_element(
        self,
        x_dim_padded: int,
        channels_padded: int,
        y_padded: jax.Array,
        z_dim: int,
        num_yis: int,
        y_dim: int,
        dtype_bytes: int,
    ) -> int:
        """VMEM bytes consumed per batch element (pipeline double-buffered + body intermediates).

        Pipeline: dz, x (in+out), y (in+out) — all double-buffered by emit_pipeline.
        Body intermediates: transposed x, preloaded y slices, dy accumulators,
        dx_rows list, dx_block stack.
        """
        y_elems = math.prod(self.y_block_shape(y_padded, 1))
        pipeline = (
            2
            * dtype_bytes
            * (
                z_dim * channels_padded  # dz in
                + 2 * x_dim_padded * channels_padded  # x in + dx out
                + 2 * y_elems  # y in + dy out
            )
        )
        body = (
            4
            * channels_padded
            * (
                x_dim_padded  # x_vmem transposed
                + num_yis  # preloaded y slices
                + y_dim  # dy accumulators (one per yi_used = range(y_dim))
                + 2 * x_dim_padded  # dx_rows list + dx_block stack
            )
        )
        return pipeline + body

    @abstractmethod
    def _validate_y(self, x: jax.Array, y: jax.Array) -> None:
        """Assert `y` has the rank/shape this mode expects relative to `x`."""

    @abstractmethod
    def pack_y(self, y: jax.Array, n_batch_packed_in_channel: int) -> jax.Array:
        """Fold ``k`` batch rows of ``y`` to match :meth:`_pack_x`'s lane layout."""

    @abstractmethod
    def dy_kernel_shape(
        self, y_padded: jax.Array, kernel_batch: int
    ) -> tuple[int, ...]:
        """Return the full HBM shape of the ``dy`` output for this mode."""

    @abstractmethod
    def y_block_shape(
        self, y_padded: jax.Array, batch_block_size: int
    ) -> tuple[int, ...]:
        """Per-tile `BlockSpec` shape (shared by `y` input and `dy` output)."""

    @abstractmethod
    def y_index_map(self, i: int) -> tuple[int, ...]:
        """Map batch-tile index `i` to the y/dy block's grid offset."""

    @abstractmethod
    def preload_y(
        self,
        y_vmem: jax.Array,
        yis: frozenset[int],
        batch_block_size: int,
        num_channels_vmem: int,
        n_batch_packed_in_channel: int,
    ) -> dict[int, jax.Array]:
        """Materialize a map from ``yi`` to ``(batch_block_size, num_channels)``.
        We materialize this ones and then all access don't need any expensive broadcast.

        ``n_batch_packed_in_channel`` describes the active lane packing (``== 1`` when off).
        """

    @abstractmethod
    def write_dy(
        self,
        dy_vmem: jax.Array,
        yi: int,
        dy_acc: jax.Array,
        n_batch_packed_in_channel: int,
    ) -> None:
        """Store the accumulated ``(batch_block_size, channel)`` ``dy_acc`` for component ``yi``.

        ``n_batch_packed_in_channel`` describes the active lane packing (``== 1`` when off).
        """

    @abstractmethod
    def finalize_dy(
        self,
        dy: jax.Array,
        y_dim: int,
        num_channels: int,
        batch: int,
        n_batch_packed_in_channel: int,
    ) -> jax.Array:
        """Trim/unpack the raw dy buffer back to the user-facing shape."""


class _BwdTrailingChannelOuterKernel(_BwdTrailingChannelKernel):
    """OUTER (y_channel=1): y is (batch, y_dim); dy reduced over the channel axis."""

    def _validate_y(self, x: jax.Array, y: jax.Array) -> None:
        assert (
            len(y.shape) == 2
        ), f"OUTER mode requires 2D y (batch, y_dim), got {y.shape}"

    def pack_y(self, y: jax.Array, n_batch_packed_in_channel: int) -> jax.Array:
        """pack y:``(batch, y_dim)`` -> ``(packed_batch_size, y_dim * k)``."""
        batch_size, y_lm_dim = y.shape[0], y.shape[1]
        assert batch_size % n_batch_packed_in_channel == 0
        packed_batch_size = batch_size // n_batch_packed_in_channel

        y = y.reshape(
            packed_batch_size, n_batch_packed_in_channel, y_lm_dim
        )  # (packed_batch_size, k, y_dim)
        y = y.transpose(0, 2, 1)  # (packed_batch_size, y_dim, k)
        return y.reshape(
            packed_batch_size, y_lm_dim * n_batch_packed_in_channel
        )  # (packed_batch_size, y_dim * k)

    def dy_kernel_shape(
        self, y_padded: jax.Array, kernel_batch: int
    ) -> tuple[int, ...]:
        return (kernel_batch, y_padded.shape[1])

    def y_block_shape(
        self, y_padded: jax.Array, batch_block_size: int
    ) -> tuple[int, ...]:
        return (batch_block_size, y_padded.shape[1])

    def y_index_map(self, i: int) -> tuple[int, ...]:
        return (i, 0)

    def preload_y(
        self,
        y_vmem: jax.Array,
        yis: frozenset[int],
        batch_block_size: int,
        num_channels_vmem: int,
        n_batch_packed_in_channel: int,
    ) -> dict[int, jax.Array]:
        if n_batch_packed_in_channel > 1:
            real_channels = num_channels_vmem // n_batch_packed_in_channel

            def _broadcast(yi: int) -> jax.Array:
                seg = y_vmem[
                    :, yi * n_batch_packed_in_channel : (yi + 1) * n_batch_packed_in_channel
                ]  # (batch_block_size, k)
                seg = jnp.broadcast_to(
                    seg[:, :, None], (batch_block_size, n_batch_packed_in_channel, real_channels)
                )
                return seg.reshape(
                    batch_block_size, n_batch_packed_in_channel * real_channels
                )  # (batch_block_size, 128 == channel)

            return {yi: _broadcast(yi) for yi in yis}
        return {
            yi: jnp.broadcast_to(
                y_vmem[:, yi][:, None], (batch_block_size, num_channels_vmem)
            )
            for yi in yis
        }

    def write_dy(
        self,
        dy_vmem: jax.Array,
        yi: int,
        dy_acc: jax.Array,
        n_batch_packed_in_channel: int,
    ) -> None:
        # OUTER's dy is (batch, y_dim): reduce dy_acc over the channel axis.
        if n_batch_packed_in_channel > 1:
            real_channels = dy_acc.shape[-1] // n_batch_packed_in_channel
            # Reduce per k
            for j in range(n_batch_packed_in_channel):
                seg = dy_acc[
                    :, j * real_channels : (j + 1) * real_channels
                ]  # (batch_block_size, real_channels)
                dy_vmem[:, yi * n_batch_packed_in_channel + j] = jnp.sum(
                    seg, axis=-1
                ).astype(dy_vmem.dtype)
            return
        dy_vmem[:, yi] = jnp.sum(dy_acc, axis=-1).astype(dy_vmem.dtype)

    def finalize_dy(
        self,
        dy: jax.Array,
        y_dim: int,
        num_channels: int,
        batch: int,
        n_batch_packed_in_channel: int,
    ) -> jax.Array:
        if n_batch_packed_in_channel == 1:
            return dy[:, :y_dim]
        packed_batch_size = dy.shape[0]
        dy = dy[
            :, : y_dim * n_batch_packed_in_channel
        ]  # (packed_batch_size, y_dim * n_batch_packed_in_channel)
        dy = dy.reshape(
            packed_batch_size, y_dim, n_batch_packed_in_channel
        )  # (packed_batch_size, y_dim, n_batch_packed_in_channel)
        dy = dy.transpose(0, 2, 1)  # (packed_batch_size, n_batch_packed_in_channel, y_dim)
        return dy.reshape(
            packed_batch_size * n_batch_packed_in_channel, y_dim
        )  # (batch, y_dim)


class _BwdTrailingChannelMapKernel(_BwdTrailingChannelKernel):
    """MAP: y is (batch, y_dim, channel); dy is elementwise over the channel axis."""

    def _validate_y(self, x: jax.Array, y: jax.Array) -> None:
        assert (
            len(y.shape) == 3 and x.shape[-1] == y.shape[-1]
        ), f"MAP mode requires 3D y with matching channel, got x={x.shape} y={y.shape}"

    def pack_y(self, y: jax.Array, n_batch_packed_in_channel: int) -> jax.Array:
        """MAP y is ``(batch, y_dim, channel)`` -> packed identically to x: ``(packed_batch_size, y_dim, 128)``."""
        return self._pack_rows(y, n_batch_packed_in_channel)

    def dy_kernel_shape(
        self, y_padded: jax.Array, kernel_batch: int
    ) -> tuple[int, ...]:
        return (kernel_batch, y_padded.shape[1], y_padded.shape[2])

    def y_block_shape(
        self, y_padded: jax.Array, batch_block_size: int
    ) -> tuple[int, ...]:
        return (batch_block_size, y_padded.shape[1], y_padded.shape[2])

    def y_index_map(self, i: int) -> tuple[int, ...]:
        return (i, 0, 0)

    def preload_y(
        self,
        y_vmem: jax.Array,
        yis: frozenset[int],
        batch_block_size: int,
        num_channels_vmem: int,
        n_batch_packed_in_channel: int,
    ) -> dict[int, jax.Array]:
        return {yi: y_vmem[:, yi, :] for yi in yis}

    def write_dy(
        self,
        dy_vmem: jax.Array,
        yi: int,
        dy_acc: jax.Array,
        n_batch_packed_in_channel: int,
    ) -> None:
        dy_vmem[:, yi, :] = dy_acc.astype(dy_vmem.dtype)

    def finalize_dy(
        self,
        dy: jax.Array,
        y_dim: int,
        num_channels: int,
        batch: int,
        n_batch_packed_in_channel: int,
    ) -> jax.Array:
        if n_batch_packed_in_channel == 1:
            return dy[:, :y_dim, :num_channels]
        # dy is packed (packed_batch_size, y_dim_padded, 128); unpack like dx (it shares x's
        # lane layout) back to (batch, y_dim, channel).
        return self._unpack_dx(dy, batch, y_dim, num_channels, n_batch_packed_in_channel)


def _tensor_product_kernel_mosaic_tpu_bwd(
    x: jax.Array,
    y: jax.Array,
    dz: jax.Array,
    params: PallasMosaicTPUTensorProductParams,
) -> tuple[jax.Array, jax.Array]:
    """Backward pass dispatching on `params.layout`/`params.mode`."""
    if params.layout == options.Layout.TRAILING_CHANNELS:
        if params.mode == options.TPMode.MAP:
            return _BwdTrailingChannelMapKernel()(x, y, dz, params)
        if params.mode == options.TPMode.OUTER:
            return _BwdTrailingChannelOuterKernel()(x, y, dz, params)
    raise NotImplementedError(f"Mosaic TPU TP bwd unsupported for {params}")
