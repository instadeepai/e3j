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

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from e3j.pallas_ops.tensor_product.mosaic_tpu.params import (
    BWD_BATCH_BLOCK_SIZE_CANDIDATES,
    PallasMosaicTPUTensorProductParams,
    PallasMosaicTPUTensorProductParamsBwdConfig,
)
from e3j.pallas_ops.utils.coef import group_coef_by_xi
from e3j.pallas_ops.utils.mosaic_tpu import (
    pad_for_tpu,
    round_up,
    select_batch_block_size,
)
from e3j.utils import options


class BwdKernel(ABC):
    """Backward tensor-product kernel."""

    @abstractmethod
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
    Inside each program, `emit_pipeline` walks a grid of `n_batch_blocks` batch
    tiles of `batch_block_size` rows each. `core_axis=0` distributes tiles
    across cores, `batch_block_size` is chosen to fit the double-buffered VMEM
    working set, see `_vmem_bytes_per_batch_element`. `emit_pipeline`
    double-buffers the `dz` / `x` / `y` input tiles and the `dx` / `dy`
    output tiles so the next tile's DMA overlaps the current tile's compute.

    `x` / `dx` move as whole `(batch_block_size, x_dim, 128)` blocks, one per
    lane tile, and each `x[xi]` / `dx[xi]` row is read and written with a
    sublane-strided load / store, so no transpose is materialized. The body
    loops over 8-row sub-tiles so the accumulators stay in registers. Per sub-tile:
      1. Load each used `yi` as a `(8, num_channels)` operand (OUTER broadcasts
         the per-channel scalar, MAP slices it).
      2. For each `xi` and each `yi`, form `t = sum_zi v * dz[zi]` once, then
         accumulate `y[yi] * t` into `dx[xi]` and `x[xi] * t` into `dy_acc[yi]`.
      3. Store each `dy_acc[yi]` via `write_dy` — OUTER reduces over the channel axis to
         a per-channel scalar, MAP writes the full `(8, num_channels)` vector.
    """

    _DIM_LEADING = False

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
        dim_leading = (
            self._DIM_LEADING and x.shape[-2] % 8 != 0 and x.shape[-1] % 128 == 0
        )

        assert len(x.shape) == 3, f"x must be (batch, x_dim, channel), got {x.shape}"
        assert (
            len(dz.shape) == 3
        ), f"dz must be (batch, out_dim, channel), got {dz.shape}"
        self._validate_y(x, y)
        assert x.shape[0] == dz.shape[0], "Batch sizes must match for x and dz"
        assert x.shape[0] == y.shape[0], "Batch sizes must match for x and y"

        batch_size = x.shape[0]
        dtype = x.dtype
        x_dim = x.shape[1]
        channels = x.shape[2]
        y_dim = y.shape[1]

        tpu_info = pltpu.get_tpu_info()
        num_kernel = tpu_info.num_cores

        n_batch_packed_in_channel = (
            128 // channels if channels < 128 and 128 % channels == 0 else 1
        )
        packing = (
            not dim_leading
            and n_batch_packed_in_channel >= self._PACK_MIN_K
            and batch_size % n_batch_packed_in_channel == 0
            and (batch_size // n_batch_packed_in_channel) % num_kernel == 0
        )
        n_batch_packed_in_channel = n_batch_packed_in_channel if packing else 1
        if packing:
            x = self._pack_rows(
                x, n_batch_packed_in_channel
            )  # (packed_batch_size, x_dim, k*channel == 128)
            dz = self._pack_rows(
                dz, n_batch_packed_in_channel
            )  # (packed_batch_size, out_dim, 128)
            y = self.pack_y(y, n_batch_packed_in_channel)

        if dim_leading:
            x = pad_for_tpu(jnp.transpose(x, (1, 0, 2)))
            y = self._orient_y_dim_leading(y)
            dz = pad_for_tpu(jnp.transpose(dz, (1, 0, 2)))
        else:
            # Mosaic has no strided loads of packed dtypes.
            x = pad_for_tpu(x.astype(jnp.float32))
            dz = pad_for_tpu(dz.transpose(1, 0, 2))
            y = pad_for_tpu(y)

        z_dim = dz.shape[0]
        num_channels_padded = x.shape[-1]
        if dim_leading:
            kernel_batch = x.shape[1]
            x_dim_padded = x.shape[0]
            dy_kernel_shape = self._dy_kernel_shape_dim_leading(y, kernel_batch)
        else:
            kernel_batch = x.shape[0]
            x_dim_padded = x.shape[1]
            dy_kernel_shape = self.dy_kernel_shape(y, kernel_batch)

        pairs_by_xi = []
        for xi, contribs in group_coef_by_xi(params.indices, params.values).items():
            by_yi = {}
            for zi, yi, v in contribs:
                by_yi.setdefault(yi, []).append((zi, v))
            pairs_by_xi.append((xi, tuple(by_yi.items())))
        xis = frozenset(xi for xi, _ in pairs_by_xi)
        yis = frozenset(yi for _, pairs in pairs_by_xi for yi, _ in pairs)

        # Packed y/dy relayouts across lanes, so do them once per block, not per sub-tile.
        staged = n_batch_packed_in_channel > 1
        n_pairs = sum(len(pairs) for _, pairs in pairs_by_xi)
        batch_block_size = select_batch_block_size(
            batch_size=kernel_batch,
            vmem_bytes_per_batch_element=self._vmem_bytes_per_batch_element(
                x_dim_padded,
                num_channels_padded,
                round_up(y.size // kernel_batch, 128),
                z_dim,
                (
                    2 * y_dim
                    + self.staging_rows(
                        len(yis), y_dim, y.shape[1], n_batch_packed_in_channel
                    )
                    if staged
                    else 0
                ),
                x.dtype.itemsize,
            ),
            batch_block_size_candidates=BWD_BATCH_BLOCK_SIZE_CANDIDATES,
            num_kernel=num_kernel,
            # Mosaic may hoist every `vdz` of a sub-tile and spill them: one
            # (sub_rows, num_channels) tile per (xi, yi) pair, whatever the block size.
            vmem_capacity_bytes=tpu_info.vmem_capacity_bytes
            - 4 * 8 * num_channels_padded * n_pairs,
            override=bwd_config.batch_block_size,
        )
        n_batch_blocks = kernel_batch // batch_block_size
        # One native sublane tile of dz: 8 fp32 or 16 bf16 rows.
        sub_rows = min(32 // dz.dtype.itemsize, batch_block_size)
        n_lane_tiles = 1 if dim_leading else num_channels_padded // 128

        def bwd_kernel(dz_ref, x_ref, y_ref, dx_ref, dy_ref, *scratch):
            def rows(xi, start, n):
                """Index of `n` batch rows of component `xi` in an x/dx block."""
                if dim_leading:
                    return xi, pl.ds(start, n)
                # (batch * x_dim, 128) blocks: the rows of `xi` are x_dim apart.
                return pl.ds(start * x_dim_padded + xi, n, stride=x_dim_padded)

            def load_x(x_tiles, xi, start, n):
                parts = [t[rows(xi, start, n)] for t in x_tiles]
                return parts[0] if len(parts) == 1 else jnp.concatenate(parts, -1)

            def store_dx(dx_tiles, xi, start, n, val):
                w = val.shape[-1] // len(dx_tiles)
                for t, ref in enumerate(dx_tiles):
                    ref[rows(xi, start, n)] = val[:, t * w : (t + 1) * w]

            def body(dz_vmem, *refs):
                x_tiles, dx_tiles = refs[:n_lane_tiles], refs[n_lane_tiles + 1 : -1]
                if not dim_leading:
                    rows_2d = (batch_block_size * x_dim_padded, 128)
                    x_tiles = [t.reshape(*rows_2d) for t in x_tiles]
                    dx_tiles = [t.reshape(*rows_2d) for t in dx_tiles]
                y_vmem, dy_vmem = refs[n_lane_tiles], refs[-1]
                if staged:
                    y_scratch, dy_scratch = scratch
                    pre_broadcast_y = self.preload_y(
                        y_vmem,
                        yis,
                        batch_block_size,
                        num_channels_padded,
                        n_batch_packed_in_channel,
                    )
                    for yi in yis:
                        y_scratch[yi] = pre_broadcast_y[yi].astype(jnp.float32)
                for xi in range(x_dim_padded):
                    if xi not in xis:
                        store_dx(
                            dx_tiles,
                            xi,
                            0,
                            batch_block_size,
                            jnp.zeros((batch_block_size, num_channels_padded), x.dtype),
                        )

                def sub_tile(r, carry):
                    r0 = pl.multiple_of(r * sub_rows, sub_rows)
                    rs = pl.ds(r0, sub_rows)
                    if staged:
                        pre_broadcast_y = {yi: y_scratch[yi, rs, :] for yi in yis}
                    elif dim_leading:
                        y_sub = self._batch_rows_dim_leading(y_vmem, rs)
                        dy_sub = self._batch_rows_dim_leading(dy_vmem, rs)
                        pre_broadcast_y = self._load_y_dim_leading(
                            y_sub, yis, sub_rows, num_channels_padded
                        )
                    else:
                        pre_broadcast_y = self.preload_y(
                            y_vmem.at[rs], yis, sub_rows, num_channels_padded, 1
                        )
                    zero = jnp.zeros((sub_rows, num_channels_padded), jnp.float32)
                    dy_acc = [zero] * y_dim
                    for xi, pairs in pairs_by_xi:
                        x_slice = load_x(x_tiles, xi, r0, sub_rows)
                        acc_dx = zero
                        for yi, terms in pairs:
                            vdz = terms[0][1] * dz_vmem[terms[0][0], rs, :]
                            for zi, v in terms[1:]:
                                vdz = vdz + v * dz_vmem[zi, rs, :]
                            acc_dx = acc_dx + pre_broadcast_y[yi] * vdz
                            dy_acc[yi] = dy_acc[yi] + x_slice * vdz
                        store_dx(dx_tiles, xi, r0, sub_rows, acc_dx.astype(x.dtype))
                    for yi in range(y_dim):
                        if staged:
                            dy_scratch[yi, rs, :] = dy_acc[yi]
                        elif dim_leading:
                            self._store_dy_dim_leading(dy_sub, yi, dy_acc[yi])
                        else:
                            self.write_dy(dy_vmem.at[rs], yi, dy_acc[yi], 1)
                    return carry

                jax.lax.fori_loop(0, batch_block_size // sub_rows, sub_tile, 0)
                if staged:
                    for yi in range(y_dim):
                        self.write_dy(
                            dy_vmem, yi, dy_scratch[yi], n_batch_packed_in_channel
                        )

            dz_spec = pl.BlockSpec(
                block_shape=(z_dim, batch_block_size, num_channels_padded),
                index_map=lambda i: (0, i, 0),
            )
            if dim_leading:
                x_specs = [
                    pl.BlockSpec(
                        block_shape=(
                            x_dim_padded,
                            batch_block_size,
                            num_channels_padded,
                        ),
                        index_map=lambda i: (0, i, 0),
                    )
                ]
                y_spec = self._y_blockspec_dim_leading(
                    y, batch_block_size, num_channels_padded
                )
            else:
                # One block per 128-lane tile: strided loads need a 128-lane memref.
                x_specs = [
                    pl.BlockSpec(
                        block_shape=(batch_block_size, x_dim_padded, 128),
                        index_map=lambda i, t=t: (i, 0, t),
                    )
                    for t in range(n_lane_tiles)
                ]
                y_spec = pl.BlockSpec(
                    block_shape=self.y_block_shape(y, batch_block_size),
                    index_map=self.y_index_map,
                )
            pltpu.emit_pipeline(
                body,
                grid=(n_batch_blocks,),
                in_specs=[dz_spec, *x_specs, y_spec],
                out_specs=[*x_specs, y_spec],
                core_axis=0,
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
            )(
                dz_ref,
                *[x_ref] * n_lane_tiles,
                y_ref,
                *[dx_ref] * n_lane_tiles,
                dy_ref,
            )

        dx, dy = pl.pallas_call(
            bwd_kernel,
            out_shape=[
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(dy_kernel_shape, dtype),
            ],
            in_specs=[
                pl.BlockSpec(a.shape, memory_space=pltpu.HBM) for a in (dz, x, y)
            ],
            out_specs=[
                pl.BlockSpec(x.shape, memory_space=pltpu.HBM),
                pl.BlockSpec(dy_kernel_shape, memory_space=pltpu.HBM),
            ],
            grid=(num_kernel,),
            scratch_shapes=[
                pltpu.VMEM((y_dim, batch_block_size, num_channels_padded), jnp.float32)
                for _ in range(2 if staged else 0)
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,)
            ),
            name="tensor_product_mtpu_trailing_channel_bwd",
        )(dz, x, y)

        if dim_leading:
            return (
                dx.transpose(1, 0, 2)[:batch_size, :, :channels],
                self._finalize_dy_dim_leading(dy, y_dim, channels, batch_size),
            )
        dx = self._unpack_dx(
            dx.astype(dtype),
            batch_size,
            x_dim,
            channels,
            n_batch_packed_in_channel,
        )
        return dx, self.finalize_dy(
            dy, y_dim, channels, batch_size, n_batch_packed_in_channel
        )

    @staticmethod
    def _pack_rows(x: jax.Array, n_batch_packed_in_channel: int) -> jax.Array:
        """Fold `n_batch_packed_in_channel` batch rows into the lane axis."""
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
        """Reconstruct `dx (batch, x_dim, channel)` from the kernel output."""
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

    @staticmethod
    def _vmem_bytes_per_batch_element(
        x_dim_padded: int,
        channels_padded: int,
        y_elems: int,
        z_dim: int,
        scratch_rows: int,
        dtype_bytes: int,
    ) -> int:
        """VMEM bytes per batch element: double-buffered dz, x, dx, y and dy blocks
        plus `scratch_rows` fp32 channel rows held at block scope."""
        return (
            2
            * dtype_bytes
            * (
                z_dim * channels_padded
                + 2 * x_dim_padded * channels_padded
                + 2 * y_elems
            )
            + 4 * channels_padded * scratch_rows
        )

    @abstractmethod
    def _validate_y(self, x: jax.Array, y: jax.Array) -> None:
        """Assert `y` has the rank/shape this mode expects relative to `x`."""

    @abstractmethod
    def pack_y(self, y: jax.Array, n_batch_packed_in_channel: int) -> jax.Array:
        """Fold `k` batch rows of `y` to match :meth:`_pack_x`'s lane layout."""

    @abstractmethod
    def dy_kernel_shape(
        self, y_padded: jax.Array, kernel_batch: int
    ) -> tuple[int, ...]:
        """Return the full HBM shape of the `dy` output for this mode."""

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
        """Materialize a map from `yi` to `(batch_block_size, num_channels)`.

        We materialize these once so that all accesses avoid an expensive broadcast.

        `n_batch_packed_in_channel` describes the active lane packing (`== 1` when off).
        """

    @abstractmethod
    def staging_rows(
        self, n_yis: int, y_dim: int, y_cols: int, n_batch_packed_in_channel: int
    ) -> int:
        """fp32 channel rows the packed path's `preload_y` / `write_dy` hold per
        batch element, besides the `y` / `dy` scratch."""

    @abstractmethod
    def write_dy(
        self,
        dy_vmem: jax.Array,
        yi: int,
        dy_acc: jax.Array,
        n_batch_packed_in_channel: int,
    ) -> None:
        """Store the accumulated `(batch_block_size, num_channels)` `dy_acc` for component `yi`.

        `n_batch_packed_in_channel` describes the active lane packing (`== 1` when off).
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

    _DIM_LEADING = True

    def _orient_y_dim_leading(self, y: jax.Array) -> jax.Array:
        return pad_for_tpu(y)

    def _batch_rows_dim_leading(self, ref: jax.Array, rows: pl.Slice) -> jax.Array:
        return ref.at[rows]

    def _dy_kernel_shape_dim_leading(
        self, y: jax.Array, kernel_batch: int
    ) -> tuple[int, ...]:
        return (kernel_batch, y.shape[1])

    def _y_blockspec_dim_leading(
        self, y: jax.Array, batch_block_size: int, channels: int
    ) -> pl.BlockSpec:
        return pl.BlockSpec(
            block_shape=(batch_block_size, y.shape[1]), index_map=lambda i: (i, 0)
        )

    def _load_y_dim_leading(
        self,
        y_vmem: jax.Array,
        yis: frozenset[int],
        batch_block_size: int,
        channels: int,
    ) -> dict[int, jax.Array]:
        return {
            yi: jnp.broadcast_to(y_vmem[:, yi][:, None], (batch_block_size, channels))
            for yi in yis
        }

    def _store_dy_dim_leading(
        self, dy_vmem: jax.Array, yi: int, dy_acc: jax.Array
    ) -> None:
        dy_vmem[:, yi] = jnp.sum(dy_acc, axis=-1).astype(dy_vmem.dtype)

    def _finalize_dy_dim_leading(
        self, dy: jax.Array, y_dim: int, num_channels: int, batch: int
    ) -> jax.Array:
        return dy[:batch, :y_dim]

    def _validate_y(self, x: jax.Array, y: jax.Array) -> None:
        assert (
            len(y.shape) == 2
        ), f"OUTER mode requires 2D y (batch, y_dim), got {y.shape}"

    def pack_y(self, y: jax.Array, n_batch_packed_in_channel: int) -> jax.Array:
        """pack y:`(batch, y_dim)` -> `(packed_batch_size, y_dim * k)`."""
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
                    :,
                    yi
                    * n_batch_packed_in_channel : (yi + 1)
                    * n_batch_packed_in_channel,
                ]  # (batch_block_size, k)
                seg = jnp.broadcast_to(
                    seg[:, :, None],
                    (batch_block_size, n_batch_packed_in_channel, real_channels),
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

    def staging_rows(
        self, n_yis: int, y_dim: int, y_cols: int, n_batch_packed_in_channel: int
    ) -> int:
        # Per used yi: the k-wide slice, the (k, channel) broadcast padded to a full
        # tile, its reshape and cast. Per yi: the loaded dy row and its k segments.
        k = n_batch_packed_in_channel
        return n_yis * (3 + round_up(k, 8)) + y_dim * (1 + k)

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
        dy = dy.transpose(
            0, 2, 1
        )  # (packed_batch_size, n_batch_packed_in_channel, y_dim)
        return dy.reshape(
            packed_batch_size * n_batch_packed_in_channel, y_dim
        )  # (batch, y_dim)


class _BwdTrailingChannelMapKernel(_BwdTrailingChannelKernel):
    """MAP: y is (batch, y_dim, channel); dy is elementwise over the channel axis."""

    _DIM_LEADING = True

    def _orient_y_dim_leading(self, y: jax.Array) -> jax.Array:
        return pad_for_tpu(jnp.transpose(y, (1, 0, 2)))

    def _batch_rows_dim_leading(self, ref: jax.Array, rows: pl.Slice) -> jax.Array:
        return ref.at[:, rows]

    def _dy_kernel_shape_dim_leading(
        self, y: jax.Array, kernel_batch: int
    ) -> tuple[int, ...]:
        return (y.shape[0], kernel_batch, y.shape[2])

    def _y_blockspec_dim_leading(
        self, y: jax.Array, batch_block_size: int, channels: int
    ) -> pl.BlockSpec:
        return pl.BlockSpec(
            block_shape=(y.shape[0], batch_block_size, channels),
            index_map=lambda i: (0, i, 0),
        )

    def _load_y_dim_leading(
        self,
        y_vmem: jax.Array,
        yis: frozenset[int],
        batch_block_size: int,
        channels: int,
    ) -> dict[int, jax.Array]:
        return {yi: y_vmem[yi, :, :] for yi in yis}

    def _store_dy_dim_leading(
        self, dy_vmem: jax.Array, yi: int, dy_acc: jax.Array
    ) -> None:
        dy_vmem[yi, :, :] = dy_acc.astype(dy_vmem.dtype)

    def _finalize_dy_dim_leading(
        self, dy: jax.Array, y_dim: int, num_channels: int, batch: int
    ) -> jax.Array:
        return dy.transpose(1, 0, 2)[:batch, :, :num_channels]

    def _validate_y(self, x: jax.Array, y: jax.Array) -> None:
        assert (
            len(y.shape) == 3 and x.shape[-1] == y.shape[-1]
        ), f"MAP mode requires 3D y with matching channel, got x={x.shape} y={y.shape}"

    def pack_y(self, y: jax.Array, n_batch_packed_in_channel: int) -> jax.Array:
        """MAP y is `(batch, y_dim, channel)` -> packed identically to x: `(packed_batch_size, y_dim, 128)`."""
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
        y_t = jax.lax.transpose(y_vmem[...], (1, 0, 2))
        return {yi: y_t[yi, :, :] for yi in yis}

    def staging_rows(
        self, n_yis: int, y_dim: int, y_cols: int, n_batch_packed_in_channel: int
    ) -> int:
        # The transposed y block and the loaded dy rows.
        return y_cols + y_dim

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
        return self._unpack_dx(
            dy, batch, y_dim, num_channels, n_batch_packed_in_channel
        )


def _tensor_product_kernel_mosaic_tpu_bwd(
    x: jax.Array,
    y: jax.Array,
    dz: jax.Array,
    params: PallasMosaicTPUTensorProductParams,
) -> tuple[jax.Array, jax.Array]:
    """Backward pass dispatching on `params.layout`/`params.mode`."""
    if params.layout == options.Layout.TRAILING_CHANNELS:
        if params.mode == options.MixingMode.MAP:
            return _BwdTrailingChannelMapKernel()(x, y, dz, params)
        if params.mode == options.MixingMode.OUTER:
            return _BwdTrailingChannelOuterKernel()(x, y, dz, params)
    raise NotImplementedError(f"Mosaic TPU TP bwd unsupported for {params}")
