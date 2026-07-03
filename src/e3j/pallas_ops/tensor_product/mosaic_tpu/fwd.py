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

"""Forward pass of the tensor product using Pallas Mosaic TPU."""

import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from e3j.pallas_ops.tensor_product.mosaic_tpu.params import (
    PallasMosaicTPUTensorProductParams,
    PallasMosaicTPUTensorProductParamsFwdConfig,
)
from e3j.pallas_ops.utils.coef import group_coef_by_zi
from e3j.pallas_ops.utils.mosaic_tpu import pad_for_tpu, select_batch_block_size
from e3j.pallas_ops.utils.named_scope import named_scope
from e3j.utils import options


class FwdKernel(ABC):
    """Forward tensor-product kernel."""

    @abstractmethod
    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        params: PallasMosaicTPUTensorProductParams,
    ) -> jax.Array:
        """Implement your kernel here"""


class _FwdTrailingChannelKernel(FwdKernel):
    """TRAILING_CHANNELS forward tensor product on Mosaic TPU.

    Computes `z[zi] = sum_{(xi, yi, v)} v * y[yi] * x[xi]` per output index
    `zi`, where the Clebsch-Gordan triples `(xi, yi, v)` are grouped by `zi`
    (see `group_coef_by_zi`). The product is elementwise over the trailing
    `mul` (multiplicity) channel axis; `y` may either broadcast over `mul`
    (OUTER) or be elementwise on it (MAP) — that difference is the only thing the
    two concrete subclasses change, via the abstract `y_*` / `preload_y` hooks.

    Layout / shapes
    ---------------
    The TRAILING_CHANNELS layout puts the multiplicity last, so the natural unit
    of work is a `(mul,)` vector lying along the TPU lane axis.

    - `x`: `(batch, x_dim, mul)`.
    - `y`: subclass-defined — OUTER `(batch, y_dim)`, MAP `(batch, y_dim, mul)`.
    - `z`: `(batch, z_dim, mul)`.

    `x` and `y` are padded with `pad_for_tpu` so `x_dim` / `mul` are tile
    multiples; the extra `mul` columns are sliced off the final result.

    Note: This lead to unused compute for mul != 128. Future work will
    implement multiplicity packing.

    Algorithm
    ----------
    A single `pallas_call` runs over a megacore grid of `tpu_info.num_cores` with
    PARALLEL semantics, with both operands and the output living in HBM. Inside
    each program, `emit_pipeline` walks a grid of `num_blocks` batch tiles of
    `batch_block_size` rows each (`core_axis=0` distributes tiles across cores;
    `batch_block_size` is chosen to fit the double-buffered VMEM working set, see
    `_vmem_bytes_per_batch_element`). emit_pipeline double-buffers the x / y / z
    tiles so the next tile's DMA overlaps the current tile's compute.

    Per pipeline step (body), working in VMEM:
      1. Transpose the x tile `(batch_block, x_dim, mul) -> (x_dim, batch_block, mul)`
         so an `x[xi]` slice is a contiguous `(batch_block, mul)` operand (the
         CG loop indexes `xi` repeatedly; making it the leading axis avoids a
         strided gather on every contribution).
      2. Preload each used `yi` into a `(batch_block, mul)` operand once
         (OUTER broadcasts the per-channel scalar across `mul` while MAP slices it).
         so a `yi` reused across several `zi` is only materialized one time.
      3. For each `zi`, zero an fp32 `(batch_block, mul)` accumulator, add
         `v * y[yi] * x[xi]` over its contributions, and store into the z tile.

    The z output buffer is itself `(z_dim, batch, mul)` (z_dim leading, matching
    the transposed-x orientation).
    We assume `transpose(1, 0, 2)` back to `(batch, z_dim, mul)`
    is a free bitcast (no data movement) before the `mul` padding is dropped.
    We take advantage of this layout of z in the backward pass see _BwdTrailingChannelKernel.
    """

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        params: PallasMosaicTPUTensorProductParams,
    ) -> jax.Array:
        config = params.fwd_config or PallasMosaicTPUTensorProductParamsFwdConfig()
        x_origin_shape = x.shape
        assert len(x_origin_shape) == 3, f"x must be (batch, x_dim, mul), got {x.shape}"
        self._validate_y(x, y)
        mul = x.shape[-1]

        tpu_info = pltpu.get_tpu_info()
        num_kernel = tpu_info.num_cores

        x = pad_for_tpu(x)
        y = pad_for_tpu(y)
        x_dim_padded = x.shape[1]
        mul_padded = x.shape[-1]
        z_dim = params.z_space.dim

        batch_size = x_origin_shape[0]

        cg_by_zi = group_coef_by_zi(params.indices, params.values)
        cg_groups = tuple(
            (zi, tuple((xi, yi, v) for xi, yi, v in cg_by_zi.get(zi, ())))
            for zi in range(z_dim)
        )
        yis = frozenset(yi for _, contribs in cg_groups for _, yi, _ in contribs)

        batch_block_size = select_batch_block_size(
            batch_size=batch_size,
            vmem_bytes_per_batch_element=self._vmem_bytes_per_batch_element(
                x_dim_padded, mul_padded, y, z_dim, len(yis), x.dtype.itemsize
            ),
            vmem_capacity_bytes=tpu_info.vmem_capacity_bytes,
            num_kernel=num_kernel,
            override=config.batch_block_size,
        )
        num_blocks = batch_size // batch_block_size

        def tp_kernel(x, y, z):
            def body(x_vmem_src, y_vmem_src, z_vmem_src):
                with named_scope("x_vmem transpose"):
                    # Transpose x to (x_dim, batch_block_size, mul) for better memory access in the kernel.
                    x_vmem = jax.lax.transpose(x_vmem_src[...], (1, 0, 2))
                with named_scope("preload_y"):
                    pre_y_vmem = self.preload_y(
                        y_vmem_src, yis, batch_block_size, mul_padded
                    )
                for zi, contributions in cg_groups:
                    with named_scope(f"zi:{zi}"):
                        with named_scope(f"create acc"):
                            acc = jnp.zeros(
                                (z_vmem_src.shape[1], z_vmem_src.shape[2]), jnp.float32
                            )

                        for xi, yi, v in contributions:
                            with named_scope(f"xi,yi:{xi},{yi}"):
                                with named_scope("load_x"):
                                    x_slice = x_vmem[xi, :, :]
                                with named_scope("acc"):
                                    tmp = v * pre_y_vmem[yi] * x_slice
                                    acc = acc + tmp

                        with named_scope("write to vmem io"):
                            z_vmem_src[zi, :, :] = acc.astype(z_vmem_src.dtype)

            pltpu.emit_pipeline(
                body,
                grid=(num_blocks,),
                in_specs=[
                    pl.BlockSpec(
                        block_shape=(batch_block_size, x_dim_padded, mul_padded),
                        index_map=lambda i: (i, 0, 0),
                    ),
                    pl.BlockSpec(
                        block_shape=self.y_block_shape(
                            y.shape, batch_block_size, mul_padded
                        ),
                        index_map=self.y_index_map,
                    ),
                ],
                out_specs=pl.BlockSpec(
                    block_shape=(z_dim, batch_block_size, mul_padded),
                    index_map=lambda i: (0, i, 0),
                ),
                core_axis=0,
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
            )(x, y, z)

        tmp = pl.pallas_call(
            tp_kernel,
            out_shape=jax.ShapeDtypeStruct((z_dim, batch_size, mul_padded), x.dtype),
            in_specs=[pl.BlockSpec(a.shape, memory_space=pltpu.HBM) for a in (x, y)],
            out_specs=pl.BlockSpec(
                (z_dim, batch_size, mul_padded),
                memory_space=pltpu.HBM,
            ),
            grid=(num_kernel,),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,)
            ),
            name="tensor_product_mtpu_trailing_channel",
        )(x, y)

        # transpose(1,0,2) is a free bitcast: {2,1,0} on (z_dim,batch,mul)
        # becomes {2,0,1} on (batch,z_dim,mul) — no data movement.
        return tmp.transpose(1, 0, 2)[:, :, :mul]

    def _vmem_bytes_per_batch_element(
        self,
        x_dim_padded: int,
        mul_padded: int,
        y_padded: jax.Array,
        z_dim: int,
        num_yis: int,
        dtype_bytes: int,
    ) -> int:
        """VMEM bytes consumed per batch element (pipeline double-buffered + body intermediates).

        Pipeline buffers are double-buffered by emit_pipeline.
        Body intermediates: transposed x, preloaded y slices, one accumulator.
        """
        y_elems = math.prod(self.y_block_shape(y_padded.shape, 1, mul_padded))
        pipeline = (
            2 * dtype_bytes * (x_dim_padded * mul_padded + y_elems + z_dim * mul_padded)
        )
        body = 4 * mul_padded * (x_dim_padded + num_yis + 1)
        return pipeline + body

    @abstractmethod
    def _validate_y(self, x: jax.Array, y: jax.Array) -> None:
        """Assert `y` has the rank/shape this mode expects relative to `x`."""

    @abstractmethod
    def y_block_shape(
        self, y_shape: tuple[int, ...], bbs: int, mul: int
    ) -> tuple[int, ...]:
        """Return the per-tile y `BlockSpec` shape for a batch block of `bbs`."""

    @abstractmethod
    def y_index_map(self, i: int) -> tuple[int, ...]:
        """Map batch-tile index `i` to the y block's grid offset."""

    @abstractmethod
    def preload_y(
        self, y_vmem: jax.Array, yis: frozenset[int], bbs: int, mul: int
    ) -> dict[int, jax.Array]:
        """Materialize a `yi -> (bbs, mul)` operand for each used y component."""


class _FwdTrailingChannelOuterKernel(_FwdTrailingChannelKernel):
    """OUTER (y_mul=1): y is (batch, y_dim); broadcasts over x's mul axis."""

    def _validate_y(self, x: jax.Array, y: jax.Array) -> None:
        assert (
            len(y.shape) == 2
        ), f"OUTER mode (y_mul=1) requires 2D y (batch, y_dim), got y={y.shape}"

    def y_block_shape(
        self, y_shape: tuple[int, ...], bbs: int, mul: int
    ) -> tuple[int, ...]:
        return (bbs, y_shape[1])

    def y_index_map(self, i: int) -> tuple[int, ...]:
        return (i, 0)

    def preload_y(
        self, y_vmem: jax.Array, yis: frozenset[int], bbs: int, mul: int
    ) -> dict[int, jax.Array]:
        return {yi: jnp.broadcast_to(y_vmem[:, yi][:, None], (bbs, mul)) for yi in yis}


class _FwdTrailingChannelMapKernel(_FwdTrailingChannelKernel):
    """MAP: y is (batch, y_dim, mul); elementwise over the mul axis."""

    def _validate_y(self, x: jax.Array, y: jax.Array) -> None:
        assert (
            len(y.shape) == 3 and x.shape[-1] == y.shape[-1]
        ), f"MAP mode requires 3D y with matching mul, got x={x.shape} y={y.shape}"

    def y_block_shape(
        self, y_shape: tuple[int, ...], bbs: int, mul: int
    ) -> tuple[int, ...]:
        return (bbs, y_shape[1], mul)

    def y_index_map(self, i: int) -> tuple[int, ...]:
        return (i, 0, 0)

    def preload_y(
        self, y_vmem: jax.Array, yis: frozenset[int], bbs: int, mul: int
    ) -> dict[int, jax.Array]:
        return {yi: y_vmem[:, yi, :] for yi in yis}


def _tensor_product_kernel_mosaic_tpu_fwd(
    x: jax.Array,
    y: jax.Array,
    params: PallasMosaicTPUTensorProductParams,
) -> jax.Array:
    if params.layout == options.Layout.TRAILING_CHANNELS:
        if params.mode == options.TPMode.MAP:
            return _FwdTrailingChannelMapKernel()(x, y, params)
        if params.mode == options.TPMode.OUTER:
            return _FwdTrailingChannelOuterKernel()(x, y, params)
    raise NotImplementedError(f"Mosaic TPU TP fwd: unsupported params {params}")
