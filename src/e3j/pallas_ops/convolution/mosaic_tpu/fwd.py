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

"""Fused MP-conv forward kernel (Pallas / Mosaic-TPU).

Gather + tensor product + scalar mixing + receiver scatter in one kernel
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from e3j.pallas_ops.convolution.mosaic_tpu.common import (
    FlushBuffers,
    WalkScratch,
    make_group_flush,
    sorted_reduce_and_flush,
)
from e3j.pallas_ops.convolution.mosaic_tpu.params import (
    PallasMosaicTPUMessagePassingConvolutionParams,
)
from e3j.pallas_ops.utils.coef import group_coef_by_zi_then_xi
from e3j.pallas_ops.utils.mosaic_tpu import round_up, select_batch_block_size
from e3j.pallas_ops.utils.named_scope import named_scope


class FwdScratch(NamedTuple):
    x_stage_vmem: (
        pl.MemoryRef
    )  # (batch_block_size, x_dim, channels): gathered x[senders]
    accumulator_vmem: (
        pl.MemoryRef
    )  # (out_dim, chunk, channels): open-group sublane partials
    stage_vmem: pl.MemoryRef  # (1, out_dim, channels): flush staging
    current_receiver_smem: pl.MemoryRef  # (1,) i32: open receiver group
    mask_smem: pl.MemoryRef  # (1,) i32: remaining-transitions bitmask
    previous_position_smem: pl.MemoryRef  # (1,) i32: current segment start within chunk
    flush_pending_smem: pl.MemoryRef  # (1,) i32: async flush in flight?
    gather_semaphore: pltpu.SemaphoreType  # DMA semaphore for the sender gathers
    flush_semaphore: pltpu.SemaphoreType  # DMA semaphore for the async flush


class _FwdOperands(NamedTuple):
    """Padded operands fed to the fwd `pallas_call`, in argument order."""

    x: jax.Array  # (n_nodes, x_dim, channels)
    y: jax.Array  # (y_dim, n_edges)
    edge_scalars: jax.Array  # (num_blocks, n_edges, channels)
    senders: jax.Array  # (n_edges,)
    receivers: jax.Array  # (n_edges,)
    zeros_out: (
        jax.Array
    )  # (num_cores, n_nodes, out_dim, channels): zero-init for the output


class _FwdConstants(NamedTuple):
    """Static plan closed over by the fwd kernel (shapes, tiling, CG structure)."""

    num_cores: int  # megacore grid width
    n_blocks: int  # edge tiles walked by emit_pipeline
    n_nodes: int
    x_dim: int
    channels: int  # unpadded multiplicity (for the final slice)
    out_dim: int  # unpadded output dim (for the final slice)
    channels_padded: int
    y_dim_padded: int
    out_dim_padded: int
    num_blocks: int  # edge-scalar irrep blocks
    batch_block_size: int
    chunk: int
    dtype: Any
    cg_groups: tuple  # ((zi, ((xi, ((yi, v), ...)), ...)), ...)
    irrep_block_of_output: tuple  # zi -> edge-scalar block


def _scratch_shapes(c: _FwdConstants) -> FwdScratch:
    return FwdScratch(
        x_stage_vmem=pltpu.VMEM(
            (c.batch_block_size, c.x_dim, c.channels_padded), c.dtype
        ),
        accumulator_vmem=pltpu.VMEM(
            (c.out_dim_padded, c.chunk, c.channels_padded), c.dtype
        ),
        stage_vmem=pltpu.VMEM((1, c.out_dim_padded, c.channels_padded), c.dtype),
        current_receiver_smem=pltpu.SMEM((1,), jnp.int32),
        mask_smem=pltpu.SMEM((1,), jnp.int32),
        previous_position_smem=pltpu.SMEM((1,), jnp.int32),
        flush_pending_smem=pltpu.SMEM((1,), jnp.int32),
        gather_semaphore=pltpu.SemaphoreType.DMA,
        flush_semaphore=pltpu.SemaphoreType.DMA,
    )


def _call_in_specs(ops: _FwdOperands) -> list[pl.BlockSpec]:
    """All operands live in HBM; blocks are the full arrays."""
    return [pl.BlockSpec(a.shape, memory_space=pltpu.HBM) for a in ops]


class _MessagePassingFwdKernel:
    """Fused message-passing convolution forward on Mosaic TPU.

    Computes, per receiver node, `out[receiver] = sum_{edge: receivers[edge]=receiver}
    m[edge]` where each edge message is a tensor product mixed by the edge scalars::

        m[edge, zi] = edge_scalars[edge, block(zi)]
                      * sum_{(xi, yi, v)} v * y[edge, yi] * x[senders[edge], xi]

    `zi` is the output index, `x` the sender node features, `y` the per-edge
    spherical harmonics, and the Clebsch-Gordan triples `(xi, yi, v)` are grouped
    by output index `zi` then `xi` (see `group_coef_by_zi`) so each `x[xi]` tile is
    loaded once and reused across its `yi` paths. `block(zi)` (`irrep_block_of_output`) maps an
    output coordinate to its target-irrep block, selecting the edge-scalar channel
    used for the mixing. Products are elementwise over the trailing `channels`
    (multiplicity) channel axis. Edges must be receiver-sorted.

    Layout / shapes
    ---------------
    The TRAILING_CHANNELS layout puts the multiplicity last, so the natural unit of
    work is a `(channels,)` vector lying along the TPU lane axis.

    - `x`: `(n_nodes, x_dim, channels)`.
    - `y`: `(n_edges, y_dim)`, transposed in-kernel to `(y_dim, n_edges)`.
    - `edge_scalars`: `(n_edges, num_blocks, channels)`.
    - `senders` / `receivers`: `(n_edges,)`.
    - `out`: `(n_nodes, out_dim, channels)`.

    `channels` is padded to a `num_lanes` multiple, `y_dim` to a `num_sublanes` multiple, and
    the edge axis to a whole number of `batch_block_size * num_cores` tiles (padded edges
    repeat the last receiver so no extra group is created); the padding is sliced
    off the result.

    Algorithm
    ----------
    A single `pallas_call` runs over a megacore grid of `tpu_info.num_cores` with
    PARALLEL semantics, all operands and the output living in HBM. Inside each
    program, `emit_pipeline` walks a grid of `n_blocks` edge tiles of `batch_block_size`
    edges each (`core_axis=0` distributes tiles across cores; `batch_block_size` is
    chosen to fit the double-buffered VMEM working set). The output is accumulated
    into a per-core `(num_cores, n_nodes, out_dim, channels)` HBM buffer
    (zero-initialised via `input_output_aliases`) that is summed over cores at the
    end, so cores never race on a shared node.

    Per pipeline step (body), working in VMEM:
      1. Gather the tile's sender features: one async DMA per edge copies
         `x[senders[k]]` into the `(batch_block_size, x_dim, channels)` stage.
      2. Tensor product + scalar mixing: for each output index `zi`, zero an
         accumulator, add `v * y[yi] * x[xi]` over its CG contributions, then multiply
         by the edge scalar of `block(zi)` to get the message `m[zi]`.
      3. Chunked scatter to receivers: stack the messages into an `(out_dim, ...)`
         plane and hand it to `sorted_reduce_and_flush`, which — exploiting the
         receiver-sorted order — accumulates each receiver group's partials and
         async DMAs a closed group's row out to the node's slice of the per-core
         buffer.

    The last still-open receiver group is flushed after the pipeline; the per-core
    buffer is then summed over cores and the `channels` / node padding dropped.
    """

    def _prolog(
        self,
        x: jax.Array,
        y: jax.Array,
        edge_scalars: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        params: PallasMosaicTPUMessagePassingConvolutionParams,
    ) -> tuple[_FwdOperands, _FwdConstants]:
        """Host-side prep: pad operands, group CG coefficients, size the tiling."""
        tpu_info = pltpu.get_tpu_info()
        num_lanes = tpu_info.num_lanes
        num_sublanes = tpu_info.num_sublanes
        num_cores = tpu_info.num_cores
        no_pad = (0, 0)
        chunk = num_sublanes

        config = params.fwd_config
        assert (
            config.assume_receiver_sorted
        ), "fwd requires receiver-sorted edges; sort outside the kernel otherwise."
        n_nodes, x_dim, channels = x.shape
        n_edges, y_dim = y.shape
        y = jnp.transpose(y, (1, 0))  # (y_dim, n_edges)
        _, num_blocks, _ = edge_scalars.shape
        edge_scalars = jnp.transpose(
            edge_scalars, (1, 0, 2)
        )  # (num_blocks, n_edges, channels)
        out_dim = params.z_space.dim

        batch_block_size = config.batch_block_size
        if batch_block_size is None:
            batch_block_size = select_batch_block_size(
                batch_size=n_edges,
                vmem_bytes_per_batch_element=int(
                    (x_dim + 2 * num_blocks + 2)
                    * round_up(channels, num_lanes)
                    * 4
                    * 1.2
                ),
                batch_block_size_candidates=(128, 64, 32, 16, 8),
                allow_padding=True,
            )
        assert batch_block_size % chunk == 0

        # Pad the edge axis to a whole number of blocks;
        # padded edges repeat the last receiver so no extra group is created.
        n_edges_padded = round_up(n_edges, batch_block_size * num_cores)
        pad_e = n_edges_padded - n_edges
        if pad_e > 0:
            y = jnp.pad(y, (no_pad, (0, pad_e)))
            edge_scalars = jnp.pad(edge_scalars, (no_pad, (0, pad_e), no_pad))
            senders = jnp.pad(senders, (0, pad_e))
            last_r = receivers[-1]
            receivers = jnp.concatenate(
                [receivers, jnp.full((pad_e,), last_r, dtype=receivers.dtype)]
            )

        # Pad the y_dim to a multiple of num_sublanes.
        y_dim_padded = round_up(y_dim, num_sublanes)
        y = jnp.pad(y, ((0, y_dim_padded - y_dim), no_pad))

        # Pad the channels to a multiple of num_lanes.
        channels_padded = round_up(channels, num_lanes)
        x = jnp.pad(x, (no_pad, no_pad, (0, channels_padded - channels)))
        edge_scalars = jnp.pad(
            edge_scalars, (no_pad, no_pad, (0, channels_padded - channels))
        )

        out_dim_padded = round_up(out_dim, num_sublanes)

        # Group by output index zi, then by input xi so the kernel loads each
        # x[xi] tile once, reused across its yi paths.
        indices, values = np.array(params.indices), np.array(params.values)
        cg_groups = group_coef_by_zi_then_xi(indices, values)
        irrep_block_of_output = tuple(int(b) for b in params.irrep_block_of_output)

        n_blocks = n_edges_padded // batch_block_size
        dtype = x.dtype

        zeros_out = jnp.zeros(
            (num_cores, n_nodes, out_dim_padded, channels_padded), dtype=dtype
        )

        ops = _FwdOperands(x, y, edge_scalars, senders, receivers, zeros_out)
        constants = _FwdConstants(
            num_cores=num_cores,
            n_blocks=n_blocks,
            n_nodes=n_nodes,
            x_dim=x_dim,
            channels=channels,
            out_dim=out_dim,
            channels_padded=channels_padded,
            y_dim_padded=y_dim_padded,
            out_dim_padded=out_dim_padded,
            num_blocks=num_blocks,
            batch_block_size=batch_block_size,
            chunk=chunk,
            dtype=dtype,
            cg_groups=cg_groups,
            irrep_block_of_output=irrep_block_of_output,
        )
        return ops, constants

    def _make_kernel(self, c: _FwdConstants):
        """Build the `pallas_call` kernel closing over the static `c`."""

        def _pipeline_in_specs(c: _FwdConstants) -> list[pl.BlockSpec]:
            """emit_pipeline in_specs for (y, edge_scalars, senders, receivers)."""
            return [
                pl.BlockSpec(
                    block_shape=(c.y_dim_padded, c.batch_block_size),
                    index_map=lambda i: (0, i),
                ),
                pl.BlockSpec(
                    block_shape=(c.num_blocks, c.batch_block_size, c.channels_padded),
                    index_map=lambda i: (0, i, 0),
                ),
                pl.BlockSpec(
                    block_shape=(c.batch_block_size,),
                    index_map=lambda i: (i,),
                    memory_space=pltpu.SMEM,
                ),
                pl.BlockSpec(
                    block_shape=(c.batch_block_size,),
                    index_map=lambda i: (i,),
                    memory_space=pltpu.SMEM,
                ),
            ]

        def kernel(
            x_hbm,  # (n_nodes, x_dim, channels)
            y_hbm,  # (y_dim, n_edges)
            edge_scalars_hbm,  # (num_blocks, n_edges, channels)
            senders_hbm,  # (n_edges,)
            receivers_hbm,  # (n_edges,)
            _zeros_hbm,  # (num_cores, n_nodes, out_dim, channels)
            # output:
            out_hbm,  # (num_cores, n_nodes, out_dim, channels)
            # scratch, see FwdScratch for shapes:
            x_stage_vmem,
            accumulator_vmem,
            stage_vmem,
            current_receiver_smem,
            mask_smem,
            previous_position_smem,
            flush_pending_smem,
            gather_semaphore,
            flush_semaphore,
        ):
            core_id = pl.program_id(0)

            # zero the accumulator so the first group's += starts clean (flushes re-zero after)
            accumulator_vmem[:, :, :] = jnp.zeros_like(accumulator_vmem[:, :, :])

            current_receiver_smem[0] = jnp.int32(-1)
            flush_pending_smem[0] = jnp.int32(0)

            # per-receiver-group reduce + async DMA out (double-buffered).
            flush_one, drain_flush = make_group_flush(
                FlushBuffers(
                    dst_hbm=out_hbm,
                    stage_vmem=stage_vmem,
                    accumulator_vmem=accumulator_vmem,
                    flush_pending=flush_pending_smem,
                    semaphore=flush_semaphore,
                ),
                core_id,
            )

            def body(y_vmem, edge_scalars_vmem, senders_smem, receivers_smem):

                # ---- 0. Load the needed sender node features ----
                with named_scope("fwd_copying_senders"):
                    copies = [
                        pltpu.make_async_copy(
                            x_hbm.at[pl.ds(senders_smem[k], 1), :, :],
                            x_stage_vmem.at[pl.ds(k, 1), :, :],
                            gather_semaphore,
                        )
                        for k in range(c.batch_block_size)
                    ]
                    for copy in copies:
                        copy.start()
                    for copy in copies:
                        copy.wait()

                # ---- 1. Tensor product + scalar mixing ----
                message_by_zi = {}
                with named_scope("fwd_tp_compute"):
                    for zi, by_xi in c.cg_groups:
                        block = c.irrep_block_of_output[zi]
                        edge_scalars_tile = edge_scalars_vmem[block, :, :]
                        tp_acc = jnp.zeros_like(edge_scalars_tile)
                        for xi, paths in by_xi:
                            x_tile = x_stage_vmem[:, xi, :]
                            for yi, v in paths:
                                y_vec = y_vmem[yi, :]
                                tp_acc = tp_acc + v * y_vec[:, None] * x_tile
                        message_by_zi[zi] = tp_acc * edge_scalars_tile

                # ---- 2. Chunked scatter to receivers ----
                row_idx = jax.lax.broadcasted_iota(
                    jnp.int32, (c.chunk, c.channels_padded), 0
                )
                present_zi = list(message_by_zi)
                zero_msg = jnp.zeros_like(message_by_zi[present_zi[0]])
                message_plane = jnp.stack(
                    [message_by_zi.get(zi, zero_msg) for zi in range(c.out_dim)], axis=0
                )
                with named_scope("fwd_reduce_flush"):
                    sorted_reduce_and_flush(
                        read_key=lambda i: receivers_smem[i],
                        cell_position=row_idx,
                        present_planes=present_zi,
                        source=message_plane,
                        accumulator=accumulator_vmem,
                        flush_one=flush_one,
                        open_key=current_receiver_smem,
                        scratch=WalkScratch(
                            transition_mask=mask_smem,
                            open_segment_start=previous_position_smem,
                        ),
                    )

            pltpu.emit_pipeline(
                body,
                grid=(c.n_blocks,),
                in_specs=_pipeline_in_specs(c),
                out_specs=[],
                core_axis=0,
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
            )(y_hbm, edge_scalars_hbm, senders_hbm, receivers_hbm)

            # accumulator_vmem holds the last open group, so we flush it
            final_receiver = current_receiver_smem[0]

            @pl.when(final_receiver >= jnp.int32(0))
            def _final_flush():
                flush_one(final_receiver)

            drain_flush()

        return kernel

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        edge_scalars: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        params: PallasMosaicTPUMessagePassingConvolutionParams,
    ) -> jax.Array:
        ops, c = self._prolog(x, y, edge_scalars, senders, receivers, params)

        out_packed = pl.pallas_call(
            self._make_kernel(c),
            out_shape=jax.ShapeDtypeStruct(
                (c.num_cores, c.n_nodes, c.out_dim_padded, c.channels_padded), c.dtype
            ),
            scratch_shapes=_scratch_shapes(c),
            in_specs=_call_in_specs(ops),
            out_specs=pl.BlockSpec(
                (c.num_cores, c.n_nodes, c.out_dim_padded, c.channels_padded),
                memory_space=pltpu.HBM,
            ),
            grid=(c.num_cores,),
            input_output_aliases={5: 0},  # use the zero-init for the output
            name="fwd_convolution",
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
                disable_bounds_checks=True,
            ),
        )(*ops)

        out_summed = out_packed.sum(0)
        return out_summed[: c.n_nodes, : c.out_dim, : c.channels]


def _message_passing_kernel_mosaic_tpu_fwd(
    x: jax.Array,
    y: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    params: PallasMosaicTPUMessagePassingConvolutionParams,
) -> jax.Array:
    return _MessagePassingFwdKernel()(x, y, edge_scalars, senders, receivers, params)
