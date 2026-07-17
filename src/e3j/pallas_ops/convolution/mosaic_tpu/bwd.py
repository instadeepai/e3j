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

"""Fused MP-conv backward kernel (Pallas / Mosaic-TPU).

gather x[sender] and dz[receiver]
+ compute the three gradients (dx, dy, d_edge_scalars)
+ sender scatter dx.
"""

from collections import defaultdict
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
from e3j.pallas_ops.utils.coef import group_coef_by_xi, group_coef_by_zi
from e3j.pallas_ops.utils.mosaic_tpu import (
    pad_for_tpu,
    round_up,
    select_batch_block_size,
)
from e3j.pallas_ops.utils.named_scope import named_scope


class BwdScratch(NamedTuple):
    x_stage: (
        pl.MemoryRef
    )  # (batch_block_size, x_dim*128): folded x[senders], one edge per row
    dz_stage: (
        pl.MemoryRef
    )  # (batch_block_size, out_dim*128): folded dz[receivers], one edge per row
    dz_plane: pl.MemoryRef  # (out_dim, batch_block_size, 128): dz[receiver], per edge
    x_plane: (
        pl.MemoryRef
    )  # (x_dim, batch_block_size, 128): sender-feature plane x[senders]
    dx_plane: pl.MemoryRef  # (x_dim, batch_block_size, 128): per-edge dx contribution
    accumulator: pl.MemoryRef  # (x_dim, chunk_rows, 128): open-group dx partials
    flush_stage: pl.MemoryRef  # (1, x_dim, 128): dx flush staging
    current_sender_smem: pl.MemoryRef  # (1,) i32: open sender group
    mask_smem: pl.MemoryRef  # (1,) i32: remaining-transitions bitmask
    previous_position_smem: pl.MemoryRef  # (1,) i32: current segment start within chunk
    count_smem: pl.MemoryRef  # (1,) i32: per-core block counter
    flush_pending_smem: pl.MemoryRef  # (1,) i32: async dx flush in flight?
    gather_semaphore: pltpu.SemaphoreType  # DMA semaphore: x/dz gathers
    stage_semaphore: pltpu.SemaphoreType  # DMA semaphore: senders/receivers row stages
    flush_semaphore: pltpu.SemaphoreType  # DMA semaphore: async dx flush


def _bwd_planes(
    out_dim,
    x_dim,
    y_dim,
    num_blocks,
    x_row_pad,
    dz_row_pad,
    lanes,
):
    return (
        x_row_pad // lanes  # x_stage
        + dz_row_pad // lanes  # dz_stage
        + out_dim  # dz_plane
        + 2 * x_dim  # x_plane, dx_plane
        + 2
        * (
            y_dim + 2 * num_blocks + 1
        )  # pipeline x2: y, edge_scalars, d_edge_scalars, dy
    )


class _BwdOperands(NamedTuple):
    """Padded operands fed to the bwd `pallas_call`, in argument order."""

    x: jax.Array  # (n_nodes, x_dim*channels_padded): folded node feats
    dz: jax.Array  # (n_nodes, out_dim*channels_padded): folded output cotangent
    y: jax.Array  # (n_edges, num_lanes)
    edge_scalars: jax.Array  # (num_blocks, n_edges, channels)
    senders: jax.Array  # (n_blocks, batch_block_size)
    receivers: jax.Array  # (n_blocks, batch_block_size)
    zeros_dx: jax.Array  # (num_cores, n_nodes, x_dim_padded, num_lanes): dx zero-init


class _BwdConstants(NamedTuple):
    """Static plan closed over by the bwd kernel (shapes, tiling, CG structure)."""

    num_cores: int  # megacore grid width
    n_blocks: int  # edge tiles walked by emit_pipeline
    blocks_per_core: int
    n_nodes: int
    x_dim: int
    x_dim_padded: int
    out_dim: int
    channels: int  # unpadded multiplicity (for the final slice)
    channels_padded: int  # lane-padded multiplicity
    y_dim: int  # unpadded y dim (for the final slice)
    num_blocks: int  # edge-scalar irrep blocks
    n_edges: int  # unpadded edge count (for the final slice)
    n_edges_padded: int
    batch_block_size: int
    chunk_rows: int
    num_lanes: int
    dtype: Any
    cg_by_block: tuple  # ((block, ((zi, (ZiContribution, ...)), ...)), ...)
    xi_used: tuple  # input indices with CG contributions
    yi_used: tuple  # spherical-harmonic indices with CG contributions
    irrep_block_of_output: tuple  # zi -> edge-scalar block


def _scratch_shapes(c: _BwdConstants) -> BwdScratch:
    return BwdScratch(
        x_stage=pltpu.VMEM((c.batch_block_size, c.x_dim * c.num_lanes), c.dtype),
        dz_stage=pltpu.VMEM((c.batch_block_size, c.out_dim * c.num_lanes), c.dtype),
        dz_plane=pltpu.VMEM((c.out_dim, c.batch_block_size, c.num_lanes), c.dtype),
        x_plane=pltpu.VMEM((c.x_dim, c.batch_block_size, c.num_lanes), c.dtype),
        dx_plane=pltpu.VMEM((c.x_dim, c.batch_block_size, c.num_lanes), c.dtype),
        accumulator=pltpu.VMEM((c.x_dim_padded, c.chunk_rows, c.num_lanes), c.dtype),
        flush_stage=pltpu.VMEM((1, c.x_dim_padded, c.num_lanes), c.dtype),
        current_sender_smem=pltpu.SMEM((1,), jnp.int32),
        mask_smem=pltpu.SMEM((1,), jnp.int32),
        previous_position_smem=pltpu.SMEM((1,), jnp.int32),
        count_smem=pltpu.SMEM((1,), jnp.int32),
        flush_pending_smem=pltpu.SMEM((1,), jnp.int32),
        gather_semaphore=pltpu.SemaphoreType.DMA,
        stage_semaphore=pltpu.SemaphoreType.DMA,
        flush_semaphore=pltpu.SemaphoreType.DMA,
    )


def _call_in_specs(ops: _BwdOperands) -> list[pl.BlockSpec]:
    """All operands live in HBM; blocks are the full arrays."""
    return [pl.BlockSpec(a.shape, memory_space=pltpu.HBM) for a in ops]


class _MessagePassingBwdKernel:
    """Fused message-passing convolution backward on Mosaic TPU.

    Backpropagates the output cotangent `dz` through the forward pass (see
    `_message_passing_kernel_mosaic_tpu_fwd`) to its three inputs. Regrouping the
    Clebsch-Gordan triples by output index `zi` (see `group_coef_by_zi`), one sweep
    over the triples produces all three gradients per edge (with
    `sender = senders[edge]`, `receiver = receivers[edge]`, and
    `es_dz = edge_scalars[edge, block(zi)] * dz[receiver, zi]`):

        dx[sender, xi]              += sum_{(zi, yi, v)} v * y[edge, yi] * es_dz
        dy[edge, yi]                += sum_{(zi, xi, v)} v * x[sender, xi] * es_dz  (over channels)
        d_edge_scalars[edge, block] += sum_{zi in block} m[edge, zi] * dz[receiver, zi]

    where the unmixed message is:

        m[edge, zi] = sum_{(xi, yi, v)} v * y[edge, yi] * x[sender, xi]

    Products are elementwise over the trailing `channels` (multiplicity) channel axis;
    `dy` additionally reduces over it. Edges must be sender-sorted (the `dx` scatter
    groups by sender).

    Layout / shapes
    ---------------
    - `x`: `(n_nodes, x_dim, channels)`.
    - `y`: `(n_edges, y_dim)`, lane-padded to `(n_edges, num_lanes)`.
    - `edge_scalars`: `(n_edges, num_blocks, channels)`.
    - `dz`: `(n_nodes, out_dim, channels)`.
    - `senders` / `receivers`: `(n_edges,)`.
    - outputs `dx`: `(n_nodes, x_dim, channels)`, `dy`: `(n_edges, y_dim)`,
      `d_edge_scalars`: `(n_edges, num_blocks, channels)`.

    Algorithm
    ----------
    A single `pallas_call` runs over a megacore grid of `tpu_info.num_cores` with
    PARALLEL semantics, operands and outputs in HBM. `emit_pipeline` walks `n_blocks`
    edge tiles of `batch_block_size` edges each (`core_axis=0` distributes tiles across
    cores; `batch_block_size` fits the double-buffered VMEM working set, see
    `_bwd_planes`). `dx` is accumulated into a per-core
    `(num_cores, n_nodes, x_dim, num_lanes)` HBM buffer (zero-initialised via
    `input_output_aliases`) summed over cores at the end, so cores never race on a
    shared sender; `dy` and `d_edge_scalars` are written per edge.

    Per pipeline step (body), working in VMEM:
      1. Gather per edge: async DMA `x[senders[k]]` and `dz[receivers[k]]` into
         per-edge stages, one edge per row (also prefetches the next tile).
      2. Slice the gathered tiles into per-component planes: `x[xi]` and `dz[zi]`
         as `(batch_block_size, num_lanes)` operands.
      3. CG sweep: in one pass over the triples grouped by output index `zi`, accumulate
         `dx[xi]`, `dy[yi]`, and per-block `d_edge_scalars`, then reduce `dy` over
         the `channels` axis.
      4. Chunked scatter to senders: hand `dx`'s per-`xi` plane to
         `sorted_reduce_and_flush`, which — exploiting the sender-sorted order —
         accumulates each sender group and async DMAs a closed group out to the
         sender's slice of the per-core buffer.

    The last still-open sender group is flushed after the pipeline; the per-core
    `dx` buffer is then summed over cores and all padding dropped.
    """

    def _prolog(
        self,
        x: jax.Array,
        y: jax.Array,
        edge_scalars: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        dz: jax.Array,
        params: PallasMosaicTPUMessagePassingConvolutionParams,
    ) -> tuple[_BwdOperands, _BwdConstants]:
        """Host-side prep: pad operands, group CG coefficients, size the tiling."""
        tpu_info = pltpu.get_tpu_info()
        num_lanes = tpu_info.num_lanes
        num_sublanes = tpu_info.num_sublanes
        num_cores = tpu_info.num_cores
        no_pad = (0, 0)

        n_nodes, x_dim, channels = x.shape
        chunk_rows = num_sublanes
        n_edges, y_dim = y.shape
        assert (
            y_dim <= num_lanes
        ), f"bwd requires y_dim <= {num_lanes} (single lane tile); got {y_dim}."
        _, num_blocks, _ = edge_scalars.shape
        edge_scalars = jnp.transpose(
            edge_scalars, (1, 0, 2)
        )  # (num_blocks, n_edges, channels)
        out_dim = params.z_space.dim

        channels_padded = round_up(channels, num_lanes)
        x = jnp.pad(x, (no_pad, no_pad, (0, channels_padded - channels)))
        dz = jnp.pad(dz, (no_pad, no_pad, (0, channels_padded - channels)))
        x = x.reshape(n_nodes, x_dim * channels_padded)
        dz = dz.reshape(n_nodes, out_dim * channels_padded)
        edge_scalars = jnp.pad(
            edge_scalars, (no_pad, no_pad, (0, channels_padded - channels))
        )
        bwd_config = params.bwd_config
        assert (
            bwd_config.assume_sender_sorted
        ), "bwd requires sender-sorted edges; sort outside the kernel otherwise."
        batch_block_size = bwd_config.batch_block_size
        if batch_block_size is None:
            batch_block_size = select_batch_block_size(
                batch_size=n_edges,
                vmem_bytes_per_batch_element=int(
                    _bwd_planes(
                        out_dim,
                        x_dim,
                        y_dim,
                        num_blocks,
                        round_up(x_dim * channels_padded, num_lanes),
                        round_up(out_dim * channels_padded, num_lanes),
                        num_lanes,
                    )
                    * num_lanes
                    * 4
                    * 1.8
                ),
                batch_block_size_candidates=(64, 32, 16, 8),
                allow_padding=True,
            )

        # Pad the edge axis to a whole number of per-core blocks; padded edges
        # repeat the last sender/receiver so no extra group is created.
        n_edges_padded = round_up(n_edges, batch_block_size * num_cores)
        pad_edges = n_edges_padded - n_edges
        if pad_edges > 0:
            y = jnp.pad(y, ((0, pad_edges), no_pad))
            edge_scalars = jnp.pad(edge_scalars, (no_pad, (0, pad_edges), no_pad))
            last_sender = senders[-1]
            last_receiver = receivers[-1]
            senders = jnp.concatenate(
                [senders, jnp.full((pad_edges,), last_sender, senders.dtype)]
            )
            receivers = jnp.concatenate(
                [receivers, jnp.full((pad_edges,), last_receiver, receivers.dtype)]
            )

        x_dim_padded = round_up(x_dim, num_sublanes)
        dtype = x.dtype
        # lane-pad y
        y = pad_for_tpu(y)

        irrep_block_of_output = tuple(int(b) for b in params.irrep_block_of_output)
        indices, values = np.array(params.indices), np.array(params.values)
        cg_by_zi = group_coef_by_zi(
            indices, values
        )  # zi -> [ZiContribution(xi, yi, value)]
        xi_used = tuple(group_coef_by_xi(indices, values))
        yi_used = tuple(sorted({c.yi for paths in cg_by_zi.values() for c in paths}))

        # Group the per-zi CG paths by their irreps output block, used by d_edge_scalars
        _cg_by_block: dict[int, list] = defaultdict(list)
        for zi, paths in cg_by_zi.items():
            _cg_by_block[irrep_block_of_output[zi]].append((zi, paths))
        cg_by_block = tuple(
            (block, tuple(group)) for block, group in sorted(_cg_by_block.items())
        )

        n_blocks = n_edges_padded // batch_block_size
        blocks_per_core = n_blocks // num_cores
        senders = senders.reshape(n_blocks, batch_block_size)
        receivers = receivers.reshape(n_blocks, batch_block_size)
        zeros_dx = jnp.zeros((num_cores, n_nodes, x_dim_padded, num_lanes), dtype)

        ops = _BwdOperands(x, dz, y, edge_scalars, senders, receivers, zeros_dx)
        constants = _BwdConstants(
            num_cores=num_cores,
            n_blocks=n_blocks,
            blocks_per_core=blocks_per_core,
            n_nodes=n_nodes,
            x_dim=x_dim,
            x_dim_padded=x_dim_padded,
            out_dim=out_dim,
            channels=channels,
            channels_padded=channels_padded,
            y_dim=y_dim,
            num_blocks=num_blocks,
            n_edges=n_edges,
            n_edges_padded=n_edges_padded,
            batch_block_size=batch_block_size,
            chunk_rows=chunk_rows,
            num_lanes=num_lanes,
            dtype=dtype,
            cg_by_block=cg_by_block,
            xi_used=xi_used,
            yi_used=yi_used,
            irrep_block_of_output=irrep_block_of_output,
        )
        return ops, constants

    def _make_kernel(self, c: _BwdConstants):
        """Build the `pallas_call` kernel closing over the static plan `c`.

        `body` stays nested: it closes over the scratch and HBM refs that Pallas
        only binds as `kernel`'s parameters.
        """

        def _pipeline_in_specs(c: _BwdConstants) -> list[pl.BlockSpec]:
            """emit_pipeline in_specs for (y, edge_scalars, senders, receivers,
            senders_next, receivers_next)."""
            return [
                pl.BlockSpec((c.batch_block_size, c.num_lanes), lambda i: (i, 0)),
                pl.BlockSpec(
                    (c.num_blocks, c.batch_block_size, c.num_lanes),
                    lambda i: (0, i, 0),
                ),
                pl.BlockSpec(
                    (1, c.batch_block_size), lambda i: (i, 0), memory_space=pltpu.SMEM
                ),
                pl.BlockSpec(
                    (1, c.batch_block_size), lambda i: (i, 0), memory_space=pltpu.SMEM
                ),
                pl.BlockSpec(
                    (1, c.batch_block_size),
                    lambda i: (jnp.minimum(i + 1, c.n_blocks - 1), 0),
                    memory_space=pltpu.SMEM,
                ),  # prefetch next senders
                pl.BlockSpec(
                    (1, c.batch_block_size),
                    lambda i: (jnp.minimum(i + 1, c.n_blocks - 1), 0),
                    memory_space=pltpu.SMEM,
                ),  # prefetch next receivers
            ]

        def kernel(
            x_hbm,
            dz_hbm,
            y_hbm,
            edge_scalars_hbm,
            senders_hbm,
            receivers_hbm,
            _z,
            # outputs:
            d_edge_scalars_hbm,
            dy_hbm,
            dx_hbm,
            # scratch, see BwdScratch for shapes:
            x_stage,
            dz_stage,
            dz_plane,
            x_plane,
            dx_plane,
            accumulator,
            flush_stage,
            current_sender_smem,
            mask_smem,
            previous_position_smem,
            count_smem,
            flush_pending_smem,
            gather_semaphore,
            stage_semaphore,
            flush_semaphore,
        ):
            core_id = pl.program_id(0)

            # zero the accumulator so the first group's += starts clean (flushes re-zero after)
            accumulator[:, :, :] = jnp.zeros_like(accumulator[:, :, :])

            current_sender_smem[0] = jnp.int32(-1)
            count_smem[0] = jnp.int32(0)
            flush_pending_smem[0] = jnp.int32(0)

            # per-sender-group dx reduce + async DMA out (double-buffered).
            flush_one, drain_flush = make_group_flush(
                FlushBuffers(
                    dst_hbm=dx_hbm,
                    stage_vmem=flush_stage,
                    accumulator_vmem=accumulator,
                    flush_pending=flush_pending_smem,
                    semaphore=flush_semaphore,
                ),
                core_id,
            )

            def body(
                y_vmem,
                edge_scalars_vmem,
                senders_smem,
                receivers_smem,
                senders_next_smem,
                receivers_next_smem,
                d_edge_scalars_vmem,
                dy_vmem,
            ):
                block_idx = count_smem[0]
                senders_read = lambda j: senders_smem[0, j]  # noqa: E731
                receivers_read = lambda k: receivers_smem[0, k]  # noqa: E731
                senders_next_read = lambda j: senders_next_smem[0, j]  # noqa: E731
                receivers_next_read = lambda k: receivers_next_smem[0, k]  # noqa: E731

                # ---- 1. gather x[sender] and dz[receiver], one edge per row ----
                def gather_copies(read_sender, read_receiver):
                    x_copies, dz_copies = [], []
                    for k in range(c.batch_block_size):
                        x_copies.append(
                            pltpu.make_async_copy(
                                x_hbm.at[pl.ds(read_sender(k), 1), :],
                                x_stage.at[pl.ds(k, 1), :],
                                gather_semaphore,
                            )
                        )
                        dz_copies.append(
                            pltpu.make_async_copy(
                                dz_hbm.at[pl.ds(read_receiver(k), 1), :],
                                dz_stage.at[pl.ds(k, 1), :],
                                stage_semaphore,
                            )
                        )
                    return [*x_copies, *dz_copies]

                @pl.when(block_idx == jnp.int32(0))
                def _cold_start():
                    for copy in gather_copies(senders_read, receivers_read):
                        copy.start()

                for copy in gather_copies(senders_read, receivers_read):
                    copy.wait()

                zero = jnp.zeros((c.batch_block_size, c.num_lanes), c.dtype)

                # ---- 2. lane-slice the folded rows into per-component planes ----
                lanes = c.num_lanes
                with named_scope("extract"):
                    for xi in range(c.x_dim):
                        x_plane[xi, :, :] = x_stage[:, lanes * xi : lanes * (xi + 1)]
                    for zi in range(c.out_dim):
                        dz_plane[zi, :, :] = dz_stage[:, lanes * zi : lanes * (zi + 1)]

                count_smem[0] = block_idx + 1

                # prefetch the next block's gathers now
                @pl.when(block_idx < jnp.int32(c.blocks_per_core - 1))
                def _prefetch():
                    for copy in gather_copies(senders_next_read, receivers_next_read):
                        copy.start()

                y_broadcast = {
                    yi: jnp.broadcast_to(
                        y_vmem[:, yi][:, None], (c.batch_block_size, c.num_lanes)
                    )
                    for yi in c.yi_used
                }
                y_of = lambda yi: y_broadcast[yi]  # noqa: E731

                # ---- 3. CG sweep: all three gradients in one pass over the CG paths ----
                with named_scope("cg_sweep"):
                    dx_acc = {xi: zero for xi in c.xi_used}
                    dy_acc = {yi: zero for yi in c.yi_used}
                    for block, zi_group in c.cg_by_block:
                        d_edge_scalars_acc = zero
                        for zi, paths in zi_group:
                            with named_scope("computing es_dz"):
                                dz_zi = dz_plane[zi, :, :]
                                es_dz = (
                                    edge_scalars_vmem[c.irrep_block_of_output[zi], :, :]
                                    * dz_zi
                                )
                                message_zi = zero
                            for xi, yi, v in paths:
                                y_val = y_of(yi)
                                x_val = x_plane[xi, :, :]
                                message_zi = message_zi + v * y_val * x_val
                                dx_acc[xi] = dx_acc[xi] + v * y_val * es_dz
                                dy_acc[yi] = dy_acc[yi] + v * x_val * es_dz
                            d_edge_scalars_acc = d_edge_scalars_acc + message_zi * dz_zi
                        d_edge_scalars_vmem[block, :, :] = d_edge_scalars_acc
                    for xi in c.xi_used:
                        dx_plane[xi, :, :] = dx_acc[xi]

                    # dy: reduce the channel axis.
                    dy_vmem[:, :] = zero
                    for yi in c.yi_used:
                        dy_vmem[:, yi] = jnp.sum(dy_acc[yi], axis=-1)
                dx_indices = list(c.xi_used)

                # ---- 4. walk: accumulate dx_plane per sender group, scatter to dx.
                chunk_edge_position = jax.lax.broadcasted_iota(
                    jnp.int32, (c.chunk_rows, c.num_lanes), 0
                )

                with named_scope("bwd_reduce_flush"):
                    sorted_reduce_and_flush(
                        read_key=senders_read,
                        cell_position=chunk_edge_position,
                        present_planes=dx_indices,
                        source=dx_plane,
                        accumulator=accumulator,
                        flush_one=flush_one,
                        open_key=current_sender_smem,
                        scratch=WalkScratch(
                            transition_mask=mask_smem,
                            open_segment_start=previous_position_smem,
                        ),
                    )

            pipe = pltpu.emit_pipeline(
                body,
                grid=(c.n_blocks,),
                in_specs=_pipeline_in_specs(c),
                out_specs=[
                    pl.BlockSpec(
                        (c.num_blocks, c.batch_block_size, c.num_lanes),
                        lambda i: (0, i, 0),
                    ),
                    pl.BlockSpec((c.batch_block_size, c.num_lanes), lambda i: (i, 0)),
                ],
                core_axis=0,
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
            )
            pipe(
                y_hbm,
                edge_scalars_hbm,
                senders_hbm,
                receivers_hbm,
                senders_hbm,
                receivers_hbm,
                d_edge_scalars_hbm,
                dy_hbm,
            )

            # Flush the last still-open group
            final_sender = current_sender_smem[0]

            @pl.when(final_sender >= jnp.int32(0))
            def _flush_final_group():
                flush_one(final_sender)

            drain_flush()

        return kernel

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        edge_scalars: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        dz: jax.Array,
        params: PallasMosaicTPUMessagePassingConvolutionParams,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        ops, c = self._prolog(x, y, edge_scalars, senders, receivers, dz, params)

        d_edge_scalars_packed, dy_packed, dx_cores = pl.pallas_call(
            self._make_kernel(c),
            out_shape=[
                jax.ShapeDtypeStruct(
                    (c.num_blocks, c.n_edges_padded, c.num_lanes), c.dtype
                ),
                jax.ShapeDtypeStruct((c.n_edges_padded, c.num_lanes), c.dtype),
                jax.ShapeDtypeStruct(
                    (c.num_cores, c.n_nodes, c.x_dim_padded, c.num_lanes), c.dtype
                ),
            ],
            scratch_shapes=_scratch_shapes(c),
            in_specs=_call_in_specs(ops),
            out_specs=[
                pl.BlockSpec(
                    (c.num_blocks, c.n_edges_padded, c.num_lanes),
                    memory_space=pltpu.HBM,
                ),
                pl.BlockSpec((c.n_edges_padded, c.num_lanes), memory_space=pltpu.HBM),
                pl.BlockSpec(
                    (c.num_cores, c.n_nodes, c.x_dim_padded, c.num_lanes),
                    memory_space=pltpu.HBM,
                ),
            ],
            grid=(c.num_cores,),
            input_output_aliases={6: 2},  # use the zero-init for dx
            name="bwd_convolution",
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
                disable_bounds_checks=True,
            ),
        )(*ops)

        dx_sum = dx_cores.sum(0)
        dx = dx_sum[:, : c.x_dim, : c.channels]  # (n_nodes, x_dim, channels)

        dy = dy_packed.reshape(c.n_edges_padded, c.channels_padded)[
            : c.n_edges, : c.y_dim
        ]
        d_edge_scalars = jnp.transpose(
            d_edge_scalars_packed.reshape(
                c.num_blocks, c.n_edges_padded, c.channels_padded
            ),
            (1, 0, 2),
        )[: c.n_edges, :, : c.channels]
        return dx, dy, d_edge_scalars


def _message_passing_kernel_mosaic_tpu_bwd(
    x: jax.Array,
    y: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    dz: jax.Array,
    params: PallasMosaicTPUMessagePassingConvolutionParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _MessagePassingBwdKernel()(
        x, y, edge_scalars, senders, receivers, dz, params
    )
