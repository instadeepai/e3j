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

from collections import defaultdict
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from e3j.pallas_ops.convolution.mosaic_tpu import edges, padding
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
from e3j.pallas_ops.utils.mosaic_tpu import (
    BlockPlan,
    plan_blocks,
    round_up,
    vmem_step_bytes,
)
from e3j.pallas_ops.utils.named_scope import named_scope


class FwdScratch(NamedTuple):
    x_stage_vmem: (
        pl.MemoryRef
    )  # (2, x_dim, edge_block_size, channels): gathered x[senders], double-buffered
    accumulator_vmem: (
        pl.MemoryRef
    )  # (out_dim, chunk, channels): open-group sublane partials
    stage_vmem: pl.MemoryRef  # (1, out_dim, channels): flush staging
    current_receiver_smem: pl.MemoryRef  # (1,) i32: open receiver group
    mask_smem: pl.MemoryRef  # (1,) i32: remaining-transitions bitmask
    previous_position_smem: pl.MemoryRef  # (1,) i32: current segment start within chunk
    flush_pending_smem: pl.MemoryRef  # (1,) i32: async flush in flight?
    flush_count_smem: pl.MemoryRef  # (1,) i32: flushes done (first one is a partial)
    step_smem: pl.MemoryRef  # (1,) i32: pipeline steps run, selects the x_stage slot
    gather_semaphore: pltpu.SemaphoreType  # (2,) DMA semaphores, one per x_stage slot
    flush_semaphore: pltpu.SemaphoreType  # DMA semaphore for the async flush


class _FwdOperands(NamedTuple):
    """Padded operands fed to the fwd `pallas_call`, in argument order."""

    x: jax.Array  # (n_nodes, x_dim, channels)
    y: jax.Array  # (y_dim, n_edge_blocks, edge_block_size)
    edge_scalars: jax.Array  # (num_irrep_blocks, n_edges, channels), or edge-major
    senders: jax.Array  # (n_edge_blocks, edge_block_size)
    receivers: jax.Array  # (n_edge_blocks, edge_block_size)
    zeros_out: jax.Array  # (n_nodes, out_dim, channels): zero-init, shared by cores
    zeros_boundary: (
        jax.Array
    )  # (num_cores, 2, out_dim, channels): zero-init for the straddling groups
    block_order: jax.Array  # (n_edge_blocks,) i32: walked blocks first
    n_walked_blocks: jax.Array  # (1,) i32: how many of them hold a real edge


class _FwdConstants(NamedTuple):
    """Static plan closed over by the fwd kernel (shapes, tiling, CG structure)."""

    num_cores: int  # megacore grid width
    n_edge_blocks: int  # edge tiles: n_edges_padded // edge_block_size
    n_nodes: int
    x_dim: int
    channels: int  # unpadded multiplicity (for the final slice)
    out_dim: int  # unpadded output dim (for the final slice)
    channels_padded: int
    plan: BlockPlan
    y_dim_padded: int
    out_dim_padded: int
    num_irrep_blocks: int  # edge scalars per edge, one per output irrep block
    chunk: int
    dtype: Any
    tp_plan: tuple  # see _plan_scalings
    irrep_block_of_output: tuple  # zi -> edge-scalar block
    dynamic_edges: bool  # walk only the blocks holding a real edge
    edge_major: bool  # edge scalars layout, see edges.edge_scalars_edge_major
    n_edges: int  # edge scalars rows; the last block reads past them


_MIN_EDGE_BLOCK_SIZE = 64  # measured: this kernel prefers fewer passes at equal area
_EDGE_BLOCK_SIZES = (128, 64)  # edge tiles this kernel can run


def _vmem_shapes(
    out_dim: int,
    out_dim_padded: int,
    x_dim: int,
    y_dim_padded: int,
    chunk: int,
    num_irrep_blocks: int,
    edge_block_size: int,
    channel_block: int,
) -> tuple[dict[str, tuple[int, ...]], ...]:
    """Shapes of the VMEM scratch, the pipelined operands and the live values."""
    scratch = {
        "x_stage_vmem": (2, x_dim, edge_block_size, channel_block),
        "accumulator_vmem": (out_dim_padded, chunk, channel_block),
        "stage_vmem": (1, out_dim_padded, channel_block),
    }
    operands = {
        "y": (y_dim_padded, 1, edge_block_size),
        "edge_scalars": (num_irrep_blocks, edge_block_size, channel_block),
    }
    values = {"message_plane": (out_dim, edge_block_size, channel_block)}
    return scratch, operands, values


def _plan_scalings(cg_groups: tuple) -> tuple:
    """Per zi, `((z_scale, ((xi, yi, y_scale, x_scale, sign), ...)), ...)`.

    Each path's `|v|` goes on its y tile, its x tile or its zi partial sum, chosen by
    greedy set cover so that one scaled tile or sum serves as many paths as it can.
    Ties go to y: it is scaled before its broadcast, one vmul instead of a tile of them.
    """
    paths = [
        (zi, xi, yi, abs(float(v)), 1 if v > 0 else -1)
        for zi, by_xi in cg_groups
        for xi, yi_paths in by_xi
        for yi, v in yi_paths
    ]
    covers = defaultdict(set)
    for i, (zi, xi, yi, q, _) in enumerate(paths):
        if q != 1.0:
            for key in (("y", yi, q), ("x", xi, q), ("z", zi, q)):
                covers[key].add(i)
    uncovered = set().union(*covers.values())
    side = {}
    while uncovered:
        key = max(
            sorted(covers), key=lambda k: (len(covers[k] & uncovered), k[0] == "y")
        )
        for i in covers[key] & uncovered:
            side[i] = key[0]
        uncovered -= covers[key]

    plan = defaultdict(lambda: defaultdict(list))
    for i, (zi, xi, yi, q, sign) in enumerate(paths):
        on = side.get(i)
        scale = {s: q if s == on else 1.0 for s in "yxz"}
        plan[zi][scale["z"]].append((xi, yi, scale["y"], scale["x"], sign))
    return tuple(
        (
            zi,
            tuple(
                (z_scale, tuple(sorted(terms, key=lambda t: -t[-1])))
                for z_scale, terms in by_z_scale.items()
            ),
        )
        for zi, by_z_scale in plan.items()
    )


def _scratch_shapes(c: _FwdConstants) -> FwdScratch:
    scratch, _, _ = _vmem_shapes(
        c.out_dim,
        c.out_dim_padded,
        c.x_dim,
        c.y_dim_padded,
        c.chunk,
        c.num_irrep_blocks,
        c.plan.edge_block_size,
        c.plan.channel_block,
    )
    return FwdScratch(
        **{name: pltpu.VMEM(shape, c.dtype) for name, shape in scratch.items()},
        current_receiver_smem=pltpu.SMEM((1,), jnp.int32),
        mask_smem=pltpu.SMEM((1,), jnp.int32),
        previous_position_smem=pltpu.SMEM((1,), jnp.int32),
        flush_pending_smem=pltpu.SMEM((1,), jnp.int32),
        flush_count_smem=pltpu.SMEM((1,), jnp.int32),
        step_smem=pltpu.SMEM((1,), jnp.int32),
        gather_semaphore=pltpu.SemaphoreType.DMA((2,)),
        flush_semaphore=pltpu.SemaphoreType.DMA,
    )


def _call_in_specs(ops: _FwdOperands) -> list[pl.BlockSpec]:
    """Array operands live in HBM as whole blocks; the edge bound lives in SMEM."""
    return [
        *(pl.BlockSpec(a.shape, memory_space=pltpu.HBM) for a in ops[:-2]),
        *(pl.BlockSpec(memory_space=pltpu.SMEM) for _ in ops[-2:]),
    ]


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
    - `edge_scalars`: `(n_edges, num_irrep_blocks, channels)`.
    - `senders` / `receivers`: `(n_edges,)`.
    - `out`: `(n_nodes, out_dim, channels)`.

    `channels` is padded up to `num_lanes` unless it already divides it and `y_dim` to a
    `num_sublanes` multiple. `y` and the indices are padded to whole `edge_block_size`
    tiles (padded edges repeat the last receiver so no extra group is created); the
    edge scalars are not (see `edges`). The padding is sliced off the result.

    Algorithm
    ----------
    A single `pallas_call` runs over a megacore grid of `tpu_info.num_cores` with
    PARALLEL semantics, all operands and the output living in HBM. Inside each
    program, `emit_pipeline` walks a grid of `n_edge_blocks` edge tiles of `edge_block_size`
    edges each (`core_axis=0` distributes tiles across cores; `edge_block_size` is
    chosen to fit the double-buffered VMEM working set). The output is accumulated
    into a per-core `(num_cores, n_nodes, out_dim, channels)` HBM buffer
    (zero-initialised via `input_output_aliases`) that is summed over cores at the
    end, so cores never race on a shared node.

    Per pipeline step (body), working in VMEM:
      1. Gather the tile's sender features: one async DMA per edge copies
         `x[senders[k]]` into one slot of the double-buffered `(x_dim,
         edge_block_size, channels)` stage, issued a step ahead.
      2. Tensor product + scalar mixing: for each output index `zi`, sum
         `v * y[yi] * x[xi]` over its CG contributions, then multiply
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
        edge_is_real: jax.Array | None = None,
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
        _, num_irrep_blocks, _ = edge_scalars.shape
        edge_major = edges.edge_scalars_edge_major(num_irrep_blocks, channels)
        if not edge_major:
            edge_scalars = jnp.transpose(
                edge_scalars, (1, 0, 2)
            )  # (num_irrep_blocks, n_edges, channels)
        out_dim = params.z_space.dim

        channels_padded = (
            channels if num_lanes % channels == 0 else round_up(channels, num_lanes)
        )
        y_dim_padded = round_up(y_dim, num_sublanes)
        out_dim_padded = round_up(out_dim, num_sublanes)

        def vmem_bytes(edge_block_size: int, channel_block: int) -> int:
            scratch, operands, values = _vmem_shapes(
                out_dim,
                out_dim_padded,
                x_dim,
                y_dim_padded,
                chunk,
                num_irrep_blocks,
                edge_block_size,
                channel_block,
            )
            return vmem_step_bytes(
                scratch.values(),
                operands.values(),
                values.values(),
                channel_tiles=max(1, channel_block // num_lanes),
                num_lanes=num_lanes,
                num_sublanes=num_sublanes,
                itemsize=x.dtype.itemsize,
            )

        plan = plan_blocks(
            channels_padded=channels_padded,
            num_lanes=num_lanes,
            vmem_bytes=vmem_bytes,
            edge_block_sizes=_EDGE_BLOCK_SIZES,
            min_edge_block_size=_MIN_EDGE_BLOCK_SIZE,
            min_strided_edge_block_size=_MIN_EDGE_BLOCK_SIZE,
            vmem_capacity_bytes=tpu_info.vmem_capacity_bytes,
            edge_block_size=config.edge_block_size,
        )
        edge_block_size = plan.edge_block_size
        assert edge_block_size % chunk == 0

        edge_scalars = edges.pad_to_one_block(
            edge_scalars, 0 if edge_major else 1, edge_block_size
        )
        # Pad y and the indices to whole blocks; the edge scalars stay ragged.
        # Padded edges repeat the last receiver so no extra group is created.
        n_edges_padded = round_up(n_edges, edge_block_size)
        pad_e = n_edges_padded - n_edges
        if pad_e > 0:
            y = jnp.pad(y, (no_pad, (0, pad_e)))
            senders = jnp.pad(senders, (0, pad_e))
            last_r = receivers[-1]
            receivers = jnp.concatenate(
                [receivers, jnp.full((pad_e,), last_r, dtype=receivers.dtype)]
            )

        y = jnp.pad(y, ((0, y_dim_padded - y_dim), no_pad))

        if channels_padded != channels:
            x = jnp.pad(x, (no_pad, no_pad, (0, channels_padded - channels)))
            edge_scalars = jnp.pad(
                edge_scalars, (no_pad, no_pad, (0, channels_padded - channels))
            )

        # Group by output index zi, then by input xi so the kernel loads each
        # x[xi] tile once, reused across its yi paths.
        indices, values = np.array(params.indices), np.array(params.values)
        tp_plan = _plan_scalings(group_coef_by_zi_then_xi(indices, values))
        irrep_block_of_output = tuple(int(b) for b in params.irrep_block_of_output)

        n_edge_blocks = n_edges_padded // edge_block_size
        y = y.reshape(y_dim_padded, n_edge_blocks, edge_block_size)
        senders = senders.reshape(n_edge_blocks, edge_block_size)
        receivers = receivers.reshape(n_edge_blocks, edge_block_size)
        dtype = x.dtype

        zeros_out = jnp.zeros((n_nodes, out_dim_padded, channels_padded), dtype=dtype)
        zeros_boundary = jnp.zeros(
            (num_cores, 2, out_dim_padded, channels_padded), dtype=dtype
        )

        dynamic_edges = edge_is_real is not None
        if dynamic_edges:
            block_order, n_walked_blocks = padding.block_order(
                edge_is_real, n_edge_blocks, edge_block_size
            )
        else:
            block_order = np.zeros((1,), np.int32)  # unread: index maps stay static
            n_walked_blocks = np.int32(n_edge_blocks)
        n_walked_blocks = jnp.reshape(n_walked_blocks.astype(jnp.int32), (1,))

        ops = _FwdOperands(
            x,
            y,
            edge_scalars,
            senders,
            receivers,
            zeros_out,
            zeros_boundary,
            block_order,
            n_walked_blocks,
        )
        constants = _FwdConstants(
            num_cores=num_cores,
            n_edge_blocks=n_edge_blocks,
            n_nodes=n_nodes,
            x_dim=x_dim,
            channels=channels,
            out_dim=out_dim,
            channels_padded=channels_padded,
            plan=plan,
            y_dim_padded=y_dim_padded,
            out_dim_padded=out_dim_padded,
            num_irrep_blocks=num_irrep_blocks,
            chunk=chunk,
            dtype=dtype,
            tp_plan=tp_plan,
            irrep_block_of_output=irrep_block_of_output,
            dynamic_edges=dynamic_edges,
            edge_major=edge_major,
            n_edges=max(n_edges, edge_block_size),
        )
        return ops, constants

    def _make_kernel(self, c: _FwdConstants):
        """Build the `pallas_call` kernel closing over the static `c`."""

        def _pipeline_in_specs(
            c: _FwdConstants, block_order_smem, n_walked_blocks_smem, stride: int
        ) -> list[pl.BlockSpec]:
            """emit_pipeline in_specs for (y, edge_scalars, senders, receivers, next senders)."""
            walked_block = (
                (lambda i: block_order_smem[i]) if c.dynamic_edges else (lambda i: i)
            )
            last_block = (
                n_walked_blocks_smem[0] - 1 if c.dynamic_edges else c.n_edge_blocks - 1
            )
            return [
                pl.BlockSpec(
                    block_shape=(c.y_dim_padded, 1, c.plan.edge_block_size),
                    index_map=lambda i: (0, walked_block(i), 0),
                ),
                edges.edge_scalars_spec(
                    c.edge_major,
                    c.plan.edge_block_size,
                    c.num_irrep_blocks,
                    c.plan.channel_block,
                    walked_block,
                    stride,
                ),
                pl.BlockSpec(
                    block_shape=(1, c.plan.edge_block_size),
                    index_map=lambda i: (walked_block(i), 0),
                    memory_space=pltpu.SMEM,
                ),
                pl.BlockSpec(
                    block_shape=(1, c.plan.edge_block_size),
                    index_map=lambda i: (walked_block(i), 0),
                    memory_space=pltpu.SMEM,
                ),
                pl.BlockSpec(
                    block_shape=(1, c.plan.edge_block_size),
                    index_map=lambda i: (
                        walked_block(jnp.minimum(i + 1, last_block)),
                        0,
                    ),
                    memory_space=pltpu.SMEM,
                ),
            ]

        def kernel(
            x_hbm,  # (n_nodes, x_dim, channels)
            y_hbm,  # (y_dim, n_edge_blocks, edge_block_size)
            edge_scalars_hbm,  # (num_irrep_blocks, n_edges, channels)
            senders_hbm,  # (n_edges,)
            receivers_hbm,  # (n_edges,)
            _zeros_hbm,  # (n_nodes, out_dim, channels)
            _zeros_boundary_hbm,  # (num_cores, 2, out_dim, channels)
            block_order_smem,  # (n_edge_blocks,) i32
            n_walked_blocks_smem,  # (1,) i32
            # outputs:
            out_hbm,  # (n_nodes, out_dim, channels)
            boundary_hbm,  # (num_cores, 2, out_dim, channels)
            # scratch, see FwdScratch for shapes:
            x_stage_vmem,
            accumulator_vmem,
            stage_vmem,
            current_receiver_smem,
            mask_smem,
            previous_position_smem,
            flush_pending_smem,
            flush_count_smem,
            step_smem,
            gather_semaphore,
            flush_semaphore,
        ):
            core_id = pl.program_id(0)

            def run_stride(stride: int):
                channel_slice = c.plan.channel_slice(stride)

                # zero the accumulator so the first group's += starts clean
                accumulator_vmem[:, :, :] = jnp.zeros_like(accumulator_vmem[:, :, :])

                current_receiver_smem[0] = jnp.int32(-1)
                step_smem[0] = jnp.int32(0)
                flush_pending_smem[0] = jnp.int32(0)
                flush_count_smem[0] = jnp.int32(0)

                # per-receiver-group reduce + async DMA out (double-buffered).
                flush_one, drain_flush = make_group_flush(
                    FlushBuffers(
                        dst_hbm=out_hbm,
                        stage_vmem=stage_vmem,
                        accumulator_vmem=accumulator_vmem,
                        flush_pending=flush_pending_smem,
                        semaphore=flush_semaphore,
                        boundary_hbm=boundary_hbm,
                        flush_count=flush_count_smem,
                        channel_slice=channel_slice,
                    ),
                    core_id,
                )

                def gathers(sender, slot):
                    return [
                        pltpu.make_async_copy(
                            x_hbm.at[sender(k), :, channel_slice],
                            x_stage_vmem.at[slot, :, k, :],
                            gather_semaphore.at[slot],
                        )
                        for k in range(c.plan.edge_block_size)
                    ]

                def body(
                    y_vmem,
                    edge_scalars_vmem,
                    senders_smem,
                    receivers_smem,
                    next_senders_smem,
                ):
                    step = step_smem[0]
                    slot = step % 2
                    edge_block = edges.walked_edge_block(
                        core_id,
                        step,
                        n_walked_blocks_smem[0] if c.dynamic_edges else c.n_edge_blocks,
                        c.num_cores,
                        block_order_smem if c.dynamic_edges else None,
                    )

                    def edge_scalar_tile(block):
                        tile = edges.load_edge_scalars(
                            c.edge_major, edge_scalars_vmem, block
                        )
                        return edges.mask_ragged_rows(tile, edge_block, c.n_edges)

                    # ---- 0. Load the needed sender node features ----
                    # this block's were prefetched by the previous step
                    with named_scope("fwd_copying_senders"):

                        @pl.when(step == 0)
                        def _gather_first_block():
                            for copy in gathers(lambda k: senders_smem[0, k], 0):
                                copy.start()

                        for copy in gathers(lambda k: senders_smem[0, k], slot):
                            copy.wait()
                        for copy in gathers(
                            lambda k: next_senders_smem[0, k], 1 - slot
                        ):
                            copy.start()
                        step_smem[0] = step + 1

                    # ---- 1. Tensor product + scalar mixing ----
                    x_plane = x_stage_vmem[slot]
                    message_by_zi = {}
                    with named_scope("fwd_tp_compute"):
                        tile_shape = (c.plan.edge_block_size, c.plan.channel_block)
                        y_tiles, x_tiles = {}, {}

                        def y_tile(yi, scale):
                            if (yi, scale) not in y_tiles:
                                y_vec = y_vmem[yi, 0, :]
                                y_vec = y_vec if scale == 1.0 else scale * y_vec
                                y_tiles[yi, scale] = jnp.broadcast_to(
                                    y_vec[:, None], tile_shape
                                )
                            return y_tiles[yi, scale]

                        def x_tile(xi, scale):
                            if (xi, scale) not in x_tiles:
                                x_b = x_plane[xi, :, :]
                                x_tiles[xi, scale] = (
                                    x_b if scale == 1.0 else scale * x_b
                                )
                            return x_tiles[xi, scale]

                        for zi, by_z_scale in c.tp_plan:
                            tp_acc = None
                            for z_scale, terms in by_z_scale:
                                partial = None
                                for xi, yi, y_scale, x_scale, sign in terms:
                                    term = y_tile(yi, y_scale) * x_tile(xi, x_scale)
                                    if partial is None:
                                        partial = term if sign > 0 else -term
                                    elif sign > 0:
                                        partial = partial + term
                                    else:
                                        partial = partial - term
                                if z_scale != 1.0:
                                    partial = z_scale * partial
                                tp_acc = partial if tp_acc is None else tp_acc + partial
                            block = c.irrep_block_of_output[zi]
                            message_by_zi[zi] = tp_acc * edge_scalar_tile(block)

                    # ---- 2. Chunked scatter to receivers ----
                    row_idx = jax.lax.broadcasted_iota(
                        jnp.int32, (c.chunk, c.plan.channel_block), 0
                    )
                    present_zi = list(message_by_zi)
                    zero_msg = jnp.zeros_like(message_by_zi[present_zi[0]])
                    message_plane = jnp.stack(
                        [message_by_zi.get(zi, zero_msg) for zi in range(c.out_dim)],
                        axis=0,
                    )
                    with named_scope("fwd_reduce_flush"):
                        sorted_reduce_and_flush(
                            read_key=lambda i: receivers_smem[0, i],
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

                n_edge_blocks = (
                    n_walked_blocks_smem[0] if c.dynamic_edges else c.n_edge_blocks
                )

                pltpu.emit_pipeline(
                    body,
                    grid=(n_edge_blocks,),
                    in_specs=_pipeline_in_specs(
                        c, block_order_smem, n_walked_blocks_smem, stride
                    ),
                    out_specs=[],
                    core_axis=0,
                    dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
                )(y_hbm, edge_scalars_hbm, senders_hbm, receivers_hbm, senders_hbm)

                # the last step prefetched a block it never reads
                @pl.when(step_smem[0] > 0)
                def _drain_gathers():
                    for copy in gathers(lambda k: 0, step_smem[0] % 2):
                        copy.wait()

                # accumulator_vmem holds the last open group, so we flush it
                final_receiver = current_receiver_smem[0]

                @pl.when(final_receiver >= jnp.int32(0))
                def _final_flush():
                    flush_one(final_receiver, final=True)

                drain_flush()

            for stride in range(c.plan.n_channel_strides):
                run_stride(stride)

        return kernel

    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        edge_scalars: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        params: PallasMosaicTPUMessagePassingConvolutionParams,
        edge_is_real: jax.Array | None = None,
    ) -> jax.Array:
        ops, c = self._prolog(
            x, y, edge_scalars, senders, receivers, params, edge_is_real
        )

        out_packed = pl.pallas_call(
            self._make_kernel(c),
            out_shape=[
                jax.ShapeDtypeStruct(
                    (c.n_nodes, c.out_dim_padded, c.channels_padded), c.dtype
                ),
                jax.ShapeDtypeStruct(
                    (c.num_cores, 2, c.out_dim_padded, c.channels_padded), c.dtype
                ),
            ],
            scratch_shapes=_scratch_shapes(c),
            in_specs=_call_in_specs(ops),
            out_specs=[
                pl.BlockSpec(
                    (c.n_nodes, c.out_dim_padded, c.channels_padded),
                    memory_space=pltpu.HBM,
                ),
                pl.BlockSpec(
                    (c.num_cores, 2, c.out_dim_padded, c.channels_padded),
                    memory_space=pltpu.HBM,
                ),
            ],
            grid=(c.num_cores,),
            input_output_aliases={5: 0, 6: 1},  # use the zero-inits for the outputs
            name="fwd_convolution",
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
                disable_bounds_checks=True,
            ),
        )(*ops)
        out, boundary = out_packed

        # Each core's first-closed and final groups may be shared with a
        # neighbouring core, so they were parked in `boundary` instead of being
        # written in place. Their node ids are the keys at the core's first and
        # last edge -- known here, no need to report them from the kernel.
        block_order = (
            ops.block_order
            if c.dynamic_edges
            else jnp.arange(c.n_edge_blocks, dtype=jnp.int32)
        )
        first_edge, last_edge = padding.core_edge_bounds(
            block_order, ops.n_walked_blocks, c.num_cores, c.plan.edge_block_size
        )
        for core in range(c.num_cores):
            for slot, edge in enumerate((first_edge[core], last_edge[core])):
                block, slot_in_block = divmod(edge, c.plan.edge_block_size)
                row = ops.receivers[block, slot_in_block].astype(jnp.int32)
                start = (row, jnp.int32(0), jnp.int32(0))
                current = jax.lax.dynamic_slice(
                    out, start, (1, c.out_dim_padded, c.channels_padded)
                )
                out = jax.lax.dynamic_update_slice(
                    out, current + boundary[core, slot][None], start
                )

        return out[: c.n_nodes, : c.out_dim, : c.channels]


def _message_passing_kernel_mosaic_tpu_fwd(
    x: jax.Array,
    y: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    params: PallasMosaicTPUMessagePassingConvolutionParams,
    edge_is_real: jax.Array | None = None,
) -> jax.Array:
    return _MessagePassingFwdKernel()(
        x, y, edge_scalars, senders, receivers, params, edge_is_real
    )
