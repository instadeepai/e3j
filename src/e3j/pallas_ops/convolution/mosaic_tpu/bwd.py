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

from e3j.pallas_ops.convolution.mosaic_tpu import padding
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
    BlockPlan,
    pad_for_tpu,
    plan_blocks,
    round_up,
    vmem_step_bytes,
)
from e3j.pallas_ops.utils.named_scope import named_scope


class BwdScratch(NamedTuple):
    x_stage: (
        pl.MemoryRef
    )  # (edge_block_size, x_dim*channels): folded x[senders], one edge per row
    dz_stage: (
        pl.MemoryRef
    )  # (edge_block_size, out_dim*channels): folded dz[receivers], one edge per row
    dz_plane: pl.MemoryRef  # (out_dim, edge_block_size, channels): dz[receiver]
    x_plane: (
        pl.MemoryRef
    )  # (x_dim, edge_block_size, channels): sender-feature plane x[senders]
    dx_plane: pl.MemoryRef  # (x_dim, edge_block_size, channels): per-edge dx
    accumulator: pl.MemoryRef  # (x_dim, chunk_rows, channels): open-group dx partials
    flush_stage: pl.MemoryRef  # (1, x_dim, channels): dx flush staging
    current_sender_smem: pl.MemoryRef  # (1,) i32: open sender group
    mask_smem: pl.MemoryRef  # (1,) i32: remaining-transitions bitmask
    previous_position_smem: pl.MemoryRef  # (1,) i32: current segment start within chunk
    count_smem: pl.MemoryRef  # (1,) i32: per-core block counter
    flush_pending_smem: pl.MemoryRef  # (1,) i32: async dx flush in flight?
    flush_count_smem: pl.MemoryRef  # (1,) i32: flushes done (first one is a partial)
    gather_semaphore: pltpu.SemaphoreType  # DMA semaphore: x/dz gathers
    stage_semaphore: pltpu.SemaphoreType  # DMA semaphore: senders/receivers row stages
    flush_semaphore: pltpu.SemaphoreType  # DMA semaphore: async dx flush


_MIN_EDGE_BLOCK_SIZE = 32  # a 32-edge tile is worth a single pass ...
_MIN_STRIDED_EDGE_BLOCK_SIZE = 64  # ... but a strided pass wants the wider tile
_EDGE_BLOCK_SIZES = (128, 64, 32, 16, 8)  # 16/8 only reachable via the fallback


class _BwdOperands(NamedTuple):
    """Padded operands fed to the bwd `pallas_call`, in argument order."""

    x: jax.Array  # (n_nodes, x_dim*channels_padded): folded node feats
    dz: jax.Array  # (n_nodes, out_dim*channels_padded): folded output cotangent
    y: jax.Array  # (n_edges, num_lanes)
    edge_scalars: jax.Array  # (num_irrep_blocks, n_edges, channels)
    senders: jax.Array  # (n_edge_blocks, edge_block_size)
    receivers: jax.Array  # (n_edge_blocks, edge_block_size)
    zeros_dx: (
        jax.Array
    )  # (n_nodes, x_dim_padded, channels): dx zero-init, shared by cores
    zeros_boundary: (
        jax.Array
    )  # (num_cores, 2, x_dim_padded, channels): zero-init for the straddling groups
    block_order: jax.Array  # (n_edge_blocks,) i32: walked blocks first
    n_walked_blocks: jax.Array  # (1,) i32: how many of them hold a real edge


class _BwdConstants(NamedTuple):
    """Static plan closed over by the bwd kernel (shapes, tiling, CG structure)."""

    num_cores: int  # megacore grid width
    n_edge_blocks: int  # edge tiles: n_edges_padded // edge_block_size
    blocks_per_core: int
    n_nodes: int
    x_dim: int
    x_dim_padded: int
    out_dim: int
    channels: int  # unpadded multiplicity (for the final slice)
    channels_padded: int  # lane-padded multiplicity
    plan: BlockPlan
    y_dim: int  # unpadded y dim (for the final slice)
    num_irrep_blocks: int  # edge scalars per edge, one per output irrep block
    n_edges: int  # unpadded edge count (for the final slice)
    n_edges_padded: int
    chunk_rows: int
    num_lanes: int
    dtype: Any
    cg_by_block: tuple  # ((block, ((zi, (ZiContribution, ...)), ...)), ...)
    xi_used: tuple  # input indices with CG contributions
    yi_used: tuple  # spherical-harmonic indices with CG contributions
    irrep_block_of_output: tuple  # zi -> edge-scalar block
    dynamic_edges: bool  # walk only the blocks holding a real edge


def _vmem_shapes(
    out_dim: int,
    x_dim: int,
    y_dim: int,
    x_dim_padded: int,
    chunk_rows: int,
    num_irrep_blocks: int,
    num_lanes: int,
    edge_block_size: int,
    channel_block: int,
) -> tuple[dict[str, tuple[int, ...]], ...]:
    """Shapes of the VMEM scratch, the pipelined operands and the live values."""
    scratch = {
        "x_stage": (edge_block_size, x_dim * channel_block),
        "dz_stage": (edge_block_size, out_dim * channel_block),
        "dz_plane": (out_dim, edge_block_size, channel_block),
        "x_plane": (x_dim, edge_block_size, channel_block),
        "dx_plane": (x_dim, edge_block_size, channel_block),
        "accumulator": (x_dim_padded, chunk_rows, channel_block),
        "flush_stage": (1, x_dim_padded, channel_block),
    }
    operands = {
        "y": (edge_block_size, num_lanes),
        "edge_scalars": (num_irrep_blocks, edge_block_size, channel_block),
        "d_edge_scalars": (num_irrep_blocks, edge_block_size, channel_block),
        "dy": (1, edge_block_size, num_lanes),
    }
    values = {
        "dx_acc": (x_dim, edge_block_size, channel_block),
        "dy_acc": (y_dim, edge_block_size, channel_block),
        "y_broadcast": (y_dim, edge_block_size, channel_block),
    }
    return scratch, operands, values


def _scratch_shapes(c: _BwdConstants) -> BwdScratch:
    scratch, _, _ = _vmem_shapes(
        c.out_dim,
        c.x_dim,
        c.y_dim,
        c.x_dim_padded,
        c.chunk_rows,
        c.num_irrep_blocks,
        c.plan.num_lanes,
        c.plan.edge_block_size,
        c.plan.channel_block,
    )
    return BwdScratch(
        **{name: pltpu.VMEM(shape, c.dtype) for name, shape in scratch.items()},
        current_sender_smem=pltpu.SMEM((1,), jnp.int32),
        mask_smem=pltpu.SMEM((1,), jnp.int32),
        previous_position_smem=pltpu.SMEM((1,), jnp.int32),
        count_smem=pltpu.SMEM((1,), jnp.int32),
        flush_pending_smem=pltpu.SMEM((1,), jnp.int32),
        flush_count_smem=pltpu.SMEM((1,), jnp.int32),
        gather_semaphore=pltpu.SemaphoreType.DMA,
        stage_semaphore=pltpu.SemaphoreType.DMA,
        flush_semaphore=pltpu.SemaphoreType.DMA,
    )


def _zero_unwalked_blocks(
    d_edge_scalars: jax.Array,
    dy: jax.Array,
    block_order: jax.Array,
    n_walked_blocks: jax.Array,
    c: _BwdConstants,
) -> tuple[jax.Array, jax.Array]:
    """Zero the per-edge cotangent blocks the bwd pipeline did not walk."""

    def kernel(
        block_order_smem,
        n_walked_blocks_smem,
        _d_edge_scalars,
        _dy,
        d_edge_scalars_hbm,
        dy_hbm,
    ):
        n_walked_blocks = n_walked_blocks_smem[0]
        unwalked_block = lambda i: block_order_smem[n_walked_blocks + i]  # noqa: E731

        def body(d_edge_scalars_vmem, dy_vmem):
            d_edge_scalars_vmem[...] = jnp.zeros_like(d_edge_scalars_vmem)
            dy_vmem[...] = jnp.zeros_like(dy_vmem)

        pltpu.emit_pipeline(
            body,
            grid=(c.n_edge_blocks - n_walked_blocks,),
            in_specs=[],
            out_specs=[
                pl.BlockSpec(
                    (c.num_irrep_blocks, c.plan.edge_block_size, c.channels_padded),
                    lambda i: (0, unwalked_block(i), 0),
                ),
                pl.BlockSpec(
                    (c.plan.n_channel_strides, c.plan.edge_block_size, c.num_lanes),
                    lambda i: (0, unwalked_block(i), 0),
                ),
            ],
            core_axis=0,
            dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
        )(d_edge_scalars_hbm, dy_hbm)

    return pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct(d_edge_scalars.shape, c.dtype),
            jax.ShapeDtypeStruct(dy.shape, c.dtype),
        ],
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(d_edge_scalars.shape, memory_space=pltpu.HBM),
            pl.BlockSpec(dy.shape, memory_space=pltpu.HBM),
        ],
        out_specs=[
            pl.BlockSpec(d_edge_scalars.shape, memory_space=pltpu.HBM),
            pl.BlockSpec(dy.shape, memory_space=pltpu.HBM),
        ],
        grid=(c.num_cores,),
        input_output_aliases={2: 0, 3: 1},
        name="bwd_convolution_zero_unwalked_blocks",
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
        ),
    )(block_order, n_walked_blocks, d_edge_scalars, dy)


def _call_in_specs(ops: _BwdOperands) -> list[pl.BlockSpec]:
    """Array operands live in HBM as whole blocks; the edge bound lives in SMEM."""
    return [
        *(pl.BlockSpec(a.shape, memory_space=pltpu.HBM) for a in ops[:-2]),
        *(pl.BlockSpec(memory_space=pltpu.SMEM) for _ in ops[-2:]),
    ]


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
    - `edge_scalars`: `(n_edges, num_irrep_blocks, channels)`.
    - `dz`: `(n_nodes, out_dim, channels)`.
    - `senders` / `receivers`: `(n_edges,)`.
    - outputs `dx`: `(n_nodes, x_dim, channels)`, `dy`: `(n_edges, y_dim)`,
      `d_edge_scalars`: `(n_edges, num_irrep_blocks, channels)`.

    Algorithm
    ----------
    A single `pallas_call` runs over a megacore grid of `tpu_info.num_cores` with
    PARALLEL semantics, operands and outputs in HBM. `emit_pipeline` walks `n_edge_blocks`
    edge tiles of `edge_block_size` edges each, sized by `plan_blocks` to fit the
    double-buffered VMEM working set. `core_axis=0` gives each core a contiguous block
    range, so at most one sender group straddles a core boundary: `dx` groups are written
    in place to a shared `(n_nodes, x_dim, channels)` buffer and only each core's
    first-closed and final groups go to a `(num_cores, 2, ...)` side buffer, added back
    in the epilogue. `dy` and `d_edge_scalars` are written per edge.

    Per pipeline step (body), working in VMEM:
      1. Gather per edge: async DMA `x[senders[k]]` and `dz[receivers[k]]` into
         per-edge stages, one edge per row (also prefetches the next tile).
      2. Slice the gathered tiles into per-component planes: `x[xi]` and `dz[zi]`
         as `(edge_block_size, channels)` operands.
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
        edge_is_real: jax.Array | None = None,
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
        _, num_irrep_blocks, _ = edge_scalars.shape
        edge_scalars = jnp.transpose(
            edge_scalars, (1, 0, 2)
        )  # (num_irrep_blocks, n_edges, channels)
        out_dim = params.z_space.dim

        channels_padded = round_up(channels, num_lanes)
        bwd_config = params.bwd_config
        lane_tile = min(channels_padded, num_lanes)
        x_dim_padded = round_up(x_dim, num_sublanes)

        def vmem_bytes(edge_block_size: int, channel_block: int) -> int:
            scratch, operands, values = _vmem_shapes(
                out_dim,
                x_dim,
                y_dim,
                x_dim_padded,
                chunk_rows,
                num_irrep_blocks,
                num_lanes,
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
            min_strided_edge_block_size=_MIN_STRIDED_EDGE_BLOCK_SIZE,
            vmem_capacity_bytes=tpu_info.vmem_capacity_bytes,
            edge_block_size=bwd_config.edge_block_size,
            channel_block=bwd_config.channel_block,
        )
        channel_block = plan.channel_block
        n_channel_strides = plan.n_channel_strides
        edge_block_size = plan.edge_block_size
        assert (
            channels_padded % channel_block == 0 and channel_block % lane_tile == 0
        ), (
            f"channel_block={channel_block} must be a whole number of {lane_tile}-lane "
            f"tiles dividing the padded channel count {channels_padded}."
        )
        assert (
            bwd_config.assume_sender_sorted
        ), "bwd requires sender-sorted edges; sort outside the kernel otherwise."
        x = jnp.pad(x, (no_pad, no_pad, (0, channels_padded - channels)))
        dz = jnp.pad(dz, (no_pad, no_pad, (0, channels_padded - channels)))
        x = jnp.transpose(
            x.reshape(n_nodes, x_dim, n_channel_strides, channel_block), (2, 0, 1, 3)
        ).reshape(n_channel_strides, n_nodes, x_dim * channel_block)
        dz = jnp.transpose(
            dz.reshape(n_nodes, out_dim, n_channel_strides, channel_block), (2, 0, 1, 3)
        ).reshape(n_channel_strides, n_nodes, out_dim * channel_block)
        edge_scalars = jnp.pad(
            edge_scalars, (no_pad, no_pad, (0, channels_padded - channels))
        )
        # Pad the edge axis to a whole number of per-core blocks; padded edges
        # repeat the last sender/receiver so no extra group is created.
        n_edges_padded = round_up(n_edges, edge_block_size * num_cores)
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

        n_edge_blocks = n_edges_padded // edge_block_size
        blocks_per_core = n_edge_blocks // num_cores
        senders = senders.reshape(n_edge_blocks, edge_block_size)
        receivers = receivers.reshape(n_edge_blocks, edge_block_size)
        zeros_dx = jnp.zeros((n_nodes, x_dim_padded, channels_padded), dtype)
        zeros_boundary = jnp.zeros((num_cores, 2, x_dim_padded, channels_padded), dtype)

        dynamic_edges = edge_is_real is not None
        if dynamic_edges:
            block_order, n_walked_blocks = padding.block_order(
                edge_is_real, n_edge_blocks, edge_block_size
            )
        else:
            block_order = jnp.zeros((1,), jnp.int32)  # unread: index maps stay static
            n_walked_blocks = jnp.int32(n_edge_blocks)
        n_walked_blocks = jnp.reshape(n_walked_blocks.astype(jnp.int32), (1,))

        ops = _BwdOperands(
            x,
            dz,
            y,
            edge_scalars,
            senders,
            receivers,
            zeros_dx,
            zeros_boundary,
            block_order,
            n_walked_blocks,
        )
        constants = _BwdConstants(
            num_cores=num_cores,
            n_edge_blocks=n_edge_blocks,
            blocks_per_core=blocks_per_core,
            n_nodes=n_nodes,
            x_dim=x_dim,
            x_dim_padded=x_dim_padded,
            out_dim=out_dim,
            channels=channels,
            channels_padded=channels_padded,
            plan=plan,
            y_dim=y_dim,
            num_irrep_blocks=num_irrep_blocks,
            n_edges=n_edges,
            n_edges_padded=n_edges_padded,
            chunk_rows=chunk_rows,
            num_lanes=num_lanes,
            dtype=dtype,
            cg_by_block=cg_by_block,
            xi_used=xi_used,
            yi_used=yi_used,
            irrep_block_of_output=irrep_block_of_output,
            dynamic_edges=dynamic_edges,
        )
        return ops, constants

    def _make_kernel(self, c: _BwdConstants):
        """Build the `pallas_call` kernel closing over the static plan `c`.

        `body` stays nested: it closes over the scratch and HBM refs that Pallas
        only binds as `kernel`'s parameters.
        """

        def _pipeline_in_specs(
            c: _BwdConstants, block_order_smem, stride: int
        ) -> list[pl.BlockSpec]:
            """emit_pipeline in_specs for (y, edge_scalars, senders, receivers,
            senders_next, receivers_next)."""
            walked_block = (
                (lambda i: block_order_smem[i]) if c.dynamic_edges else (lambda i: i)
            )
            next_walked_block = lambda i: walked_block(  # noqa: E731
                jnp.minimum(i + 1, c.n_edge_blocks - 1)
            )
            return [
                pl.BlockSpec(
                    (c.plan.edge_block_size, c.num_lanes),
                    lambda i: (walked_block(i), 0),
                ),
                pl.BlockSpec(
                    (c.num_irrep_blocks, c.plan.edge_block_size, c.plan.channel_block),
                    lambda i, s=stride: (0, walked_block(i), s),
                ),
                pl.BlockSpec(
                    (1, c.plan.edge_block_size),
                    lambda i: (walked_block(i), 0),
                    memory_space=pltpu.SMEM,
                ),
                pl.BlockSpec(
                    (1, c.plan.edge_block_size),
                    lambda i: (walked_block(i), 0),
                    memory_space=pltpu.SMEM,
                ),
                pl.BlockSpec(
                    (1, c.plan.edge_block_size),
                    lambda i: (next_walked_block(i), 0),
                    memory_space=pltpu.SMEM,
                ),  # prefetch next senders
                pl.BlockSpec(
                    (1, c.plan.edge_block_size),
                    lambda i: (next_walked_block(i), 0),
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
            _z_boundary,
            block_order_smem,
            n_walked_blocks_smem,
            # outputs:
            d_edge_scalars_hbm,
            dy_hbm,
            dx_hbm,
            boundary_hbm,
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
            flush_count_smem,
            gather_semaphore,
            stage_semaphore,
            flush_semaphore,
        ):
            core_id = pl.program_id(0)

            if c.dynamic_edges:
                n_edge_blocks = n_walked_blocks_smem[0]
                base = n_edge_blocks // c.num_cores
                blocks_this_core = base + jnp.where(
                    core_id < n_edge_blocks - base * c.num_cores, 1, 0
                )
            else:
                n_edge_blocks = c.n_edge_blocks
                blocks_this_core = jnp.int32(c.blocks_per_core)

            def run_stride(stride: int):
                # zero the accumulator; flushes re-zero it after each group
                accumulator[:, :, :] = jnp.zeros_like(accumulator[:, :, :])

                current_sender_smem[0] = jnp.int32(-1)
                count_smem[0] = jnp.int32(0)
                flush_pending_smem[0] = jnp.int32(0)
                flush_count_smem[0] = jnp.int32(0)

                # per-sender-group dx reduce + async DMA out (double-buffered).
                flush_one, drain_flush = make_group_flush(
                    FlushBuffers(
                        dst_hbm=dx_hbm,
                        stage_vmem=flush_stage,
                        accumulator_vmem=accumulator,
                        flush_pending=flush_pending_smem,
                        semaphore=flush_semaphore,
                        boundary_hbm=boundary_hbm,
                        flush_count=flush_count_smem,
                        channel_slice=c.plan.channel_slice(stride),
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
                    receivers_next_read = lambda k: receivers_next_smem[
                        0, k
                    ]  # noqa: E731

                    # ---- 1. gather x[sender] and dz[receiver], one edge per row ----
                    def gather_copies(read_sender, read_receiver):
                        x_copies, dz_copies = [], []
                        for k in range(c.plan.edge_block_size):
                            x_copies.append(
                                pltpu.make_async_copy(
                                    x_hbm.at[stride, pl.ds(read_sender(k), 1), :],
                                    x_stage.at[pl.ds(k, 1), :],
                                    gather_semaphore,
                                )
                            )
                            dz_copies.append(
                                pltpu.make_async_copy(
                                    dz_hbm.at[stride, pl.ds(read_receiver(k), 1), :],
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

                    zero = jnp.zeros(
                        (c.plan.edge_block_size, c.plan.channel_block), c.dtype
                    )

                    # ---- 2. lane-slice the folded rows into per-component planes ----
                    lanes = c.plan.channel_block
                    with named_scope("extract"):
                        for xi in range(c.x_dim):
                            x_plane[xi, :, :] = x_stage[
                                :, lanes * xi : lanes * (xi + 1)
                            ]
                        for zi in range(c.out_dim):
                            dz_plane[zi, :, :] = dz_stage[
                                :, lanes * zi : lanes * (zi + 1)
                            ]

                    count_smem[0] = block_idx + 1

                    # prefetch the next block's gathers now
                    @pl.when(block_idx < blocks_this_core - 1)
                    def _prefetch():
                        for copy in gather_copies(
                            senders_next_read, receivers_next_read
                        ):
                            copy.start()

                    y_broadcast = {
                        yi: jnp.broadcast_to(
                            y_vmem[:, yi][:, None],
                            (c.plan.edge_block_size, c.plan.channel_block),
                        )
                        for yi in c.yi_used
                    }
                    y_of = lambda yi: y_broadcast[yi]  # noqa: E731

                    # ---- 3. CG sweep: all three gradients in one CG pass ----
                    with named_scope("cg_sweep"):
                        dx_acc = {xi: zero for xi in c.xi_used}
                        dy_acc = {yi: zero for yi in c.yi_used}
                        for block, zi_group in c.cg_by_block:
                            cg_dz = {}
                            for zi, paths in zi_group:
                                dz_zi = dz_plane[zi, :, :]
                                for xi, yi, v in paths:
                                    term = v * dz_zi
                                    cg_dz[xi, yi] = (
                                        term
                                        if (xi, yi) not in cg_dz
                                        else cg_dz[xi, yi] + term
                                    )
                            dx_ungated = {}
                            dy_ungated = {}
                            for (xi, yi), cg_dz_term in cg_dz.items():
                                dx_term = y_of(yi) * cg_dz_term
                                dx_ungated[xi] = (
                                    dx_term
                                    if xi not in dx_ungated
                                    else dx_ungated[xi] + dx_term
                                )
                                dy_term = x_plane[xi, :, :] * cg_dz_term
                                dy_ungated[yi] = (
                                    dy_term
                                    if yi not in dy_ungated
                                    else dy_ungated[yi] + dy_term
                                )
                            edge_scalars_block = edge_scalars_vmem[block, :, :]
                            d_edge_scalars_acc = zero
                            for xi, dx_term in dx_ungated.items():
                                dx_acc[xi] = dx_acc[xi] + edge_scalars_block * dx_term
                                d_edge_scalars_acc = (
                                    d_edge_scalars_acc + x_plane[xi, :, :] * dx_term
                                )
                            for yi, dy_term in dy_ungated.items():
                                dy_acc[yi] = dy_acc[yi] + edge_scalars_block * dy_term
                            d_edge_scalars_vmem[block, :, :] = d_edge_scalars_acc
                        for xi in c.xi_used:
                            dx_plane[xi, :, :] = dx_acc[xi]

                        # dy: reduce the channel axis.
                        dy_vmem[0, :, :] = jnp.zeros(
                            (c.plan.edge_block_size, c.num_lanes), c.dtype
                        )
                        for yi in c.yi_used:
                            dy_vmem[0, :, yi] = jnp.sum(dy_acc[yi], axis=-1)
                    dx_indices = list(c.xi_used)

                    # ---- 4. walk: accumulate dx_plane per sender group, scatter to dx.
                    chunk_edge_position = jax.lax.broadcasted_iota(
                        jnp.int32, (c.chunk_rows, c.plan.channel_block), 0
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

                walked_block = (
                    (lambda i: block_order_smem[i])
                    if c.dynamic_edges
                    else (lambda i: i)
                )
                pipe = pltpu.emit_pipeline(
                    body,
                    grid=(n_edge_blocks,),
                    in_specs=_pipeline_in_specs(c, block_order_smem, stride),
                    out_specs=[
                        pl.BlockSpec(
                            (
                                c.num_irrep_blocks,
                                c.plan.edge_block_size,
                                c.plan.channel_block,
                            ),
                            lambda i, s=stride: (0, walked_block(i), s),
                        ),
                        pl.BlockSpec(
                            (1, c.plan.edge_block_size, c.num_lanes),
                            lambda i, s=stride: (s, walked_block(i), 0),
                        ),
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
                    flush_one(final_sender, final=True)

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
        dz: jax.Array,
        params: PallasMosaicTPUMessagePassingConvolutionParams,
        edge_is_real: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        ops, c = self._prolog(
            x, y, edge_scalars, senders, receivers, dz, params, edge_is_real
        )

        d_edge_scalars_packed, dy_packed, dx_packed, dx_boundary = pl.pallas_call(
            self._make_kernel(c),
            out_shape=[
                jax.ShapeDtypeStruct(
                    (c.num_irrep_blocks, c.n_edges_padded, c.channels_padded), c.dtype
                ),
                jax.ShapeDtypeStruct(
                    (c.plan.n_channel_strides, c.n_edges_padded, c.num_lanes), c.dtype
                ),
                jax.ShapeDtypeStruct(
                    (c.n_nodes, c.x_dim_padded, c.channels_padded), c.dtype
                ),
                jax.ShapeDtypeStruct(
                    (c.num_cores, 2, c.x_dim_padded, c.channels_padded), c.dtype
                ),
            ],
            scratch_shapes=_scratch_shapes(c),
            in_specs=_call_in_specs(ops),
            out_specs=[
                pl.BlockSpec(
                    (c.num_irrep_blocks, c.n_edges_padded, c.channels_padded),
                    memory_space=pltpu.HBM,
                ),
                pl.BlockSpec(
                    (c.plan.n_channel_strides, c.n_edges_padded, c.num_lanes),
                    memory_space=pltpu.HBM,
                ),
                pl.BlockSpec(
                    (c.n_nodes, c.x_dim_padded, c.channels_padded),
                    memory_space=pltpu.HBM,
                ),
                pl.BlockSpec(
                    (c.num_cores, 2, c.x_dim_padded, c.channels_padded),
                    memory_space=pltpu.HBM,
                ),
            ],
            grid=(c.num_cores,),
            input_output_aliases={6: 2, 7: 3},  # use the zero-inits for the outputs
            name="bwd_convolution",
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
                disable_bounds_checks=True,
            ),
        )(*ops)

        if c.dynamic_edges:
            d_edge_scalars_packed, dy_packed = _zero_unwalked_blocks(
                d_edge_scalars_packed,
                dy_packed,
                ops.block_order,
                ops.n_walked_blocks,
                c,
            )

        # Each core's first-closed and final groups may be shared with a
        # neighbouring core, so they were parked in `dx_boundary` instead of being
        # written in place. Their node ids are the senders at the core's first and
        # last edge -- known here, no need to report them from the kernel.
        flat_senders = ops.senders.reshape(-1)
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
                row = flat_senders[edge].astype(jnp.int32)
                start = (row, jnp.int32(0), jnp.int32(0))
                current = jax.lax.dynamic_slice(
                    dx_packed, start, (1, c.x_dim_padded, c.channels_padded)
                )
                dx_packed = jax.lax.dynamic_update_slice(
                    dx_packed, current + dx_boundary[core, slot][None], start
                )

        dx = dx_packed[:, : c.x_dim, : c.channels]  # (n_nodes, x_dim, channels)

        dy = dy_packed.sum(0)[: c.n_edges, : c.y_dim]
        d_edge_scalars = jnp.transpose(d_edge_scalars_packed, (1, 0, 2))[
            : c.n_edges, :, : c.channels
        ]
        return dx, dy, d_edge_scalars


def _message_passing_kernel_mosaic_tpu_bwd(
    x: jax.Array,
    y: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    dz: jax.Array,
    params: PallasMosaicTPUMessagePassingConvolutionParams,
    edge_is_real: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _MessagePassingBwdKernel()(
        x, y, edge_scalars, senders, receivers, dz, params, edge_is_real
    )
