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

"""Backward kernel of the fused message-passing convolution on Mosaic GPU.

There are two versions of the kernel body. They compute the same gradients
and only differ in which endpoint of the edges each CTA is responsible for:

- :func:`_receiver_grouped_body`: each CTA owns a receiver and scatters its
  `dx` contributions to the senders.
- :func:`_sender_grouped_body`: each CTA owns a sender and keeps `dx` local.

Both use the per-edge math in :class:`_BackwardPlan`. The choice, made with
`bwd_config.grouping`, only changes performance, not the result.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import memref as memref_dialect
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.pallas import mosaic_gpu as plgpu

from e3j.pallas_ops.convolution.mosaic_gpu.common import (
    DY_TILE_ROW_GRANULARITY,
    NODE_AXIS,
    Shapes,
    csr_view,
    edge_slot,
    pad_edge_features,
    plan_channel_block,
    validate_operands,
)
from e3j.pallas_ops.convolution.mosaic_gpu.params import (
    PallasMosaicGPUMessagePassingConvolutionParams,
)
from e3j.pallas_ops.utils.coef import group_coef_by_zi_then_xi, group_coef_by_zi_then_yi
from e3j.utils import options

#: A `(channels,)` row held in registers, or `None` if no Clebsch-Gordan path
#: contributes to it (see :func:`_or_zero`).
Row = Any


def _or_zero(row: Row, zero_row: jax.Array) -> jax.Array:
    """Return `row`, or `zero_row` if no Clebsch-Gordan path contributed to it."""
    return zero_row if row is None else row


@dataclass(frozen=True)
class _BackwardPlan:
    """Constants and per-edge math shared by both backward kernel bodies.

    Everything here is fixed at trace time.

    Attributes:
        shapes: Sizes of the convolution. `channels` is the number of channels
            in this pass, not the total.
        dtype: Element type of all inputs and gradients.
        xi_paths: For each `zi`, its paths grouped by `xi`, as `(yi, value)`
            pairs.
        yi_paths: For each `zi`, its paths grouped by `yi`, as `(xi, value)`
            pairs.
        outputs_by_scalar: Pairs `(scalar, [zi, ...])` listing the output
            coordinates each edge scalar multiplies.
        row_layout: Register layout of every `(channels,)` row.
        dy_tile_rows: `y_dim` rounded up to a multiple of
            `DY_TILE_ROW_GRANULARITY`.
        dy_tile_layout: Layout used to read the `dy` staging tile.
        pipeline_depth: How many edges the receiver-grouped pipeline loads
            ahead.
        channel_offset: Where this pass's channels start in the full channel
            axis.
    """

    shapes: Shapes
    dtype: jnp.dtype
    xi_paths: dict[int, tuple]
    yi_paths: dict[int, tuple]
    outputs_by_scalar: tuple[tuple[int, tuple[int, ...]], ...]
    row_layout: Any
    dy_tile_rows: int
    dy_tile_layout: Any
    pipeline_depth: int
    channel_offset: int = 0

    @classmethod
    def build(
        cls,
        shapes: Shapes,
        dtype: jnp.dtype,
        params: PallasMosaicGPUMessagePassingConvolutionParams,
        *,
        channel_offset: int = 0,
    ) -> "_BackwardPlan":
        by_scalar: dict[int, list[int]] = defaultdict(list)
        for zi, scalar in enumerate(params.scalar_of_output.tolist()):
            by_scalar[scalar].append(zi)

        dy_tile_rows = (
            -(-shapes.y_dim // DY_TILE_ROW_GRANULARITY) * DY_TILE_ROW_GRANULARITY
        )
        return cls(
            shapes=shapes,
            dtype=dtype,
            xi_paths=dict(group_coef_by_zi_then_xi(params.indices, params.values)),
            yi_paths=dict(group_coef_by_zi_then_yi(params.indices, params.values)),
            outputs_by_scalar=tuple(
                (scalar, tuple(outputs))
                for scalar, outputs in sorted(by_scalar.items())
            ),
            row_layout=plgpu.Layout.WG_STRIDED((shapes.channels,), vec_size=1),
            dy_tile_rows=dy_tile_rows,
            dy_tile_layout=plgpu.Layout.SMEM_GMEM_COPY(
                (dy_tile_rows, shapes.channels), dtype, 128
            ),
            pipeline_depth=params.bwd_config.pipeline_depth,
            channel_offset=channel_offset,
        )

    @property
    def channel_slice(self) -> slice:
        """This pass's channels, as a slice for indexing refs."""
        return slice(self.channel_offset, self.channel_offset + self.shapes.channels)

    @property
    def channel_block_idx(self) -> int:
        """This pass's channel block index, for use in a `BlockSpec`."""
        return self.channel_offset // self.shapes.channels

    @property
    def x_dim(self) -> int:
        return self.shapes.x_dim

    @property
    def y_dim(self) -> int:
        return self.shapes.y_dim

    def zero_row(self) -> jax.Array:
        """Return a `(channels,)` row of zeros in `row_layout`.

        It has to be created inside the kernel, since kernels can't capture
        array constants. The cast makes its layout match the rows the loop
        body produces, which `fori_loop` requires.
        """
        return plgpu.layout_cast(
            jnp.zeros((self.shapes.channels,), self.dtype), self.row_layout
        )

    def _row(self, ref) -> jax.Array:
        return plgpu.load(ref, layout=self.row_layout)

    def _fold_y(self, y_smem, paths, zero_row: jax.Array) -> jax.Array:
        """Compute `sum_yi v * y[yi]` for one `(zi, xi)` group.

        The result is broadcast to a `(channels,)` row. Each `y[yi]` is a
        single scalar in shared memory, so we write the broadcast-and-FMA
        chain by hand with `inline_mgpu`.
        """

        @plgpu.inline_mgpu(
            arg_types=(self.row_layout, plgpu.RefType()),
            return_type=plgpu.ShapeDtypeStruct(
                (self.shapes.channels,), self.dtype, layout=self.row_layout
            ),
        )
        def accumulate(ctx, w, y_smem):
            # This function runs later, during lowering, not when it's defined.
            # It still sees the right `paths` because `paths` is an argument of
            # `_fold_y`, so each `(zi, xi)` group gets its own copy. If it
            # captured the caller's loop variable instead, every group would
            # see the last value of that loop.
            idx = ir.IndexType.get()
            for yi, value in paths:
                s = memref_dialect.load(y_smem, [mgpu.c(0, idx), mgpu.c(yi, idx)])
                term = mgpu.FragmentedArray.splat(s, shape=w.shape, layout=w.layout)
                coef = mgpu.FragmentedArray.splat(
                    mgpu.c(value, w.mlir_dtype), shape=w.shape, layout=w.layout
                )
                w = w + coef * term
            return w

        return accumulate(zero_row, y_smem)

    def edge_cotangents(
        self,
        *,
        edge,
        zero_row: jax.Array,
        ct_smem,  # (1, z_dim, channels): output gradient of the receiver
        x_smem,  # (1, x_dim, channels)
        y_smem,  # (1, y_dim_padded)
        s_smem,  # (1, num_scalars, channels)
        ds_gmem,  # (n_edges, num_scalars, total_channels): written at `channel_slice`
    ) -> tuple[list[Row], list[Row]]:
        """Write `ds[edge]` and return this edge's `dx` and `dy` rows.

        The caller decides how to store `dx`, which depends on the grouping.
        It then sums `dy` over channels with :meth:`stage_dy` and
        :meth:`reduce_dy`.
        """
        dx: list[Row] = [None] * self.x_dim
        dy: list[Row] = [None] * self.y_dim
        for scalar_index, outputs in self.outputs_by_scalar:
            mixing = self._row(s_smem.at[0, scalar_index])
            ds = None
            for zi in outputs:
                ct_row = self._row(ct_smem.at[0, zi])
                mixed_ct = ct_row * mixing

                # Each sum below starts from its first term instead of a row of
                # zeros, which saves an add. A value still `None` at the end
                # means no path contributes to it (see `_or_zero`).
                tp = None
                for xi, paths in self.xi_paths.get(zi, ()):
                    weight = self._fold_y(y_smem, paths, zero_row)
                    tp_term = weight * self._row(x_smem.at[0, xi])
                    tp = tp_term if tp is None else tp + tp_term
                    dx_term = weight * mixed_ct
                    dx[xi] = dx_term if dx[xi] is None else dx[xi] + dx_term

                if tp is not None:
                    ds_term = ct_row * tp
                    ds = ds_term if ds is None else ds + ds_term

                for yi, paths in self.yi_paths.get(zi, ()):
                    (first_xi, first_value), *rest_paths = paths
                    weighted_x = first_value * self._row(x_smem.at[0, first_xi])
                    for path_xi, value in rest_paths:
                        weighted_x += value * self._row(x_smem.at[0, path_xi])
                    dy_term = mixed_ct * weighted_x
                    dy[yi] = dy_term if dy[yi] is None else dy[yi] + dy_term

            ds_gmem[edge, scalar_index, self.channel_slice] = _or_zero(ds, zero_row)
        return dx, dy

    def stage_dy(self, dy: list[Row], dy_rows_smem, zero_row: jax.Array) -> None:
        """Write the `dy` rows to shared memory.

        Call `commit_smem` before :meth:`reduce_dy` reads them back.
        """
        for yi in range(self.y_dim):
            dy_rows_smem[yi, :] = _or_zero(dy[yi], zero_row)

    def reduce_dy(self, edge, dy_rows_smem, dy_gmem) -> None:
        """Sum the staged `dy` tile over channels and write it to `dy[edge]`.

        This overwrites `dy[edge]` instead of adding to it. Each channel pass
        writes its own `dy_gmem`, and the passes are summed outside the
        kernel. Adding in place would mean reading back from global memory,
        which isn't allowed for rows this short (under 128 elements).
        """
        dy_gmem[edge, 0 : self.dy_tile_rows] = jnp.sum(
            plgpu.load(dy_rows_smem, layout=self.dy_tile_layout), axis=-1
        )

    def zero_dropped_edges(
        self,
        *,
        node,
        offsets_gmem,
        dy_gmem,
        ds_gmem,
        dy_rows_smem,
        zero_row: jax.Array,
        slot_gmem=None,
    ) -> None:
        """Write zeros to `dy` and `ds` for the padding edges.

        Padding edges don't belong to any node's range, so the main walk never
        writes them. They are the edges `[offsets[n_nodes], n_edges)` at the
        end of the list. CTA `n` handles every `n_nodes`-th of them, starting
        at `n`. Zero-initializing both buffers with `jnp.zeros` instead was
        21.6% slower.
        """
        n_nodes, n_edges = self.shapes.n_nodes, self.shapes.n_edges
        walked = plgpu.load(offsets_gmem.at[n_nodes])
        first = walked + node

        # The zeros go through the staging tile and `reduce_dy`, because that
        # reduction gives the value the layout the store expects. The tile is
        # filled once, before the loop, and the loop only reads it. Refilling
        # it for every edge could overwrite it while other threads are still
        # reading the previous edge's values. This also requires the main walk
        # to be done with the tile, which the barrier at the end of both
        # kernel bodies guarantees.
        @pl.when(first < n_edges)
        def _stage_zeros():
            for row in range(self.dy_tile_rows):
                dy_rows_smem[row, :] = zero_row
            plgpu.commit_smem()

        def zero_and_advance(position):
            edge = edge_slot(slot_gmem, position)
            for scalar_index in range(self.shapes.num_scalars):
                ds_gmem[edge, scalar_index, self.channel_slice] = zero_row
            self.reduce_dy(edge, dy_rows_smem, dy_gmem)
            return position + n_nodes

        jax.lax.while_loop(lambda p: p < n_edges, zero_and_advance, first)


# ------------------------------------------------------- receiver-grouped ---


def _receiver_grouped_scratch(plan: _BackwardPlan) -> list:
    """Shared memory for the receiver-grouped body.

    The per-edge buffers are not listed here; `emit_pipeline` allocates them.
    """
    shapes, dtype = plan.shapes, plan.dtype
    return [
        plgpu.SMEM((1, shapes.z_dim, shapes.channels), dtype),  # cotangent rows
        plgpu.SMEM((1, shapes.x_dim, shapes.channels), dtype),  # dx, before storing
        plgpu.SMEM((plan.dy_tile_rows, shapes.channels), dtype),  # dy, before summing
        plgpu.Barrier(num_arrivals=1),  # waits for the node's cotangent copy
    ]


def _receiver_grouped_body(
    plan: _BackwardPlan,
    dx_gmem,  # (n_nodes, x_dim, total_channels), starts at zero, added into
    dy_gmem,  # (n_edges, y_dim_padded)
    ds_gmem,  # (n_edges, num_scalars, total_channels)
    x_gmem,  # (n_nodes, x_dim, total_channels): all channels, see `channel_block_idx`
    y_gmem,
    s_gmem,  # (n_edges, num_scalars, total_channels): all channels
    gather_gmem,  # (n_edges,) sender of each step of the walk
    offsets_gmem,
    ct_gmem,  # (n_nodes, z_dim, total_channels): all channels, see `channel_slice`
    ct_smem,  # (1, z_dim, channels)
    dx_smem,  # (1, x_dim, channels)
    dy_rows_smem,  # (dy_tile_rows, channels): dy, before summing over channels
    node_barrier,
    *,
    slot_gmem=None,  # (n_edges,) edge permutation, or None if not needed
) -> None:
    """Backward body where each CTA owns one receiver node.

    The CTA loads `ct[node]` once and reads `x[senders[e]]` for each edge.
    Since several receivers can share a sender, `dx` is added into a buffer
    of zeros with TMA add-reduction copies.
    """
    node = jax.lax.axis_index(NODE_AXIS)
    first_edge = plgpu.load(offsets_gmem.at[node])
    last_edge = plgpu.load(offsets_gmem.at[node + 1])
    with jax.named_scope("copy_ct"):
        plgpu.copy_gmem_to_smem(
            ct_gmem.at[pl.ds(node, 1), :, plan.channel_slice], ct_smem, node_barrier
        )
        plgpu.barrier_wait(node_barrier)
    zero_row = plan.zero_row()

    def edge_body(indices, x_smem, y_smem, s_smem):
        (step,) = indices
        position = first_edge + step
        edge = edge_slot(slot_gmem, position)
        # Wait for the previous edge's `dx` copy to finish reading shared
        # memory. Waiting here, rather than right after starting the copy,
        # lets the copy run while we compute, instead of delaying the next
        # edge's loads.
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

        dx, dy = plan.edge_cotangents(
            edge=edge,
            zero_row=zero_row,
            ct_smem=ct_smem,
            x_smem=x_smem,
            y_smem=y_smem,
            s_smem=s_smem,
            ds_gmem=ds_gmem,
        )
        plan.stage_dy(dy, dy_rows_smem, zero_row)
        # Write every row, including those no path contributes to, because
        # the buffer still holds the previous edge's values.
        for xi in range(plan.x_dim):
            dx_smem[0, xi] = _or_zero(dx[xi], zero_row)

        # This does two jobs. It is the memory fence needed before `dx_smem`
        # and `dy_rows_smem` are read back. It is also the warpgroup barrier
        # that keeps the pipeline from refilling the input buffers while this
        # body is still reading them. With Pallas' automatic barriers turned
        # off, nothing else would ensure that.
        plgpu.commit_smem()
        sender = plgpu.load(gather_gmem.at[position])
        plgpu.copy_smem_to_gmem(
            dx_smem,
            # `dx_gmem` holds all channels and is shared by every pass. This
            # pass only writes its own channel slice.
            dx_gmem.at[pl.ds(sender, 1), :, plan.channel_slice],
            reduction_op="add",
        )
        plan.reduce_dy(edge, dy_rows_smem, dy_gmem)

    shapes = plan.shapes
    # The number of steps is the node's degree, known only at run time, and
    # can be zero. A node with no incoming edges simply runs no steps, which
    # is correct: `dy` and `ds` are per edge, and `dx` starts at zero.
    plgpu.emit_pipeline(
        edge_body,
        grid=(last_edge - first_edge,),
        in_specs=[
            # The blocks keep a leading axis of size 1 instead of dropping it
            # with `None`. That way they are indexed the same way as the
            # buffers the sender-grouped body fills by hand. It matters for
            # `y_smem` in particular, which `inline_mgpu` reads as a 2-D
            # memref.
            plgpu.BlockSpec(
                (1, shapes.x_dim, shapes.channels),
                lambda step: (
                    plgpu.load(gather_gmem.at[first_edge + step]),
                    0,
                    plan.channel_block_idx,
                ),
            ),
            plgpu.BlockSpec(
                (1, shapes.y_dim_padded),
                lambda step: (edge_slot(slot_gmem, first_edge + step), 0),
            ),
            plgpu.BlockSpec(
                (1, shapes.num_scalars, shapes.channels),
                lambda step: (
                    edge_slot(slot_gmem, first_edge + step),
                    0,
                    plan.channel_block_idx,
                ),
            ),
        ],
        max_concurrent_steps=plan.pipeline_depth,
    )(x_gmem, y_gmem, s_gmem)
    plgpu.wait_smem_to_gmem(0)

    # Done after the node's own edges, once `dy_rows_smem` is free again.
    plan.zero_dropped_edges(
        node=node,
        offsets_gmem=offsets_gmem,
        dy_gmem=dy_gmem,
        ds_gmem=ds_gmem,
        dy_rows_smem=dy_rows_smem,
        zero_row=zero_row,
        slot_gmem=slot_gmem,
    )


# --------------------------------------------------------- sender-grouped ---


def _sender_grouped_scratch(plan: _BackwardPlan) -> list:
    """Shared memory for the sender-grouped body, per-edge buffers included."""
    shapes, dtype = plan.shapes, plan.dtype
    return [
        plgpu.SMEM((1, shapes.x_dim, shapes.channels), dtype),  # x row block
        plgpu.SMEM((1, shapes.y_dim_padded), dtype),  # y row
        plgpu.SMEM((1, shapes.num_scalars, shapes.channels), dtype),  # edge scalars
        plgpu.SMEM((1, shapes.z_dim, shapes.channels), dtype),  # cotangent rows
        plgpu.SMEM((plan.dy_tile_rows, shapes.channels), dtype),  # dy, before summing
        plgpu.Barrier(num_arrivals=3),  # waits for the three per-edge copies
        plgpu.Barrier(num_arrivals=1),  # waits for the node's `x` copy
    ]


def _sender_grouped_body(
    plan: _BackwardPlan,
    dx_gmem,  # (n_nodes, x_dim, total_channels), written once per node
    dy_gmem,  # (n_edges, y_dim_padded)
    ds_gmem,  # (n_edges, num_scalars, total_channels)
    x_gmem,  # (n_nodes, x_dim, total_channels): all channels, see `channel_slice`
    y_gmem,
    s_gmem,  # (n_edges, num_scalars, total_channels): all channels
    gather_gmem,  # (n_edges,) receiver of each step of the walk
    offsets_gmem,
    ct_gmem,  # (n_nodes, z_dim, total_channels): all channels
    x_smem,  # (1, x_dim, channels)
    y_smem,  # (1, y_dim_padded)
    s_smem,  # (1, num_scalars, channels)
    ct_smem,  # (1, z_dim, channels)
    dy_rows_smem,  # (dy_tile_rows, channels): dy, before summing over channels
    edge_barrier,
    node_barrier,
    *,
    slot_gmem=None,  # (n_edges,) edge permutation, or None if not needed
) -> None:
    """Backward body where each CTA owns one sender node.

    The CTA loads `x[node]` once and reads `ct[receivers[e]]` for each edge.
    `dx[node]` is summed in registers and written once at the end. The `ct`
    rows are too large to double-buffer, so instead of a pipeline the edges
    are loaded one at a time in a `fori_loop`.
    """
    node = jax.lax.axis_index(NODE_AXIS)
    first_edge = plgpu.load(offsets_gmem.at[node])
    last_edge = plgpu.load(offsets_gmem.at[node + 1])

    plgpu.copy_gmem_to_smem(
        x_gmem.at[pl.ds(node, 1), :, plan.channel_slice], x_smem, node_barrier
    )
    plgpu.barrier_wait(node_barrier)
    zero_row = plan.zero_row()

    def edge_body(position, dx_total):
        edge = edge_slot(slot_gmem, position)
        receiver = plgpu.load(gather_gmem.at[position])
        plgpu.copy_gmem_to_smem(
            ct_gmem.at[pl.ds(receiver, 1), :, plan.channel_slice],
            ct_smem,
            edge_barrier,
        )
        plgpu.copy_gmem_to_smem(y_gmem.at[pl.ds(edge, 1)], y_smem, edge_barrier)
        plgpu.copy_gmem_to_smem(
            s_gmem.at[pl.ds(edge, 1), :, plan.channel_slice], s_smem, edge_barrier
        )
        plgpu.barrier_wait(edge_barrier)

        dx, dy = plan.edge_cotangents(
            edge=edge,
            zero_row=zero_row,
            ct_smem=ct_smem,
            x_smem=x_smem,
            y_smem=y_smem,
            s_smem=s_smem,
            ds_gmem=ds_gmem,
        )
        plan.stage_dy(dy, dy_rows_smem, zero_row)
        # Memory fence and warpgroup barrier so that `dy_rows_smem` is fully
        # written before `reduce_dy` reads it.
        plgpu.commit_smem()
        plan.reduce_dy(edge, dy_rows_smem, dy_gmem)
        # A second barrier after the read. `reduce_dy` reads values written by
        # other threads, so the next edge's `stage_dy` must wait until every
        # thread is done reading the tile. With Pallas' automatic barriers
        # off, nothing else in between (the copy issue, the mbarrier wait)
        # guarantees that. The same barrier also stops the next iteration's
        # copies from overwriting the input buffers while they're being read.
        plgpu.commit_smem()

        return tuple(
            total if row is None else total + row for total, row in zip(dx_total, dx)
        )

    # `dx` is carried through the loop in registers. This is the advantage of
    # grouping by sender: no shared memory buffer and no read-modify-write in
    # global memory. A node that sends nothing runs no iterations and stores
    # the initial zeros.
    dx_total = jax.lax.fori_loop(
        first_edge,
        last_edge,
        edge_body,
        tuple([zero_row] * plan.x_dim),
    )
    for xi in range(plan.x_dim):
        dx_gmem[node, xi, plan.channel_slice] = dx_total[xi]

    # Done after the node's own edges, once `dy_rows_smem` is free again.
    plan.zero_dropped_edges(
        node=node,
        offsets_gmem=offsets_gmem,
        dy_gmem=dy_gmem,
        ds_gmem=ds_gmem,
        dy_rows_smem=dy_rows_smem,
        zero_row=zero_row,
        slot_gmem=slot_gmem,
    )


# ------------------------------------------------------------------ entry ---


def _convolution_kernel_mosaic_gpu_bwd(
    node_features: jax.Array,
    edge_features: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    cotangent: jax.Array,
    params: PallasMosaicGPUMessagePassingConvolutionParams,
    *,
    edges_sorted: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Compute the gradients of the fused convolution for its three inputs.

    Write `a = senders[e]` and `b = receivers[e]`, let `ct` be the gradient of
    the output, `sigma(zi)` the edge scalar that multiplies output `zi`, and
    `g[e, zi, c] = ct[b, zi, c] * s[e, sigma(zi), c]`. Then:

    ```
    ds[e, k, c]  = sum_{zi: sigma(zi) = k} ct[b, zi, c] * tp[e, zi, c]
    dx[a, xi, c] = sum_{e: senders[e] = a} sum_(zi, yi, v) v * y[e, yi] * g[e, zi, c]
    dy[e, yi]    = sum_c sum_(zi, xi, v) v * x[a, xi, c] * g[e, zi, c]
    ```

    where `tp` is the forward's tensor product before scaling by the edge
    scalars. The coefficients used here are `params.values`, without the
    parity signs that `params.forward_values` may add.

    Each CTA goes through the edges of one node. `bwd_config.grouping` decides
    whether that node is the receiver or the sender. By default it follows
    `params.graph_ordering`; the other choice needs an `argsort`, which costs
    about 2.8%. On an H100, the sender-grouped version takes 0.81x the time of
    the receiver-grouped one at `l_max=1`, 1.12x at `l_max=2` and 1.41x at
    `l_max=3`.

    `bwd_config.channel_block` splits the channels into separate passes. Each
    pass reads its slice of the inputs and writes its slice of `dx` and `ds`.
    `dy` has no channel axis, so the passes' results are summed.

    Args:
        node_features: `(n_nodes, x_dim, channels)`.
        edge_features: `(n_edges, y_dim)`, no channel axis.
        edge_scalars: `(n_edges, num_scalars, channels)`.
        senders: `(n_edges,)` node indices, sorted if the ordering is
            `SENDER`.
        receivers: `(n_edges,)` node indices, sorted if the ordering is
            `RECEIVER`.
        cotangent: `(n_nodes, z_dim, channels)` gradient of the output.
        params: Kernel parameters: the Clebsch-Gordan coefficients, the graph
            ordering and the backward's grouping.
        edges_sorted: Whether the edges really are sorted by the endpoint in
            `params.graph_ordering`. If `False` (for example under `vmap`),
            the edges are reordered with an `argsort` first.

    Returns:
        `(dx, dy, ds)`, with the same shapes as `node_features`,
        `edge_features` and `edge_scalars`. Padding edges get zero `dy` and
        `ds`.
    """
    shapes = validate_operands(
        node_features, edge_features, edge_scalars, senders, receivers, params
    )
    n_nodes, n_edges = shapes.n_nodes, shapes.n_edges
    x_dim, y_dim_padded = shapes.x_dim, shapes.y_dim_padded
    channels, num_scalars = shapes.channels, shapes.num_scalars
    if cotangent.shape != (n_nodes, shapes.z_dim, channels):
        raise ValueError(
            f"cotangent must be {(n_nodes, shapes.z_dim, channels)}, "
            f"got {cotangent.shape}"
        )
    channel_block = plan_channel_block(shapes, params, backward=True)

    dtype = node_features.dtype
    if n_edges == 0:
        # With no edges, all gradients are zero. Launching the kernel anyway
        # would crash, since it would copy from empty arrays.
        return (
            jnp.zeros_like(node_features),
            jnp.zeros_like(edge_features),
            jnp.zeros_like(edge_scalars),
        )
    grouping = params.bwd_grouping
    receiver_grouped = grouping == options.GraphOrdering.RECEIVER
    edge_features = pad_edge_features(edge_features, y_dim_padded)
    offsets, gather_index, edge_perm = csr_view(
        senders,
        receivers,
        n_nodes,
        grouping,
        params.graph_ordering if edges_sorted else None,
    )

    def run_pass(dx_ref, ds_ref, channel_offset: int) -> jax.Array:
        """Launch the kernel on one block of channels and return its `dy`."""
        plan = _BackwardPlan.build(
            shapes._replace(channels=channel_block),
            dtype,
            params,
            channel_offset=channel_offset,
        )
        operands = [
            node_features,
            edge_features,
            edge_scalars,
            gather_index,
            offsets,
            cotangent,
        ]
        if edge_perm is not None:
            operands.append(edge_perm)
        body = _receiver_grouped_body if receiver_grouped else _sender_grouped_body
        scratch_types = (
            _receiver_grouped_scratch(plan)
            if receiver_grouped
            else _sender_grouped_scratch(plan)
        )

        def launch_dy(dy_ref):
            def kernel(*refs):
                # `refs` holds the six inputs, then the edge permutation if
                # there is one, then the scratch buffers.
                body(
                    plan,
                    dx_ref,
                    dy_ref,
                    ds_ref,
                    *refs[:6],
                    *refs[len(operands) :],
                    slot_gmem=refs[6] if edge_perm is not None else None,
                )

            plgpu.kernel(
                kernel,
                out_type=(),
                scratch_types=scratch_types,
                compiler_params=plgpu.CompilerParams(
                    unsafe_no_auto_barriers=True,
                    lowering_semantics=plgpu.LoweringSemantics.Lane,
                ),
                grid=(n_nodes,),
                grid_names=(NODE_AXIS,),
                kernel_name="convolution_mgpu_bwd",
            )(*operands)

        # The node walks write every non-padding edge's row, and
        # `zero_dropped_edges` writes the padding rows, so the whole buffer is
        # written and can start uninitialized.
        return pl.run_state(launch_dy)(pl.empty((n_edges, y_dim_padded), dtype))

    def launch(outputs) -> None:
        dx_ref, dy_ref, ds_ref = outputs
        dy_total = None
        for start in range(0, channels, channel_block):
            # `dy` has no channel axis. Each pass sums over its own channels
            # only, so the full sum over channels is the sum of all passes.
            # The padding columns past `y_dim` hold leftover values from
            # `reduce_dy`, so we drop them before adding the passes together.
            dy_part = run_pass(dx_ref, ds_ref, start)[:, : shapes.y_dim]
            dy_total = dy_part if dy_total is None else dy_total + dy_part
        # This writes the whole array at once, outside the kernel, so the
        # layout restrictions `reduce_dy` has to deal with don't apply.
        dy_ref[...] = dy_total

    dx, dy, ds = pl.run_state(launch)(
        (
            # Only the receiver-grouped body needs `dx` to start at zero,
            # since several CTAs add into the same rows. The passes write
            # separate channel slices, so zeroing the buffer once is enough
            # for all of them.
            (
                jnp.zeros((n_nodes, x_dim, channels), dtype)
                if receiver_grouped
                else pl.empty((n_nodes, x_dim, channels), dtype)
            ),
            # `launch` overwrites all of it, so it can start uninitialized.
            pl.empty((n_edges, shapes.y_dim), dtype),
            pl.empty((n_edges, num_scalars, channels), dtype),
        )
    )
    # Same reason as in the forward: without this barrier, XLA can merge a
    # loop's `ys` update into the kernel calls and lose what they wrote.
    return jax.lax.optimization_barrier((dx, dy, ds))
