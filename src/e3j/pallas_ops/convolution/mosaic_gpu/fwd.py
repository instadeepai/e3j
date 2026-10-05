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

"""Forward kernel of the fused message-passing convolution on Mosaic GPU.

A single kernel gathers the sender features, takes the tensor product with
the edge features, scales it by the edge scalars and sums the messages on each
node.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from e3j.pallas_ops.convolution.mosaic_gpu.common import (
    NODE_AXIS,
    csr_view,
    edge_slot,
    pad_edge_features,
    plan_channel_block,
    used_input_coordinates,
    validate_operands,
)
from e3j.pallas_ops.convolution.mosaic_gpu.params import (
    PallasMosaicGPUMessagePassingConvolutionParams,
)
from e3j.pallas_ops.utils.coef import group_coef_by_zi_then_xi


def _convolution_kernel_mosaic_gpu_fwd(
    node_features: jax.Array,
    edge_features: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    params: PallasMosaicGPUMessagePassingConvolutionParams,
    *,
    edges_sorted: bool = True,
) -> jax.Array:
    r"""Compute the messages on each edge and sum them on the receiver nodes.

    ```
    out[b, zi, c] = sum_{e: receivers[e] = b} s[e, scalar(zi), c]
                    * sum_(xi, yi, v) v * x[senders[e], xi, c] * y[e, yi]
    ```

    Each node gets one CTA. The CTA goes through the node's edges (see
    :func:`common.csr_view`) and keeps the running sum in registers. This
    means no atomics are needed and each output row is written exactly once.
    The Clebsch-Gordan paths are unrolled while tracing, with one vector FMA
    per `(zi, xi)` group. If there are more channels than
    `fwd_config.channel_block`, the kernel is launched several times, each
    launch writing its own slice of channels in the same output buffer.

    With `SENDER` grouping, senders and receivers swap roles. That is only
    correct under the symmetry assumptions of :class:`e3j.core.Convolution`;
    if they don't hold, set `fwd_config.grouping=RECEIVER`.

    Args:
        node_features: `(n_nodes, x_dim, channels)`.
        edge_features: `(n_edges, y_dim)`.
        edge_scalars: `(n_edges, num_scalars, channels)`.
        senders: `(n_edges,)` node indices.
        receivers: `(n_edges,)` node indices.
        params: Kernel parameters.
        edges_sorted: Whether the edges really are sorted as
            `params.graph_ordering` says. If `False` (for example under
            `vmap`), the edges are reordered with an `argsort` first.

    Returns:
        `(n_nodes, z_dim, channels)` summed messages.
    """
    config = params.fwd_config
    shapes = validate_operands(
        node_features, edge_features, edge_scalars, senders, receivers, params
    )
    channel_block = plan_channel_block(shapes, params, backward=False)
    # Also plan the backward and throw the result away. If the backward could
    # not run with these settings, it's better to fail now than at the first
    # gradient computation.
    plan_channel_block(shapes, params, backward=True)

    coef_by_zi_then_xi = group_coef_by_zi_then_xi(params.indices, params.forward_values)
    used_xis, used_yis = used_input_coordinates(params)
    scalar_of_output = tuple(int(s) for s in params.scalar_of_output)

    dtype = node_features.dtype
    n_nodes = shapes.n_nodes
    if shapes.n_edges == 0:
        # With no edges the result is all zeros. Launching the kernel anyway
        # would crash, since it would index into empty edge arrays.
        return jnp.zeros((n_nodes, shapes.z_dim, shapes.channels), dtype)
    edge_features = pad_edge_features(edge_features, shapes.y_dim_padded)
    offsets, gather_index, edge_perm = csr_view(
        senders,
        receivers,
        n_nodes,
        params.fwd_grouping,
        params.graph_ordering if edges_sorted else None,
    )

    def run_pass(out_gmem, channel_offset: int) -> None:
        """Launch the kernel on the channel slice starting at `channel_offset`."""
        # Register layout used for every `(channel_block,)` row in the kernel.
        row_layout = plgpu.Layout.WG_STRIDED((channel_block,), vec_size=1)
        # `BlockSpec` index maps count in blocks, not elements. This division
        # is exact because `channel_offset` is a multiple of `channel_block`.
        channel_block_idx = channel_offset // channel_block

        def kernel(
            x_gmem,  # (n_nodes, x_dim, total_channels), sliced by the BlockSpec
            y_gmem,  # (n_edges, y_dim_padded)
            s_gmem,  # (n_edges, num_scalars, total_channels), likewise
            gather_gmem,  # (n_edges,): the endpoint edges are *not* grouped by
            offsets_gmem,  # (n_nodes + 1,)
            *rest,  # the edge permutation, if the edges need reordering
        ):
            slot_gmem = rest[0] if rest else None
            node = jax.lax.axis_index(NODE_AXIS)
            first_edge = plgpu.load(offsets_gmem.at[node])
            last_edge = plgpu.load(offsets_gmem.at[node + 1])

            def zero_row():
                # Created inside the kernel because kernel bodies can't capture
                # array constants. The cast gives it the same layout as the
                # rows the pipeline body returns.
                return plgpu.layout_cast(jnp.zeros((channel_block,), dtype), row_layout)

            def edge_of(step):
                return edge_slot(slot_gmem, first_edge + step)

            def store(totals) -> None:
                # The output buffer starts uninitialized, so every row must be
                # written, even the ones that stay zero.
                for zi in range(shapes.z_dim):
                    out_gmem[
                        node, zi, channel_offset : channel_offset + channel_block
                    ] = totals[zi]

            def add_edge(indices, x_smem, y_smem, s_smem, accumulator):
                (step,) = indices
                # Load each input row from shared memory once. An `xi` used by
                # several outputs is then reused from registers.
                x_rows = {
                    xi: plgpu.load(x_smem.at[xi], layout=row_layout) for xi in used_xis
                }
                y_scalars = {yi: y_smem[yi] for yi in used_yis}

                accumulator = list(accumulator)
                for zi, by_xi in coef_by_zi_then_xi:
                    message = None
                    for xi, paths in by_xi:
                        # Combine the scalar coefficients first, then do a
                        # single vector FMA per (zi, xi). Both sums start from
                        # their first term instead of zero, which saves an add.
                        (first_yi, first_value), *rest_paths = paths
                        weight = first_value * y_scalars[first_yi]
                        for yi, value in rest_paths:
                            weight += value * y_scalars[yi]
                        term = weight * x_rows[xi]
                        message = term if message is None else message + term
                    mixing = plgpu.load(
                        s_smem.at[scalar_of_output[zi]], layout=row_layout
                    )
                    accumulator[zi] = accumulator[zi] + mixing * message

                # The pipeline reuses its staging buffers in turn, so the next
                # copy it starts writes into the buffer we just read. We turned
                # off Pallas' automatic barriers (see the `plgpu.kernel` call),
                # so nothing stops that copy from overwriting data some threads
                # are still reading. A warpgroup barrier per edge prevents it.
                # `commit_smem` is the Pallas call that emits one; the memory
                # fence it also emits does nothing here, since we wrote nothing.
                plgpu.commit_smem()
                return tuple(accumulator)

            # The running sum stays in registers for all of the node's edges,
            # as one `(channel_block,)` row per output coordinate carried by the
            # pipeline. Keeping it in shared memory instead would need a read
            # and a write for every update, which was 2.3x slower here.
            @pl.when(last_edge > first_edge)
            def _walk_edges():
                store(
                    plgpu.emit_pipeline(
                        add_edge,
                        grid=(last_edge - first_edge,),
                        in_specs=[
                            plgpu.BlockSpec(
                                (None, shapes.x_dim, channel_block),
                                # `gather_index` is already in walk order, so
                                # we read it at the step's position directly.
                                # The per-edge inputs below go through
                                # `edge_of` instead.
                                lambda step: (
                                    plgpu.load(gather_gmem.at[first_edge + step]),
                                    0,
                                    channel_block_idx,
                                ),
                            ),
                            plgpu.BlockSpec(
                                (None, shapes.y_dim_padded),
                                lambda step: (edge_of(step), 0),
                            ),
                            plgpu.BlockSpec(
                                (None, shapes.num_scalars, channel_block),
                                lambda step: (edge_of(step), 0, channel_block_idx),
                            ),
                        ],
                        max_concurrent_steps=config.pipeline_depth,
                        init_carry=tuple(zero_row() for _ in range(shapes.z_dim)),
                    )(x_gmem, y_gmem, s_gmem)
                )

            # A pipeline always runs at least one step, so nodes with no edges
            # skip it entirely. Otherwise that step would read whichever edge
            # comes next in the list, often a padding edge. Padding edges can
            # hold NaN or inf, and multiplying by a zero weight would not clear
            # them (`0 * NaN` is NaN).
            @pl.when(last_edge <= first_edge)
            def _no_edges():
                store([zero_row()] * shapes.z_dim)

        operands = [node_features, edge_features, edge_scalars, gather_index, offsets]
        if edge_perm is not None:
            operands.append(edge_perm)

        plgpu.kernel(
            kernel,
            out_type=(),  # the kernel writes to `out_gmem`, from `pl.run_state`
            # Pallas' automatic barriers add two barriers per store, and the
            # kernel stores one row per output coordinate for every node. The
            # only ordering we actually need comes from the `commit_smem` in
            # the edge body, so the automatic barriers are turned off.
            compiler_params=plgpu.CompilerParams(
                unsafe_no_auto_barriers=True,
                lowering_semantics=plgpu.LoweringSemantics.Lane,
            ),
            grid=(n_nodes,),
            grid_names=(NODE_AXIS,),
            kernel_name="convolution_mgpu_fwd",
        )(*operands)

    def launch(out_ref) -> None:
        for start in range(0, shapes.channels, channel_block):
            run_pass(out_ref, start)

    # `pl.run_state` turns each launch into a custom call that writes into the
    # buffer it was given (its output aliases that buffer). XLA's
    # dynamic-slice fusion ignores this alias: inside a `lax.map` or
    # `lax.scan`, it merges the loop's `ys` update into the call, the kernel's
    # writes are lost, and the loop returns zeros. The optimization barrier
    # stops XLA from doing that merge.
    return jax.lax.optimization_barrier(
        pl.run_state(launch)(pl.empty((n_nodes, shapes.z_dim, shapes.channels), dtype))
    )
