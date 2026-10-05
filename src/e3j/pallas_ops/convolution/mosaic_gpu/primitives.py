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

"""Public entry point for the Mosaic GPU message-passing convolution.

Wraps the forward and backward kernels so the convolution works with `jax.grad`
(to any order) and with `jax.vmap`.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from e3j.data.graph import is_dummy_index
from e3j.pallas_ops.convolution.mosaic_gpu.bwd import _convolution_kernel_mosaic_gpu_bwd
from e3j.pallas_ops.convolution.mosaic_gpu.fwd import _convolution_kernel_mosaic_gpu_fwd
from e3j.pallas_ops.convolution.mosaic_gpu.params import (
    PallasMosaicGPUMessagePassingConvolutionParams,
)
from e3j.utils import options


def convolution_mosaic_gpu(
    node_features: jax.Array,
    edge_features: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    params: PallasMosaicGPUMessagePassingConvolutionParams,
) -> jax.Array:
    """Run the fused message-passing convolution with the Mosaic GPU kernels.

    You can differentiate the result to any order with respect to
    `node_features`, `edge_features` and `edge_scalars`. The indices get no
    gradient.

    The edge list must be sorted by the endpoint named in
    `params.graph_ordering`. When the forward groups edges by `SENDER`, it
    relies on the graph symmetries described in :class:`e3j.core.Convolution`.
    If your graph does not have them, set `fwd_config.grouping=RECEIVER`.

    Args:
        node_features: `(n_nodes, x_dim, channels)`.
        edge_features: `(n_edges, y_dim)`.
        edge_scalars: `(n_edges, num_scalars, channels)`.
        senders: `(n_edges,)` sender node of each edge.
        receivers: `(n_edges,)` receiver node of each edge.
        params: Kernel parameters.

    Returns:
        `(n_nodes, z_dim, channels)` messages summed on each receiver.
    """

    @jax.custom_vjp
    def convolution(node_features, edge_features, edge_scalars, senders, receivers):
        return _fwd_vmappable(
            node_features, edge_features, edge_scalars, senders, receivers, params
        )

    def convolution_fwd(node_features, edge_features, edge_scalars, senders, receivers):
        # The forward kernel saves no intermediate results. The backward
        # recomputes the tensor product from the inputs, so we only keep those.
        residuals = (node_features, edge_features, edge_scalars, senders, receivers)
        return convolution(*residuals), residuals

    def convolution_bwd(residuals, cotangent):
        node_features, edge_features, edge_scalars, senders, receivers = residuals
        dx, dy, ds = _convolution_vjp(
            node_features,
            edge_features,
            edge_scalars,
            senders,
            receivers,
            cotangent,
            params,
        )
        return dx, dy, ds, _index_cotangent(senders), _index_cotangent(receivers)

    convolution.defvjp(convolution_fwd, convolution_bwd)
    return convolution(node_features, edge_features, edge_scalars, senders, receivers)


def _convolution_vjp(
    node_features: jax.Array,
    edge_features: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    cotangent: jax.Array,
    params: PallasMosaicGPUMessagePassingConvolutionParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the gradients `(dx, dy, ds)` of the convolution.

    This function has its own custom VJP, so the gradients can themselves be
    differentiated, to any order.
    """

    @jax.custom_vjp
    def vjp(x, y, s, senders, receivers, ct):
        return _bwd_vmappable(x, y, s, senders, receivers, ct, params)

    def vjp_fwd(*operands):
        return vjp(*operands), operands

    def vjp_bwd(residuals, cotangents):
        # `(dx, dy, ds)` is linear in `ct`, and linear in each of `x`, `y` and
        # `s` taken separately. So `<ddx, dx> + <ddy, dy> + <dds, ds>` is
        # equal to `<ct, conv(ddx, y, s) + conv(x, ddy, s) + conv(x, y, dds)>`,
        # and we can get the second-order terms by differentiating that
        # expression with the kernels we already have.
        x, y, s, senders, receivers, ct = residuals
        ddx, ddy, dds = cotangents

        def conv(x, y, s, params):
            return convolution_mosaic_gpu(x, y, s, senders, receivers, params)

        def bwd(x, y, s, params):
            return _convolution_vjp(x, y, s, senders, receivers, ct, params)

        # A sender-grouped forward assumes its edge operands are symmetric when
        # an edge is reversed. `ddy` and `dds` are arbitrary cotangents and do
        # not have to be, so the calls that take them use receiver grouping.
        exact = _receiver_grouped_fwd(params)
        dct = conv(ddx, y, s, params) + conv(x, ddy, s, exact) + conv(x, y, dds, exact)
        _, dy_x, ds_x = bwd(ddx, y, s, params)
        dx_y, _, ds_y = bwd(x, ddy, s, exact)
        dx_s, dy_s, _ = bwd(x, y, dds, exact)
        return (
            dx_y + dx_s,
            dy_x + dy_s,
            ds_x + ds_y,
            _index_cotangent(senders),
            _index_cotangent(receivers),
            dct,
        )

    vjp.defvjp(vjp_fwd, vjp_bwd)
    return vjp(
        node_features, edge_features, edge_scalars, senders, receivers, cotangent
    )


def _receiver_grouped_fwd(
    params: PallasMosaicGPUMessagePassingConvolutionParams,
) -> PallasMosaicGPUMessagePassingConvolutionParams:
    """Return a copy of `params` whose forward groups edges by receiver."""
    if params.fwd_grouping == options.GraphOrdering.RECEIVER:
        return params
    fwd_config = params.fwd_config.model_copy(
        update={"grouping": options.GraphOrdering.RECEIVER}
    )
    return dataclasses.replace(params, fwd_config=fwd_config)


def _index_cotangent(index: jax.Array) -> np.ndarray:
    """Zero cotangent for an integer index array (indices get no gradient)."""
    return np.zeros(index.shape, jax.dtypes.float0)


# --------------------------------------------------------------- vmap ---


def _broadcast_unbatched(axis_size: int, in_batched, operands) -> list[jax.Array]:
    """Give every operand the vmapped axis, repeating the ones that lack it."""
    return [
        a if batched else jnp.broadcast_to(a[None], (axis_size,) + a.shape)
        for a, batched in zip(operands, in_batched)
    ]


def _merge_leading(a: jax.Array) -> jax.Array:
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


def _split_leading(a: jax.Array, axis_size: int) -> jax.Array:
    return a.reshape((axis_size, a.shape[0] // axis_size) + a.shape[1:])


def _fold_index(index: jax.Array, n_nodes: int) -> jax.Array:
    """Turn `(V, n_edges)` node indices into indices of one big graph.

    Graph `v` gets its node indices shifted by `v * n_nodes`, so the `V`
    graphs become disjoint parts of a single graph. Dummy indices are left
    as they are. Because of that, dummy edges can now appear in the middle
    of the edge list, so the folded call cannot assume the edges are sorted.
    """
    offsets = jnp.arange(index.shape[0], dtype=index.dtype)[:, None] * n_nodes
    return jnp.where(is_dummy_index(index), index, index + offsets).reshape(-1)


def _fwd_vmappable(
    node_features: jax.Array,
    edge_features: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    params: PallasMosaicGPUMessagePassingConvolutionParams,
    edges_sorted: bool = True,
) -> jax.Array:
    """Call the forward kernel, with a vmap rule that batches by merging graphs.

    Under `vmap`, the batch of graphs is merged into one larger graph (see
    :func:`_fold_index`) and the kernel is launched once on it.
    """

    @jax.custom_batching.custom_vmap
    def _impl(x, y, s, senders, receivers):
        return _convolution_kernel_mosaic_gpu_fwd(
            x, y, s, senders, receivers, params, edges_sorted=edges_sorted
        )

    @_impl.def_vmap
    def _rule(axis_size, in_batched, *operands):
        x, y, s, senders, receivers = _broadcast_unbatched(
            axis_size, in_batched, operands
        )
        n_nodes = x.shape[1]
        # Call `_fwd_vmappable` again rather than the kernel directly, so that
        # an outer `vmap` can merge its axis the same way.
        out = _fwd_vmappable(
            _merge_leading(x),
            _merge_leading(y),
            _merge_leading(s),
            _fold_index(senders, n_nodes),
            _fold_index(receivers, n_nodes),
            params,
            edges_sorted=False,
        )
        return _split_leading(out, axis_size), True

    return _impl(node_features, edge_features, edge_scalars, senders, receivers)


def _bwd_vmappable(
    node_features: jax.Array,
    edge_features: jax.Array,
    edge_scalars: jax.Array,
    senders: jax.Array,
    receivers: jax.Array,
    cotangent: jax.Array,
    params: PallasMosaicGPUMessagePassingConvolutionParams,
    edges_sorted: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Call the backward kernel, with the same vmap rule as `_fwd_vmappable`."""

    @jax.custom_batching.custom_vmap
    def _impl(x, y, s, senders, receivers, ct):
        return _convolution_kernel_mosaic_gpu_bwd(
            x, y, s, senders, receivers, ct, params, edges_sorted=edges_sorted
        )

    @_impl.def_vmap
    def _rule(axis_size, in_batched, *operands):
        x, y, s, senders, receivers, ct = _broadcast_unbatched(
            axis_size, in_batched, operands
        )
        n_nodes = x.shape[1]
        cotangents = _bwd_vmappable(
            _merge_leading(x),
            _merge_leading(y),
            _merge_leading(s),
            _fold_index(senders, n_nodes),
            _fold_index(receivers, n_nodes),
            _merge_leading(ct),
            params,
            edges_sorted=False,
        )
        return tuple(_split_leading(c, axis_size) for c in cotangents), (True,) * 3

    return _impl(
        node_features, edge_features, edge_scalars, senders, receivers, cotangent
    )
