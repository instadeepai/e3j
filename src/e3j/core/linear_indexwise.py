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

import functools
import math

import jax
import jax.numpy as jnp

from e3j.core.linear import (
    LinearBlock,
    check_weights,
    iter_linear_blocks,
    slice_transform,
)
from e3j.spaces import O3Space
from e3j.utils.options import Layout


def slice_inputs(
    x: jax.Array, blocks: list[LinearBlock], layout: Layout
) -> list[jax.Array]:
    """Slice features by irrep block, returning leading-channels arrays."""
    if layout == Layout.TRAILING_CHANNELS:
        xs = [x[..., block.begin : block.end, :] for block in blocks]
        return [jnp.swapaxes(x, -1, -2) for x in xs]
    return [x[..., block.begin : block.end] for block in blocks]


def join_outputs(ys: list[jax.Array], layout: Layout) -> jax.Array:
    """Concatenate per-block outputs, restoring the channel axis layout."""
    if layout == Layout.TRAILING_CHANNELS:
        ys = [jnp.swapaxes(y, -1, -2) for y in ys]
        return jnp.concat(ys, axis=-2)
    return jnp.concat(ys, axis=-1)


def apply_indexwise_linear(
    x_feats: jax.Array,
    weights: list[jax.Array],
    index: jax.Array,
    blocks: list[LinearBlock],
    layout: Layout,
    num_channels: int | None = None,
) -> jax.Array:
    """Mix multiplicities with weights gathered by `index`.

    See :meth:`LinearIndexwise.__call__` for shapes.
    """
    channels = () if num_channels is None else (num_channels,)
    check_weights(weights, [(None, *channels, *block.shape) for block in blocks])
    # flatten batch axes
    feat_shape = x_feats.shape[x_feats.ndim - 1 - len(channels) :]
    batch_shape = x_feats.shape[: x_feats.ndim - len(feat_shape)]
    if index.shape != batch_shape:
        raise ValueError(
            f"Indices of shape {index.shape} do not match the batch axes "
            f"{batch_shape} of features of shape {x_feats.shape}."
        )
    nb = math.prod(batch_shape)
    x_feats = x_feats.reshape(nb, *feat_shape)
    index = index.reshape(nb)
    xs = slice_inputs(x_feats, blocks, layout)
    ws = [w_lp[index] for w_lp in weights]
    ys = [
        slice_transform(x_lp, w_lp, block, nb, num_channels)
        for x_lp, w_lp, block in zip(xs, ws, blocks, strict=True)
    ]
    y = join_outputs(ys, layout)
    return y.reshape(*batch_shape, *y.shape[1:])


class LinearIndexwise:
    """Index-wise linear mixing, with weights passed explicitly.

    Target blocks absent from source produce zero outputs, and source blocks
    absent from target are discarded.
    """

    def __init__(
        self,
        source_irreps: str | O3Space,
        target_irreps: str | O3Space,
        layout: str | Layout,
        num_channels: int | None = None,
    ):
        self.source_irreps = source_irreps
        self.target_irreps = target_irreps
        self.layout = Layout.parse(layout)
        self.num_channels = num_channels

    @functools.cached_property
    def blocks(self) -> list[LinearBlock]:
        """Per-(l, p) blocks of the linear, in weight order."""
        return iter_linear_blocks(self.source_irreps, self.target_irreps)

    def __call__(
        self,
        feat: jax.Array,
        index: jax.Array,
        weights: list[jax.Array],
    ) -> jax.Array:
        """Mix `feat` with weights gathered by `index`.

        Args:
            feat (jax.Array): `(..., C, dim)` or `(..., dim, C)` features
                depending on :attr:`layout`, `(..., dim)` without channels.
            index (jax.Array): `(...)`-array of specie indices.
            weights (list[jax.Array]): one `(I, [C,] m_out, m_in)` array per
                block of :attr:`blocks`.
        """
        return apply_indexwise_linear(
            feat, weights, index, self.blocks, self.layout, self.num_channels
        )
