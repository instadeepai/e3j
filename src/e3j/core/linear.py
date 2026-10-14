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
from typing import NamedTuple

import jax
import jax.numpy as jnp

from e3j.spaces import O3Space
from e3j.utils.options import Layout


class LinearBlock(NamedTuple):
    """Descriptor of one constant-(l, p) block of a linear map.

    ``begin``/``end`` slice the input feature axis; ``shape`` is the
    ``(m_out, m_in)`` multiplicity-mixing matrix shape.
    """

    begin: int
    end: int
    shape: tuple[int, int]
    l: int
    p: int


def iter_linear_blocks(
    source_irreps: str | O3Space,
    target_irreps: str | O3Space,
    channels: tuple[int, int] = (1, 1),
) -> list[LinearBlock]:
    """Return list of `LinearBlock` descriptors acting on irreducibles.

    Target blocks absent from source produce zero outputs. `channels =
    (c_in, c_out)` are folded into each block's `(m_out, m_in)` shape.
    """
    begin = 0
    linear_blocks = []
    source = O3Space(source_irreps)
    target = O3Space(target_irreps)
    c_in, c_out = channels
    # accumulate input slices, weight shapes, momenta and parities
    for (m_in, ir_in), (m_out, _) in _zip_source_target(source, target):
        size_in = m_in * (2 * ir_in.l + 1)
        block = LinearBlock(
            begin,
            begin + size_in,
            (m_out * c_out, m_in * c_in),
            ir_in.l,
            ir_in.p,
        )
        linear_blocks.append(block)
        begin += size_in
    return linear_blocks


def _zip_source_target(source: O3Space, target: O3Space):
    """Zip source and target together, accepting ordered mismatching irreps."""

    # Return zip(source, target) if momenta and parities match
    if len(source.blocks) == len(target.blocks) and all(
        ir_in == ir_out for (_, ir_in), (_, ir_out) in zip(source, target)
    ):
        return zip(source, target)

    # map (l, p) -> r with r <= 2 * l_max,
    # preserving natural irreps order (parity of l comes first)
    index = lambda l, p: 2 * l + (1 + (1 + p) // 2 + l % 2) % 2
    parity = lambda r: (-1) ** (r // 2 + r % 2)

    l_max = max(source.l_max, target.l_max)
    src = [(0, (r // 2, parity(r))) for r in range(2 * (l_max + 1))]
    tgt = [(0, (r // 2, parity(r))) for r in range(2 * (l_max + 1))]

    r_in = -1
    for m_in, ir_in in source:
        r = index(ir_in.l, ir_in.p)
        assert parity(r) == ir_in.p
        src[r] = (m_in, (ir_in.l, ir_in.p))
        if not r_in < r:
            raise ValueError(
                "Source and target irreps don't match, "
                "and are not strictly ordered (0e + 0o + 1o + 1e + 2e + ...)."
            )
        r_in = r

    r_out = -1
    for m_out, ir_out in target:
        r = index(ir_out.l, ir_out.p)
        assert parity(r) == ir_out.p
        tgt[r] = (m_out, (ir_out.l, ir_out.p))
        if not r_out < r:
            raise ValueError(
                "Source and target irreps don't match, "
                "and are not strictly ordered (0e + 0o + 1o + 1e + 2e + ...)."
            )
        r_out = r

    pairs = [
        (block_in, block_out)
        for block_in, block_out in zip(O3Space(src), O3Space(tgt))
        if (block_in[0] != 0 or block_out[0] != 0)
    ]
    for (_, ir_in), (_, ir_out) in pairs:
        if ir_in != ir_out:
            raise ValueError(
                f"Momenta and parities of zip(source, target) should match\n"
                f"\t got: {source} -> {target}"
            )
    return pairs


def slice_inputs(
    x: jax.Array, blocks: list[LinearBlock], layout: Layout
) -> list[jax.Array]:
    """Prepare list of inputs, sliced by irrep.

    Args:
        x (jax.Array): inputs with shape `(N, source.dim)` for `E3NN`, or
            `(N, num_channels, source.dim)` / `(N, source.dim, num_channels)`
            for `LEADING_CHANNELS` / `TRAILING_CHANNELS`.

    Returns:
        list[jax.Array]: One `(N, m_in * (2 * l_in + 1))`-array per block,
            channels folded into the trailing axis.
    """
    if layout == Layout.E3NN:
        return [x[:, block.begin : block.end] for block in blocks]
    elif layout == Layout.LEADING_CHANNELS:
        xs = [x[:, :, block.begin : block.end] for block in blocks]
        return [x.reshape((x.shape[0], -1)) for x in xs]
    elif layout == Layout.TRAILING_CHANNELS:
        xs = [x[:, block.begin : block.end, :] for block in blocks]
        return [x.reshape((x.shape[0], -1), order="F") for x in xs]
    raise ValueError(f"Unsupported layout {layout}")


def slice_transform(
    x_lp: jax.Array,
    w_lp: jax.Array,
    block: LinearBlock,
    batch_size: int,
    num_channels: int | None = None,
) -> jax.Array:
    """Mix multiplicities on a constant-(l, p) block via batched matmul.

    Features `x_lp` have shape `(N, [C,] m_in * (2l + 1))`, the `C` axis
    being present iff `num_channels is not None`.

    Note
    ----
    Apart from the normal case `m_in * m_out > 0`, two different edge cases
    should be considered which require `batch_size` to be passed explicitly:

    * if `m_out > 0` and `m_in == 0`, return zero vector with requested
      multiplicity (contrasts with e3nn).
    * if `m_out == 0` return empty vector to be discarded during final
      concatenation.

    It turns out a single branch is enough for both cases, although they
    are morally different.
    """
    has_channels = num_channels is not None
    nb, nc = batch_size, num_channels
    m_out, m_in = block.shape
    dim_l = 2 * block.l + 1

    shape_in = (nb, nc, m_in, dim_l) if has_channels else (nb, m_in, dim_l)
    shape_out = (nb, nc, m_out * dim_l) if has_channels else (nb, m_out * dim_l)

    if w_lp.size > 0:
        y_lp = w_lp @ x_lp.reshape(shape_in)
        return y_lp.reshape(shape_out)

    # return zero for padding (possibly empty)
    return jnp.zeros(shape_out, dtype=x_lp.dtype)


def check_weights(
    weights: list[jax.Array], shapes: list[tuple[int | None, ...]]
) -> None:
    """Check per-block `weights` against their expected `shapes`.

    A `None` entry of a shape matches any size.
    """
    if len(weights) != len(shapes):
        raise ValueError(
            f"Expected {len(shapes)} weights, one per linear block, "
            f"got {len(weights)}."
        )
    for i, (w, shape) in enumerate(zip(weights, shapes, strict=True)):
        if len(w.shape) != len(shape) or any(
            n is not None and n != m for n, m in zip(shape, w.shape)
        ):
            expected = tuple("_" if n is None else n for n in shape)
            raise ValueError(
                f"Weight {i} has shape {tuple(w.shape)}, expected "
                f"({', '.join(map(str, expected))})."
            )


def join_outputs(ys: list[jax.Array], layout: Layout, num_channels: int) -> jax.Array:
    """Concatenate per-block outputs, restoring the channel axis layout."""
    if layout == Layout.E3NN:
        return jnp.concat(ys, axis=-1)
    elif layout == Layout.LEADING_CHANNELS:
        ys = [y.reshape(y.shape[0], num_channels, -1) for y in ys]
        return jnp.concat(ys, axis=-1)
    elif layout == Layout.TRAILING_CHANNELS:
        ys = [y.reshape(y.shape[0], -1, num_channels, order="F") for y in ys]
        return jnp.concat(ys, axis=-2)
    raise ValueError(f"Unsupported layout {layout}")


def apply_linear(
    x: jax.Array,
    weights: list[jax.Array],
    blocks: list[LinearBlock],
    layout: Layout,
    num_channels_out: int,
) -> jax.Array:
    """Mix channels and multiplicities, one `(m_out, m_in)` weight per block."""
    check_weights(weights, [block.shape for block in blocks])
    nb = x.shape[0]
    xs = slice_inputs(x, blocks, layout)
    ys = [
        slice_transform(x_lp, w_lp, block, nb)
        for x_lp, w_lp, block in zip(xs, weights, blocks, strict=True)
    ]
    return join_outputs(ys, layout, num_channels_out)


class Linear:
    """Linear channel mixing, with weights passed explicitly.

    `channels = (c_in, c_out)` are folded into each block's `(m_out, m_in)`
    weight shape. Target blocks absent from source produce zero outputs, and
    source blocks absent from target are discarded.
    """

    def __init__(
        self,
        source_irreps: str | O3Space,
        target_irreps: str | O3Space,
        layout: str | Layout,
        channels: tuple[int, int] = (1, 1),
    ):
        self.source_irreps = source_irreps
        self.target_irreps = target_irreps
        self.layout = Layout.parse(layout)
        self.channels = channels

    @functools.cached_property
    def blocks(self) -> list[LinearBlock]:
        """Per-(l, p) blocks of the linear, in weight order."""
        return iter_linear_blocks(self.source_irreps, self.target_irreps, self.channels)

    def __call__(self, x: jax.Array, weights: list[jax.Array]) -> jax.Array:
        """Mix `x` with one `(m_out, m_in)` weight per block of :attr:`blocks`."""
        return apply_linear(x, weights, self.blocks, self.layout, self.channels[1])
