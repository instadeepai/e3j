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

import dataclasses
import functools
from collections import namedtuple

import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as np
from jax import Array
from jax.nn.initializers import Initializer

from e3j.spaces.o3 import O3Space
from e3j.utils.config import config
from e3j.utils.options import Layout, LinearInitialization
from e3j.utils.weights_initialization import _get_weights_std_and_scaling

LinearBlock = namedtuple("LinearBlock", ["begin", "end", "shape", "l", "p"])


class Linear(nn.Module):
    """E3-equivariant linear channel mixing.

    The `Linear` module mixes equivariant channels with same angular momentum `l` by:

    * iterating over input slices indexed by `l`,
    * reshaping each slice `x_l` from shape `(-1, (2l+1) *m_in)`
      to shape `(-1, m_in)`,
    * linearly transforming slices as `y_l = x_l @ weights_l`,
    * reshaping output slices `y_l` from shape `(-1, m_out)`
      to shape `(-1, (2l+1) * m_out)`,
    * concatenating output slices on axis -1.

    Iteration over slices is not parallelised by `jax.tree.map`.
    """

    source_irreps: str
    target_irreps: str
    channels: tuple[int, int] = (1, 1)
    layout: str | Layout = dataclasses.field(default_factory=lambda: config().layout)
    kernel_init: Initializer | LinearInitialization | str = "FAN_IN"
    rescale_gradients: bool = True

    @property
    def source(self):
        return O3Space(self.source_irreps)

    @property
    def target(self):
        return O3Space(self.target_irreps)

    @functools.cached_property
    def blocks(self) -> list[LinearBlock]:
        begin = 0
        linear_blocks = []
        source = O3Space(self.source)
        target = O3Space(self.target)
        c_in, c_out = self.channels
        # accumulate input slices, weight shapes, momenta and parities
        for (m_in, ir_in), (m_out, ir_out) in _zip_source_target(source, target):
            if not (ir_in.l == ir_out.l and ir_in.p == ir_out.p):
                raise ValueError(
                    f"Momenta and parities of zip(source, target) should match\n"
                    f"\t got: {source} -> {target}"
                )
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

    def slice_inputs(self, x: Array) -> list[Array]:
        """Prepare PyTree of inputs, sliced by irrep.

        Args:
            x (Array): inputs with shape (-1, num_channels, source.dim)

        Returns:
            list[Array]: List of 2*(l_max+1) ndarrays of shape
                              (-1, num_channels, m_in*(2*l_in+1)).
        """
        layout = Layout.parse(self.layout)
        if layout == Layout.E3NN:
            return [x[:, block.begin : block.end] for block in self.blocks]
        elif layout == Layout.LEADING_CHANNELS:
            xs = [x[:, :, block.begin : block.end] for block in self.blocks]
            return [x.reshape((x.shape[0], -1)) for x in xs]
        elif layout == Layout.TRAILING_CHANNELS:
            xs = [x[:, block.begin : block.end, :] for block in self.blocks]
            return [x.reshape((x.shape[0], -1), order="F") for x in xs]
        raise ValueError(f"Unsupported layout {layout}")

    def slice_transform(
        self,
        x_lp: Array,
        w_lp: Array,
        block: LinearBlock,
        batch_size: int,
    ) -> Array:
        """Mix irreps on a constant-(l,p) block.

        The mixing of multiplicities is performed with `np.matmul` between:

        * `w_lp : (N, m_out, m_in)`, the array of specie-dependent weights,
        * `x_lp : (N, m_in * (2l + 1))`, the array of equivariant features,

        where `N` is the batch size.

        Note
        ----
        Apart from the normal case `m_in * m_out > 0`, two different
        edge cases should be considered which require `batch_size` to
        be passed explicitly:

        * if `m_out > 0` and `m_in == 0`, return zero vector with requested
          multiplicity (contrasts with e3nn).
        * if `m_out == 0` return empty vector to be discarded during final
          concatenation.

        It turns out a single branch is enough for both cases, although they
        are morally different.
        """
        nb = batch_size
        m_out, m_in = block.shape
        l = block.l
        dim_l = 2 * l + 1
        if w_lp.size > 0:
            y_lp = w_lp @ x_lp.reshape(nb, m_in, dim_l)
            return y_lp.reshape(nb, m_out * dim_l)
        # return zero for padding (possibly empty)
        return np.zeros((nb, m_out * dim_l), dtype=x_lp.dtype)

    def slice_weights(self) -> list[Array]:
        """Return arrays of weights acting on all degrees."""
        weights = []
        for i, block in enumerate(self.blocks):
            if isinstance(self.kernel_init, Initializer):
                kernel_init, scale = self.kernel_init, 1.0
                assert (
                    not self.rescale_gradients
                ), "`rescale_gradients` requires LinearInitialization option."
            else:
                std, scale = _get_weights_std_and_scaling(
                    m_out=block.shape[0],
                    m_in=block.shape[1],
                    weights_normalization=self.kernel_init,
                    rescale_gradients=self.rescale_gradients,
                )
                kernel_init = nn.initializers.normal(std)

            wi = self.param(f"weight{i}_{block.l}_{block.p}", kernel_init, block.shape)
            weights.append(wi * scale)

        return weights

    def join_outputs(self, ys: list[Array]) -> Array:
        """Concatenate outputs, eventually restoring the channel axis."""
        layout = Layout.parse(self.layout)
        c_out = self.channels[1]
        if layout == Layout.E3NN:
            return np.concat(ys, axis=-1)
        elif layout == Layout.LEADING_CHANNELS:
            ys = [y.reshape(y.shape[0], c_out, -1) for y in ys]
            return np.concat(ys, axis=-1)
        elif layout == Layout.TRAILING_CHANNELS:
            ys = [y.reshape(y.shape[0], -1, c_out, order="F") for y in ys]
            return np.concat(ys, axis=-2)
        raise ValueError("Unsupported layout")

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Transform equivariant tensors linearly."""
        # infer batch size and number of channels
        nb = x.shape[0]
        # slice weights and inputs (l, p)-wise
        xs = self.slice_inputs(x)
        ws = self.slice_weights()
        ys = [
            self.slice_transform(x_lp, w_lp, block, nb)
            for x_lp, w_lp, block in zip(xs, ws, self.blocks)
        ]
        return self.join_outputs(ys)


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

    return (
        (r_in, r_out)
        for r_in, r_out in zip(e3nn.Irreps(src), e3nn.Irreps(tgt))
        if (r_in.mul != 0 or r_out.mul != 0)
    )
