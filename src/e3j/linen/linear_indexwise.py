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
import jax.numpy as jnp
from jax.nn.initializers import Initializer

from e3j.linen.linear import _zip_source_target
from e3j.spaces.o3 import O3Space
from e3j.utils.config import config
from e3j.utils.options import Layout, LinearIndexwiseInitialization
from e3j.utils.weights_initialization import _get_weights_std_and_scaling

LinearBlock = namedtuple("LinearBlock", ["begin", "end", "shape", "l", "p"])


class LinearIndexwise(nn.Module):
    """E3-equivariant specie-wise linear mixing.

    Weights are learned for each of the `C` independent channels and
    `I` distinct indices, where `C` is inferred during `init`.

    Each linear block acts on a reducible representation of constant
    momentum `l` and parity `p`. This means that for every `(l, p)`
    pair, the module carries a weight array

        w_lp : (I, C, m_out, m_in)

    where `m_in` and `m_out` denote the multiplicities of `(l, p)`
    in the source and target representations respectively.

    Source and target irreps may have mismatching (l, p) blocks:
    target blocks absent from source produce zero outputs, and
    source blocks absent from target are discarded. This is useful
    e.g. for skip-connections from a first scalar-only layer in
    MLIPs, where the source space is ``"Kx0e"`` and the target
    includes higher-order irreps.

    Note
    ----
    Iteration over slices *may* be parallelised by the jit compiler,
    although it is written as a simple list comprehension. This is
    what `jax.tree.map` anyway ends up being lowered to, see:
    https://github.com/google/jax/issues/11394
    """

    source_irreps: str
    target_irreps: str
    num_indices: int
    num_channels: int | None = None
    layout: str | Layout = dataclasses.field(default_factory=lambda: config().layout)
    kernel_init: Initializer | LinearIndexwiseInitialization | str = "FAN_IN"
    rescale_gradients: bool = True

    @property
    def source(self):
        return e3nn.Irreps(self.source_irreps)

    @property
    def target(self):
        return e3nn.Irreps(self.target_irreps)

    def setup(self):
        """Initialize weights from list of `LinearBlock` descriptors."""
        channels = (self.num_channels,) if self.num_channels else ()
        weights = []
        for i, block in enumerate(self.blocks):
            if isinstance(self.kernel_init, Initializer):
                kernel_init, scale = self.kernel_init, 1.0
                assert (
                    not self.rescale_gradients
                ), "`rescale_gradients` requires LinearIndexwiseInitialization option."
            else:
                std, scale = _get_weights_std_and_scaling(
                    m_out=block.shape[0],
                    m_in=block.shape[1],
                    weights_normalization=self.kernel_init,
                    rescale_gradients=self.rescale_gradients,
                    num_indices=self.num_indices,
                )
                kernel_init = nn.initializers.normal(stddev=std)
            wi = self.param(
                f"weight{i}_{block.l}_{block.p}",
                kernel_init,
                (self.num_indices, *channels, *block.shape),
            )
            weights.append(wi * scale)
        self.weights = weights

    @functools.cached_property
    def blocks(self) -> list[LinearBlock]:
        """Return list of `LinearBlock` descriptors acting on irreducibles.

        Source and target may have mismatching (l, p) blocks.
        Target blocks absent from source produce zero outputs,
        which is useful e.g. for skip-connections from a first
        scalar-only layer in MLIPs.
        """
        begin = 0
        linear_blocks = []
        source = O3Space(self.source)
        target = O3Space(self.target)
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
                (m_out, m_in),
                ir_in.l,
                ir_in.p,
            )
            linear_blocks.append(block)
            begin += size_in
        return linear_blocks

    def slice_inputs(self, x: jnp.ndarray) -> list[jnp.ndarray]:
        """Prepare PyTree of inputs, sliced by irrep.

        Returns a list of arrays with shape ``(..., C, m_in * (2l+1))``
        (leading channels) regardless of input layout. For trailing
        channels, axis swap is applied after slicing.
        """
        layout = Layout.parse(self.layout)
        if layout == Layout.TRAILING_CHANNELS:
            xs = [x[..., block.begin : block.end, :] for block in self.blocks]
            return [jnp.swapaxes(x, -1, -2) for x in xs]
        return [x[..., block.begin : block.end] for block in self.blocks]

    def slice_transform(
        self,
        x_lp: jnp.ndarray,
        w_lp: jnp.ndarray,
        block: LinearBlock,
        batch_size: int,
        num_channels: int | None = None,
    ) -> jnp.ndarray:
        """Mix irreps on a constant-(l,p) block.

        The mixing of multiplicities is performed with `np.matmul` between:

        * `w_lp : (N, C, m_out, m_in)`, the array of specie-dependent weights,
        * `x_lp : (N, C, m_in * (2l + 1))`, the array of equivariant features,

        where `N` is the batch size and `C` the number of independent channels.

        Note
        ----
        Apart from the normal case `m_in * m_out > 0`, two different
        edge cases should be considered:

        * if `m_out > 0` and `m_in == 0`,
            return zero vector with requested multiplicity (in contrast with e3nn).
        * if `m_out == 0`,
            return empty vector to be discarded during concatenation.

        It turns out a single branch is enough for both cases, although they
        are morally different. They require `batch_size` and `num_channels` to be
        passed explicitly.
        """
        has_channels = num_channels is not None
        nb, nc = batch_size, num_channels
        m_out, m_in = block.shape
        l = block.l
        dim_l = 2 * l + 1

        shape_in = (nb, nc, m_in, dim_l) if has_channels else (nb, m_in, dim_l)
        shape_out = (nb, nc, m_out * dim_l) if has_channels else (nb, m_out * dim_l)

        if w_lp.size > 0:
            y_lp = w_lp @ x_lp.reshape(shape_in)
            return y_lp.reshape(shape_out)

        # return zero for padding (possibly empty)
        return jnp.zeros(shape_out, dtype=x_lp.dtype)

    def slice_weights(self, indices: jnp.ndarray) -> list[jnp.ndarray]:
        """Return specie-wise weights acting on `x_feats`.

        The learnable weights arrays, each of shape `(I, C, m_in, m_out)`, are
        pulled by the length-N integer vector `x_indices < I` to produce batches
        of weight arrays `(N, C, m_in, m_out)`.

        The I/O multiplicities depend both on momentum `l` and parity `p`.
        """
        return [w_lp[indices] for w_lp in self.weights]

    def join_outputs(self, ys: list[jnp.ndarray]) -> jnp.ndarray:
        """Concatenate outputs, restoring the channel axis layout."""
        layout = Layout.parse(self.layout)
        if layout == Layout.TRAILING_CHANNELS:
            ys = [jnp.swapaxes(y, -1, -2) for y in ys]
            return jnp.concat(ys, axis=-2)
        return jnp.concat(ys, axis=-1)

    def __call__(self, x_feats: jnp.ndarray, x_indices: jnp.ndarray) -> jnp.ndarray:
        """Transform `x_feats` linearly with `x_indices`-dependent weights.

        Args:
            x_feats (jnp.ndarray): `(N, C, source.dim)`-array of equivariant features
                (leading channels) or `(N, source.dim, C)` (trailing channels).
            x_indices (jnp.ndarray): `N`-vector of specie indices (positive and
                lower than `I = num_indices`)

        Returns:
            jnp.ndarray: Array of shape `(N, C, target.dim)` or
                `(N, target.dim, C)` depending on layout.
        """
        # slice weights and inputs (l, p)-wise
        xs = self.slice_inputs(x_feats)
        ws = self.slice_weights(x_indices)
        nb = x_feats.shape[0]
        nc = self.num_channels
        ys = [
            self.slice_transform(x_lp, w_lp, block, nb, nc)
            for x_lp, w_lp, block in zip(xs, ws, self.blocks)
        ]
        return self.join_outputs(ys)
