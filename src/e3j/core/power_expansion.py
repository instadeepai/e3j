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

import e3nn_jax as e3nn
import jax.numpy as np

from e3j.core.filter import Filter
from e3j.core.tensor_product import TensorProduct
from e3j.spaces import O3Space
from e3j.utils.irreps import irrep_range
from e3j.utils.options import Layout, TPMode


class PowerExpansion:
    """Direct sum of equivariant powers.

    This module computes equivariant powers iteratively using
    :class:`e3j.TensorProduct` at each step, returned in a list
    of arrays.
    """

    def __init__(
        self,
        source: str | O3Space,
        exponent: int,
        target_filter: None | str | O3Space = None,
        l_max: int | None = None,
        layout: str | Layout = "TRAILING_CHANNELS",
    ):
        """Initialize power expansion block.

        Irreducible blocks can be pruned from the output either by
        specifying a `target_filter` argument, or by using a global
        threshold `l_max` on all intermediate products.

        Args:
            source: the input space
            exponent: maximal power in the development, length of output.
            target_filter: irreducible blocks not in this filter will not
                           be computed.
            l_max: any intermediate irreducible block over this degree will
                   be skipped.
            layout: I/O array layout.
        """
        if not isinstance(source, O3Space):
            source = O3Space(source)
        if not isinstance(target_filter, (O3Space, type(None))):
            target_filter = O3Space(target_filter)

        self.source = source
        self.exponent = exponent
        self.target_filter = target_filter
        self.l_max = l_max
        self.layout = Layout.parse(layout)

    @property
    def hidden(self) -> tuple[O3Space, ...]:
        """Unfiltered representation of each intermediate power.

        Because the product of `l1` and `l2` can produce `l >= |l1 - l2|`,
        hidden (intermediate) features may very well contain high momenta
        that contribute to one of the lower-degree representations of the
        output filter.

        To exclude hidden representations based on a momentum threshold,
        set the `l_max` attribute. This parameter allows for a trade-off
        between expressivity vs. speed and memory.
        """
        layers = self.otimes_layers()
        return (self.source, *(otimes.target for otimes in layers))

    @property
    def target(self) -> tuple[O3Space, ...]:
        return tuple(layer.target for layer in self.filter_layers())

    def get_target_filter(self, power: int) -> str | O3Space | None:
        """Return filter on irreducible blocks at given step."""
        if power == self.exponent and self.target_filter is not None:
            return self.target_filter
        if power == self.exponent and self.target_filter is None:
            return irrep_range(self.source.l_max * self.exponent, True)
        if self.target_filter is not None:
            # l_max - p * l_max_in <= l_max_out
            l_max_out = O3Space(self.target_filter).l_max
            l_max_in = self.source.l_max
            l_max = l_max_out + l_max_in * (self.exponent - power)
            if self.l_max is not None:
                l_max = min(l_max, self.l_max)
            return irrep_range(l_max, True)
        return None

    @functools.cache
    def filter_layers(self) -> list[Filter]:
        """Return list of :class:`e3j.core.Filter` instances.

        Because intermediate powers may span representations outside of the
        `target_irreps` filter, each exponent order carries a `Filter` module
        responsible for filtering the hidden features.
        """
        layout = Layout.parse(self.layout)
        if layout == Layout.TRAILING_CHANNELS:
            axis_lm = -2
        else:
            axis_lm = -1
        sources = self.hidden
        # last output, coincides with target_irreps if not None
        target = self.get_target_filter(self.exponent)
        return [Filter(src, target, axis_lm) for src in sources]

    @functools.cache
    def otimes_layers(self) -> list[TensorProduct]:
        """Return list of :class:`e3j.core.TensorProduct` instances."""
        layers = []
        irrep_nu = self.source
        for nu in range(2, self.exponent + 1):
            # Filter output
            irreps_out = self.get_target_filter(nu)
            # TODO: support MAP mode with leading channels too, which allows
            #       to unroll over the batch axis.
            layout = Layout.parse(self.layout)
            mode = TPMode.MAP if layout == Layout.TRAILING_CHANNELS else TPMode.OUTER
            # Accumulate TP layer
            otimes = TensorProduct(
                (irrep_nu, self.source),
                irreps_out,
                layout=self.layout,
                mode=mode,
            )
            irrep_nu = otimes.target
            layers.append(otimes)
            lhs, rhs = otimes.source

        return layers

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        """Compute list of (filter-projected) equivariant powers [x, x2, ...]."""
        out = [x]
        for otimes in self.otimes_layers():
            x_nu = out[-1]
            x_nu = otimes(x_nu, x)
            out.append(x_nu)
        filters = self.filter_layers()
        return [filter_out(x_nu) for x_nu, filter_out in zip(out, filters)]
