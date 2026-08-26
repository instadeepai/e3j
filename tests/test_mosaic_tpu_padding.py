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

"""Block ordering that keeps padded edges off the Mosaic TPU pipeline."""

import jax.numpy as jnp
import numpy
import pytest

from e3j.pallas_ops.convolution.mosaic_tpu.padding import block_order

BLOCK = 4


def _order(edge_is_real, n_blocks=4, block=BLOCK):
    order, walked = block_order(jnp.asarray(edge_is_real, bool), n_blocks, block)
    return numpy.asarray(order), int(walked)


class TestBlockOrder:
    """Blocks holding a real edge are walked first, the rest form the suffix the
    backward zeroes; both runs stay ascending so the sender walk keeps its order."""

    def test_all_real_walks_every_block(self):
        order, walked = _order([True] * 16)
        assert walked == 4
        numpy.testing.assert_array_equal(order, [0, 1, 2, 3])

    def test_all_padding_walks_nothing(self):
        order, walked = _order([False] * 16)
        assert walked == 0
        numpy.testing.assert_array_equal(sorted(order), [0, 1, 2, 3])

    def test_partial_block_is_walked(self):
        # one real edge is enough to keep its block on the pipeline
        order, walked = _order([False] * 4 + [True] + [False] * 11)
        assert walked == 1
        assert order[0] == 1

    def test_folded_graphs_skip_each_padding_run(self):
        # two graphs of 8 slots, each with a trailing padding run of 4
        edge_is_real = [True] * 4 + [False] * 4 + [True] * 4 + [False] * 4
        order, walked = _order(edge_is_real)
        assert walked == 2, "a scalar bound would stop past the last graph's padding"
        numpy.testing.assert_array_equal(order[:walked], [0, 2])
        numpy.testing.assert_array_equal(sorted(order[walked:]), [1, 3])

    @pytest.mark.parametrize("n_real", range(17))
    def test_walked_and_skipped_partition_the_blocks(self, n_real):
        order, walked = _order([True] * n_real + [False] * (16 - n_real))
        numpy.testing.assert_array_equal(sorted(order), range(4))
        assert walked == -(-n_real // BLOCK)
        assert list(order[:walked]) == sorted(order[:walked])
        assert list(order[walked:]) == sorted(order[walked:])
