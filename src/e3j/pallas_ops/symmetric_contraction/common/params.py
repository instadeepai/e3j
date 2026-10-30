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

"""Backend-independent parameters of the fused symmetric contraction.

The symmetric contraction is

    powers   = [filter_1(a_1), filter_2(a_2), ..., filter_n(a_n)]
    x_sorted = concat(powers, lm)[sigma]
    out      = LinearIndexwise(x_sorted, index)

with ``a_1 = x`` and ``a_k = TP(a_{k-1}, x)``.
:class:`PallasSymmetricContractionStructure` stores this as static data: each
power's Clebsch-Gordan triples and, per output ``(l, p)`` block
(:class:`BlockPlan`), which power coordinate feeds each block input (the
filters, concatenation and ``sigma`` resolved once). The reduced plan
(:mod:`.plan`) is built from it, and each backend subclasses it to add its
kernel settings.

Only the ``TRAILING_CHANNELS`` layout ``(batch, lm, mul)`` and the ``MAP``
contraction mode are supported.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class BlockPlan(NamedTuple):
    """Where one output `(l, p)` block of the index-wise linear reads from.

    `input_map[mi * dim_l + li]` is the `(power, coordinate)` that input
    `(mi, li)` of the block reads: `power` 0 is the raw input `A`, power `k`
    is `A^(x)(k + 1)`, and `coordinate` indexes that unfiltered power. This
    replaces the concatenation and :class:`e3j.core.Permutation`.

    `offset` is the block's first coordinate in the kernel output, which
    holds every block side by side, `m_out * dim_l` coordinates each.
    """

    l: int
    p: int
    m_out: int
    input_map: tuple[tuple[int, int], ...]
    offset: int

    @property
    def dim_l(self) -> int:
        return 2 * self.l + 1

    @property
    def m_in(self) -> int:
        return len(self.input_map) // self.dim_l


@dataclass(eq=False)
class PallasSymmetricContractionStructure:
    """Static structure of one fused `SymmetricContraction` call.

    Hashable by content (:meth:`_key`, which backends extend).

    Attributes:
        stage_irreps: Each power's `(mul, l, p)` irreps in coordinate order
            (`stage_irreps[0]` is the raw input).
        stage_indices: For each power `k >= 1`, the `(nnz, 3)` `(zi, xi, yi)`
            Clebsch-Gordan triples of `A^(x)(k+1) = TP(A^(x)k, A)`: `zi`
            indexes power `k`, `xi` power `k - 1` and `yi` power 0.
        stage_values: The coefficients of `stage_indices`.
        blocks: One :class:`BlockPlan` per non-degenerate output block, in
            output order.
        num_species: Rows of every block's weight table.
        out_dim: Size of the kernel output. Coordinates outside every block
            (those of degenerate blocks) are zero (:attr:`zero_coordinates`).
    """

    stage_irreps: tuple[tuple[tuple[int, int, int], ...], ...]
    stage_indices: tuple[np.ndarray, ...]
    stage_values: tuple[np.ndarray, ...]
    blocks: tuple[BlockPlan, ...]
    num_species: int
    out_dim: int

    @property
    def correlation(self) -> int:
        """Number of powers, `A` through `A^(x)correlation`."""
        return len(self.stage_irreps)

    @property
    def stage_dims(self) -> tuple[int, ...]:
        """Size of each (unfiltered) power."""
        return tuple(
            sum(mul * (2 * l + 1) for mul, l, _ in irreps)
            for irreps in self.stage_irreps
        )

    @property
    def zero_coordinates(self) -> tuple[int, ...]:
        """Output coordinates no block writes (degenerate blocks)."""
        covered = {b.offset + k for b in self.blocks for k in range(b.m_out * b.dim_l)}
        return tuple(k for k in range(self.out_dim) if k not in covered)

    def __post_init__(self) -> None:
        if self.num_species < 1:
            raise ValueError(f"num_species must be >= 1, got {self.num_species}")
        spans = sorted(
            (b.offset, b.offset + b.m_out * b.dim_l, i)
            for i, b in enumerate(self.blocks)
        )
        for (_, end, i), (begin, _, j) in zip(spans, spans[1:]):
            if begin < end:
                raise ValueError(f"blocks[{i}] and blocks[{j}] overlap")
        if spans and (spans[0][0] < 0 or spans[-1][1] > self.out_dim):
            raise ValueError(f"the blocks do not fit in out_dim = {self.out_dim}")

    def _key(self) -> tuple:
        return (
            self.stage_irreps,
            tuple(i.tobytes() for i in self.stage_indices),
            tuple(v.tobytes() for v in self.stage_values),
            self.blocks,
            self.num_species,
            self.out_dim,
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())


def validate_operands(x1, index, weights, params: PallasSymmetricContractionStructure):
    """Check the fused kernels' operands against each other and `params`.

    Expects `x1` `(batch, stage_dims[0], channels)`, `index` `(batch,)`
    integer, and one `(num_species, channels, m_out, m_in)` weight per
    block, of `x1`'s dtype.

    Raises:
        ValueError: On any mismatch.
    """
    if x1.ndim != 3:
        raise ValueError(f"x1 must be (batch, x1_dim, channels), got {x1.shape}")
    batch, x1_dim, channels = x1.shape
    if x1_dim != params.stage_dims[0]:
        raise ValueError(
            f"x1 has coordinate width {x1_dim}, but stage_dims[0] = "
            f"{params.stage_dims[0]}"
        )
    if index.shape != (batch,):
        raise ValueError(f"index must be ({batch},), got {index.shape}")
    if not np.issubdtype(index.dtype, np.integer):
        raise ValueError(f"index must be an integer array, got {index.dtype}")
    if len(weights) != len(params.blocks):
        raise ValueError(
            f"expected {len(params.blocks)} block weights, got {len(weights)}"
        )
    for i, (w, block) in enumerate(zip(weights, params.blocks)):
        expected = (params.num_species, channels, block.m_out, block.m_in)
        if w.shape != expected:
            raise ValueError(f"weights[{i}] must be {expected}, got {w.shape}")
        if w.dtype != x1.dtype:
            raise ValueError(f"weights[{i}] is {w.dtype}, but x1 is {x1.dtype}")
