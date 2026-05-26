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

import itertools
import re
from dataclasses import dataclass
from typing import Iterable, Self, TypeAlias, overload

import e3nn_jax as e3nn
import jax

from e3j.spaces.space import Irrep, IrrepSum

_SO3_IRREP_REGEX = r"^(\d*)$"
_OPLUS_REGEX = r"\s*\+\s*"
_MUL_REGEX = r"^(\d*)\s*x\s*(.*)"


@dataclass
class SO3Irrep(Irrep):
    """Irreducible representation of SO3.

    Minimal SO3-spaces are characterized by a degree `l >= 0` which
    is the degree of the 2l+1 generating harmonic polynomials of
    `SO3Irrep(l)`.
    """

    l: int

    @overload
    def __init__(self, irreps: str): ...
    @overload
    def __init__(self, irreps: "SO3Irrep"): ...
    @overload
    def __init__(self, l: int): ...

    def __init__(self, *args):
        """Parse SO3Irrep string and validate inputs."""
        # Parse string
        assert len(args) == 1, "SO3Irrep should be initialized with a single argument"
        if isinstance(args[0], str):
            parsed = re.findall(_SO3_IRREP_REGEX, args[0])
            if not len(parsed):
                raise ValueError(f"Could not parse SO3Irrep string {args[0]} as str(l)")
            l, *_ = parsed[0]
            self.l = int(l)
        elif isinstance(args[0], SO3Irrep):
            self.l = args[0].l
        elif isinstance(args[0], int):
            self.l = args[0]
        else:
            raise ValueError(
                f"SO3Irrep expects int (l), str or SO3Irrep as first argument, got {type(args[0])}"
            )

        if self.l < 0:
            raise ValueError("Degree `l` must be positive")

    @property
    def dim(self) -> int:
        """Dimension of the irreducible block: 2l + 1."""
        return 2 * self.l + 1

    def __iter__(self):
        yield self.l

    def __rmul__(self, other: int) -> "IrrepSum":
        return SO3Space([(other, self)])

    def __mul__(self, other: "SO3Irrep") -> "IrrepSum":
        if isinstance(self, SO3Irrep):
            l_min = abs(self.l - other.l)
            l_max = self.l + other.l
            return SO3Space((1, SO3Irrep(l)) for l in range(l_min, l_max + 1))
        else:
            # We could support Space * Space too
            raise NotImplementedError("SO3Irrep can only be multiplied with SO3Irrep")

    def __str__(self) -> str:
        return str(self.l)

    def __lt__(self, other) -> bool:
        return self.l < other.l

    def __eq__(self, other) -> bool:
        return self.l == other.l

    def __hash__(self) -> int:
        return hash(str(self))


SO3Block: TypeAlias = tuple[int, SO3Irrep | int]


class SO3Space(IrrepSum[SO3Irrep]):
    """Representation of SO3, direct sum of irreducible blocks.

    SO3-Spaces yield pairs `(mul, irrep)` when iterated through, where `mul`
    denotes the multiplicity of the irreducible subspace `irrep`.
    Each `irrep` may also be given as a single integer `l`.

    In e3nn-like notation, subspaces are separated with `"+"` and multiplicities
    act on irreducibles with `"x"`, e.g.

        >>> space = SO3Space("16x0e + 8x1e + 8x2e")
        >>> space = SO3Space("16x0 + 8x1 + 8x2")
        >>> space = SO3Space([(16, 0), (8, 1), (8, 2)])
    """

    def __init__(self, blocks: str | Iterable[SO3Block]):
        """Parse blocks and layout.

        There are multiple ways a `Space` instance can be described and initialized:

        * as a string of the form `"8x0e + 4x1o"`,
        * as a list of length-2 or length-3 tuples of the form
            * `[(8, O3Irrep("0e"), (4, O3Irrep("1o"))]`
            * `[(8, "0e"), (8, "1o")]`
            * `[(8, 0, 1), (4, 1, -1)]`
        """
        self.blocks: list[tuple[int, SO3Irrep]] = []
        if isinstance(blocks, str):
            self.blocks = self._parse_string(blocks)
        elif isinstance(blocks, e3nn.Irreps):
            self.blocks = self._parse_e3nn(blocks)
        else:
            self.blocks = self._parse_list(blocks)

    @property
    def l_max(self) -> int:
        return max(ir.l for _, ir in self)

    def __mul__(self, other) -> Self:
        """Tensor product representation."""

        if isinstance(other, int):
            return other * self
        if not isinstance(other, self.__class__):
            raise TypeError("Can only right multiply with int or Space.")

        blocks = []
        for (m1, ir1), (m2, ir2) in itertools.product(self, other):
            m = m1 * m2
            lmin, lmax = abs(ir1.l - ir2.l), ir1.l + ir2.l
            blocks += [(m, SO3Irrep(l)) for l in range(lmin, lmax + 1)]

        return self.__class__(blocks)

    def _wigner_D(self, matrix: jax.Array) -> jax.Array:
        """Return Wigner D matrix from a 3x3 matrix.

        Wrapper around O3Irreps.D_from_matrix() for now.
        """
        return self._to_e3nn().D_from_matrix(matrix)

    def action(self, matrix: jax.Array) -> jax.Array:
        """Return the Wigner D matrix of a 3x3 rotation matrix.

        This is the action of a Lie-group matrix element of SO3
        on the representation.
        """
        return self._to_e3nn().D_from_matrix(matrix)

    def otimes(
        self: Self | str,
        other: Self | str,
        target_filter: Self | str,
        sort: bool = False,
    ) -> "SO3Space":
        r"""Tensor product representation, decomposed in irreducibles by CG rules.

        Clebsch-Gordan (CG) rules yield an explicit isomorphism between the
        tensor product of two irreducible blocks of degrees l1 and l2, as
        a direct sum of irreducible blocks of degrees L in
        :math:`\{ |l1-l2|, \ldots, l1+l2 \}`.

        The product of two O3-spaces can be decomposed by developing the tensor
        product space in products of irreducibles. The isomorphism of course depends
        on an arbitrary ordering of irreducible blocks, and basis function choices
        on the irreducible blocks (e.g. real vs. complex harmonics, Condon-Shortley
        phase convention...)
        """
        # TODO: Adapted from O3Space.otimes, consider inheriting instead.
        #       Can rely for this on Irrep.__mul__
        lhs = self if isinstance(self, SO3Space) else SO3Space(self)
        rhs = other if isinstance(other, SO3Space) else SO3Space(other)
        target = None if target_filter is None else SO3Space(target_filter)

        if target is not None:
            keep_ir_out = set((ir.l, 1) for _, ir in target)
            if any(m != 1 for m, _ in target):
                raise ValueError("Filter `out` should have multiplicities 1")

        irreps_out = []
        for (m1, irr1), (m2, irr2) in itertools.product(lhs, rhs):
            l1, l2 = irr1.l, irr2.l
            l_min, l_max = abs(l1 - l2), l1 + l2
            mul = m1 * m2

            # | l1 - l2 | <= l_out <= l1 + l1
            for l_out in range(l_min, l_max + 1):
                if target_filter is not None and (l_out, 1) not in keep_ir_out:
                    continue
                irreps_out.append((mul, l_out))

        target = SO3Space(irreps_out)
        return target.regroup() if sort else target

    def _to_e3nn(self) -> e3nn.Irreps:
        return e3nn.Irreps([(mul, (ir.l, 1)) for mul, ir in self])
        """Return `e3nn.Irreps` equivalent.

        Raises:
            ModuleNotFoundError if e3nn is not installed.
        """

    @classmethod
    def _parse_string(cls, blocks: str) -> list[tuple[int, SO3Irrep]]:
        out = []
        for block in re.split(_OPLUS_REGEX, blocks):
            parsed = re.findall(_MUL_REGEX, block)
            if not len(parsed):
                parsed = re.findall(_SO3_IRREP_REGEX, block)
                if not len(parsed):
                    raise ValueError(
                        f"Could not parse block '{block}' as '(<mul>x)<irrep>'"
                    )
                mul, irrep = 1, block
            else:
                mul, irrep = parsed[0]
            out.append((int(mul), SO3Irrep(irrep)))
        return out
        """Helper for initialization from e3nn-like string arguments."""

    @classmethod
    def _parse_list(
        cls, blocks: Iterable[tuple[int, SO3Irrep | int | str]]
    ) -> list[tuple[int, SO3Irrep]]:
        out = []
        for block in blocks:
            # (m, O3Irrep("0e")) or (m, "0e")
            if len(block) == 2:
                mul, irrep = block
            else:
                raise ValueError(
                    "Expecting an iterable of (mul, irrep) or (mul, l) blocks."
                )
            # Check mul
            if not isinstance(mul, int) and mul > 0:
                raise ValueError("Multiplicity should be a positive integer")
            # Cast irrep
            if isinstance(irrep, str):
                irrep = cls._irrep_type(irrep)
            elif not isinstance(irrep, SO3Irrep):
                if isinstance(irrep, tuple):
                    irrep = cls._irrep_type(*irrep)
                else:
                    irrep = cls._irrep_type(irrep)

            out.append((int(mul), irrep))

        return out
        """Helper for initialization from (mul, irrep) lists."""

    @classmethod
    def _parse_e3nn(cls, blocks: e3nn.Irreps) -> list[tuple[int, SO3Irrep]]:
        return [(r.mul, cls._irrep_type(r.ir)) for r in blocks]
        """Helper for initialization from e3nn.Irreps object."""
