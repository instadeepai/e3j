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
from jax import Array

from e3j.spaces.so3 import SO3Irrep
from e3j.spaces.space import IrrepSum

_IRREP_REGEX = r"^(\d*)([e|o])$"
_OPLUS_REGEX = r"\s*\+\s*"
_MUL_REGEX = r"^(\d*)\s*x\s*(.*)"


@dataclass
class O3Irrep(SO3Irrep):
    """Irreducible representation of O3.

    Minimal O3-spaces are characterized by:

    * a degree `l >= 0` which is the degree of harmonic polynomials,
    * a parity `p in {-1, 1}` giving the sign taken under reflections.

    The e3nn-notation of "irreps" accepts a string where parity is encoded
    with either `"e"` (even) or `"o"` (odd), e.g.

        >>> blocks = O3Irrep("1o"), O3Irrep("2e"), O3Irrep("3o")
    """

    l: int
    p: int = 1

    @overload
    def __init__(self, irreps: str): ...
    @overload
    def __init__(self, irreps: "O3Irrep"): ...
    @overload
    def __init__(self, l: int): ...
    @overload
    def __init__(self, l: int, p: int): ...
    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], str):
                parsed = re.findall(_IRREP_REGEX, args[0])
                if not len(parsed):
                    raise ValueError(
                        f"Could not parse O3Irrep string {args[0]} as str(l) + 'e|o'"
                    )
                l, p = parsed[0]
                self.l = int(l)
                self.p = 1 if p == "e" else -1
            elif isinstance(args[0], O3Irrep):
                self.l = args[0].l
                self.p = args[0].p
            elif isinstance(args[0], int):
                self.l = args[0]
                self.p = 1
            else:
                raise ValueError(
                    f"O3Irrep expects int (l), str or O3Irrep as first argument, got {type(args[0])}"
                )
        elif len(args) == 2:
            self.l, self.p = args
        else:
            raise ValueError(
                "O3Irrep should be initialized with either (l, p) or str(l + 'e|o')"
            )

    @property
    def dim(self) -> int:
        """Dimension of the irreducible block: 2l + 1."""
        return 2 * self.l + 1

    def __iter__(self):
        yield self.l
        yield self.p

    def __str__(self) -> str:
        return str(self.l) + ("e" if self.p == 1 else "o")

    def __rmul__(self, other: int) -> "O3Space":
        return O3Space([(other, self)])

    def __mul__(self, other: "SO3Irrep") -> "O3Space":
        if isinstance(other, O3Irrep):
            l_min = abs(self.l - other.l)
            l_max = self.l + other.l
            p = self.p * other.p
            return O3Space((1, O3Irrep(l, p)) for l in range(l_min, l_max + 1))
        else:
            # We could support Space * Space too
            raise NotImplementedError("O3Irrep can only be multiplied with O3Irrep")

    def __lt__(self, other) -> bool:
        if self.l < other.l:
            return True
        if self.l > other.l:
            return False
        parity_l = (-1) ** self.l
        return self.p == parity_l and other.p == -parity_l

    def __eq__(self, other) -> bool:
        return self.l == other.l and self.p == other.p

    def __hash__(self) -> int:
        return hash(str(self))


O3Block: TypeAlias = tuple[int, O3Irrep | tuple[int, int] | str]


# NOTE: O3Space cannot inherit from SO3Space because of the `Array` attribute.
#       Error raised in `Array.__init_subclass__()`.
#       Since SO3 is a subgroup of O3, any O3-space also has an SO3-action.
class O3Space(IrrepSum[O3Irrep]):
    """Representation of O3, direct sum of irreducible blocks.

    O3-Spaces yield pairs `(mul, irrep)` when iterated through, where `mul`
    denotes the multiplicity of the irreducible subspace `irrep`.
    Each `irrep` may also be given as a pair of integers `(l, p)`.

    In e3nn-notation, subspaces are separated with `"+"` and multiplicities
    act on irreducibles with `"x"`, e.g.

        >>> space = e3j.Space("16x0e + 8x1o + 8x2e")
        >>> space = e3j.Space([(16, "0e"), (8, "1o"), (8, "2e")])
        >>> space = e3j.Space([(16, (0, 1)), (8, (1, -1)), (8, (2, 1))])
    """

    def __init__(self, blocks: str | Iterable[O3Block]):
        """Parse blocks and layout.

        There are multiple ways a `Space` instance can be described and initialized:

        * as a string of the form `"8x0e + 4x1o"`,
        * as a list of length-2 or length-3 tuples of the form
            * `[(8, O3Irrep("0e"), (4, O3Irrep("1o"))]`
            * `[(8, "0e"), (8, "1o")]`
            * `[(8, 0, 1), (4, 1, -1)]`
        """

        self.blocks: list[tuple[int, O3Irrep]] = []
        if isinstance(blocks, str):
            self.blocks = self._parse_string(blocks)
        elif isinstance(blocks, e3nn.Irreps):
            self.blocks = self._parse_e3nn(blocks)
        else:
            self.blocks = self._parse_list(blocks)

    @property
    def l_max(self) -> int:
        """Maximal degree of irreducible representations."""
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
            # TODO: rewrite generically with Irrep.__mul__
            lmin, lmax = abs(ir1.l - ir2.l), ir1.l + ir2.l
            p = ir1.p * ir2.p
            blocks += [(m, O3Irrep(l, p)) for l in range(lmin, lmax + 1)]

        return self.__class__(blocks)

    def otimes(
        self: Self | str,
        other: Self | str,
        target_filter: Self | str,
        sort: bool = False,
    ) -> "O3Space":
        """Tensor product representation, decomposed in irreducibles by CG rules.

        Clebsch-Gordan (CG) rules yield an explicit isomorphism between the
        tensor product of two irreducible blocks of degrees l1 and l2, as
        a direct sum of irreducible blocks of degrees L in {|l1-l2|, ..., l1+l2}.

        The product of two O3-spaces can be decomposed by developing the tensor
        product space in products of irreducibles. The isomorphism of course depends
        on an arbitrary ordering of irreducible blocks, and basis function choices
        on the irreducible blocks (e.g. real vs. complex harmonics, Condon-Shortley
        phase convention...)
        """
        lhs = self if isinstance(self, O3Space) else O3Space(self)
        rhs = other if isinstance(other, O3Space) else O3Space(other)
        target = None if target_filter is None else O3Space(target_filter)

        if target is not None:
            keep_ir_out = set((ir.l, ir.p) for _, ir in target)
            if any(m != 1 for m, _ in target):
                raise ValueError("Filter `out` should have multiplicities 1")

        irreps_out = []
        for (m1, irr1), (m2, irr2) in itertools.product(lhs, rhs):
            l1, l2 = irr1.l, irr2.l
            l_min, l_max = abs(l1 - l2), l1 + l2
            mul, parity = m1 * m2, irr1.p * irr2.p

            # | l1 - l2 | <= l_out <= l1 + l1
            for l_out in range(l_min, l_max + 1):
                if target_filter is not None and (l_out, parity) not in keep_ir_out:
                    continue
                irreps_out.append((mul, (l_out, parity)))

        target = O3Space(irreps_out)
        return target.regroup() if sort else target

    def action(self, matrix: Array) -> Array:
        """Return the Wigner D matrix of a 3x3 rotation matrix.

        This is the action of a Lie-group matrix element of O3
        on the representation.
        """
        return self._to_e3nn().D_from_matrix(matrix)

    def _to_e3nn(self) -> e3nn.Irreps:
        """Return `e3nn.Irreps` equivalent.

        Raises:
            ModuleNotFoundError if e3nn is not installed.
        """
        # TODO: mock e3nn module when e3nn dep gets dropped and raise on call.
        return e3nn.Irreps([(mul, (ir.l, ir.p)) for mul, ir in self])

    @classmethod
    def _parse_string(cls, blocks: str) -> list[tuple[int, O3Irrep]]:
        """Helper for initialization from e3nn-like string arguments."""
        out = []
        for block in re.split(_OPLUS_REGEX, blocks):
            parsed = re.findall(_MUL_REGEX, block)
            if not len(parsed):
                parsed = re.findall(_IRREP_REGEX, block)
                if not len(parsed):
                    raise ValueError(
                        f"Could not parse block '{block}' as '(<mul>x)<irrep>'"
                    )
                mul, irrep = 1, block
            else:
                mul, irrep = parsed[0]
            out.append((int(mul), O3Irrep(irrep)))
        return out

    @classmethod
    def _parse_list(
        cls, blocks: Iterable[tuple[int, O3Irrep | tuple | str]]
    ) -> list[tuple[int, O3Irrep]]:
        """Helper for initialization from (mul, irrep) lists."""
        out = []
        for block in blocks:
            # e3nn.MulIrrep produced by tuple(e3nn.Irreps(...)), e.g. via Flax nn.Module cast
            if isinstance(block, e3nn.MulIrrep):
                irrep = cls._irrep_type(block.ir.l, block.ir.p)  # type: ignore[call-arg]
                out.append((block.mul, irrep))
                continue
            # (m, O3Irrep("0e")) or (m, "0e")
            if len(block) == 2:
                mul, irrep = block
            # (m, 0, 1)
            elif len(block) == 3:
                mul, l, p = block
                irrep = cls._irrep_type(l, p)
            else:
                raise ValueError(
                    "Expecting an iterable of (mul, irrep) or (mul, l, p) blocks."
                )
            # Check mul
            if not isinstance(mul, int) and mul > 0:
                raise ValueError("Multiplicity should be a positive integer")
            # Cast irrep
            if isinstance(irrep, str):
                irrep = cls._irrep_type(irrep)
            elif not isinstance(irrep, cls._irrep_type):
                irrep = cls._irrep_type(*irrep)

            out.append((int(mul), irrep))

        return out

    @classmethod
    def _parse_e3nn(cls, blocks: e3nn.Irreps) -> list[tuple[int, O3Irrep]]:
        """Helper for initialization from e3nn.Irreps object."""
        return [(r.mul, cls._irrep_type(r.ir.l, r.ir.p)) for r in blocks]
