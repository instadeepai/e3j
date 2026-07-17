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

import abc
from dataclasses import dataclass
from typing import (
    ClassVar,
    Generic,
    Iterator,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
from typing_extensions import get_args


@dataclass
class Irrep(abc.ABC):
    """Small dataclass representing an irreducible block for `IrrepSum` spaces.

    Implementations may depend on the symmetry group, however any `Irrep` is
    required to define:
    * `dim` : size of the irreducible block,
    * `__lt__` : used by sorting helpers,
    * `__eq__`.

    Note:
        The actual :class:`Space` interface of an irreducible representation
        is defined on `IrrepSum([(1, irrep)])`, see :class:`IrrepSum`.
    """

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        raise NotImplementedError(
            "Irrep is a an abstract base class, use e.g. O3Irrep instead."
        )

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abc.abstractmethod
    def __lt__(self, other: object) -> bool: ...


Block: TypeAlias = tuple[int, Irrep]

IrrepT = TypeVar("IrrepT", bound=Irrep)


class Space(abc.ABC):
    """Base class for G-representations, a.k.a. G-spaces."""

    _array_type: ClassVar[type["Array"]]

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimension of the representation."""
        raise NotImplementedError(
            "Space is an abstract base class, use e.g. O3Space instead."
        )

    @abc.abstractmethod
    def slices(self) -> list[slice]: ...


class IrrepSum(Space, abc.ABC, Generic[IrrepT]):
    """Base class representing sums of irreducible blocks.

    These spaces can be iterated to yield (mul, ir) blocks where
    * `mul` is an integer multiplicity,
    * `ir` is an irreducible (aka "simple") representation of the group.

    The representations of a "semi-simple" group G (e.g. O3 and SO3)
    is also called semi-simple G-module, which means it can always
    be decomposed into a direct sum of simple G-modules (aka. irrep).
    """

    _irrep_type: ClassVar[type[Irrep]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            args = get_args(base)
            if args and isinstance(args[0], type) and issubclass(args[0], Irrep):
                cls._irrep_type = args[0]
                break
        if not hasattr(cls, "_irrep_type"):
            raise TypeError("IrrepSpace subclasses must specify an Irrep type")

    def __init__(self, blocks: list[tuple[int, IrrepT]]):
        self.blocks = blocks

    @property
    def dim(self) -> int:
        """Dimension of the representation."""
        return sum(m * ir.dim for m, ir in self)

    @property
    def num_irreps(self) -> int:
        """Number of irreducible subspaces."""
        return sum(m for m, _ in self.blocks)

    def __iter__(self) -> Iterator[tuple[int, IrrepT]]:
        """Yield (mul, ir) pairs of irreducible spaces with multiplicity."""
        for block in self.blocks:
            m, ir = block
            yield m, ir

    def slices(self) -> list[slice]:
        """Slice of indices spanning each sub-block."""
        out, begin = [], 0
        for m, ir in self:
            size = m * ir.dim
            out.append(slice(begin, begin + size))
            begin += size
        return out

    @overload
    def sort(self, return_inverse: Literal[False] = ...) -> Self: ...

    @overload
    def sort(self, return_inverse: Literal[True]) -> tuple[Self, np.ndarray]: ...

    def sort(self, return_inverse: bool = False) -> Self | tuple[Self, np.ndarray]:
        """Sort irreducible blocks by degree and parity.

        Note that parities are sorted so that `p = (-1)**l`
        comes first. This choice agrees with e3nn.
        """
        blocks = self.blocks
        sorted_blocks = sorted((ir, i, mul) for i, (mul, ir) in enumerate(self))
        inv = np.array([i for _, i, _ in sorted_blocks])

        sorted_space = self.__class__([blocks[i] for i in inv])
        if return_inverse:
            return sorted_space, inv
        return sorted_space

    def regroup(self, sort: bool = True) -> Self:
        """Sort and regroup irreducible blocks with same degree and parity.

        This method yields the most compact description of the representation.

        In some cases, one may however wish to keep distinct groups of same
        degree and parity, e.g. to learn linear maps that do not mix all
        channels, but respect a given partition.
        """
        if not sort:
            raise NotImplementedError(".regroup() will always sort for now")

        grouped_blocks = []
        sorted_blocks = list(self.sort())

        if len(sorted_blocks) <= 1:
            return self

        m1, ir1 = sorted_blocks[0]
        for m2, ir2 in sorted_blocks[1:]:
            if ir1 == ir2:
                m1 += m2
            else:
                grouped_blocks.append((m1, ir1))
                m1, ir1 = m2, ir2
        grouped_blocks.append((m1, ir1))
        return self.__class__(grouped_blocks)

    def __rmul__(self, other) -> Self:
        """Multiply multiplicities."""
        if not isinstance(other, int):
            raise TypeError("Can only left multiply with int multiplicities.")
        return self.__class__(list((other * m, ir) for m, ir in self))

    def __eq__(self, other):
        if not len(self.blocks) == len(other.blocks):
            return False
        return all(m1 == m2 and ir1 == ir2 for (m1, ir1), (m2, ir2) in zip(self, other))

    def __hash__(self) -> int:
        return hash(tuple((m, ir) for m, ir in self))

    def __str__(self) -> str:
        return " + ".join(f"{m}x{ir}" for m, ir in self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self}')"
