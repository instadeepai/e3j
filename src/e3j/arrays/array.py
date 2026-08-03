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

from typing import ClassVar, Generic, Iterable, TypeVar, Union, get_args

import jax
from flax.struct import dataclass, field

from e3j.spaces import Finite, O3Space, SO3Space, Space
from e3j.utils import config, options

SpaceT = TypeVar("SpaceT", bound=Space)
IntoSpaceT = Union[str, SpaceT]
IntoLayout = Union[str, options.Layout]


@dataclass
class Array(Generic[SpaceT]):
    """JAX-compatible dataclass storing an equivariant array.

    In addition to the `.array` attribute, the annotations
    `.space` and `.layout` describe the shape further.
    """

    space: SpaceT = field(pytree_node=False)
    array: jax.Array = field(pytree_node=True)
    layout: options.Layout | None = field(pytree_node=False, default=None)

    _space_type: ClassVar[type[Space]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Walk __orig_bases__ to find Array[SomeSpace] and extract SomeSpace
        for base in getattr(cls, "__orig_bases__", ()):
            args = get_args(base)
            if args and isinstance(args[0], type) and issubclass(args[0], Space):
                space_type = args[0]
                cls._space_type = space_type
                if hasattr(space_type, "_array_type"):
                    raise RuntimeError(
                        f"Space {space_type} already has an array type {space_type._array_type}, cannot assign {cls} as its array type."
                    )
                space_type._array_type = cls  # type: ignore
                break
        if not hasattr(cls, "_space_type"):
            raise TypeError(
                f"Could not find space type for {cls}, make sure to subclass Array[SomeSpace]"
            )

    def __init__(
        self, space: IntoSpaceT, array: jax.Array, layout: IntoLayout | None = None
    ):
        if layout is None:
            layout = config().layout
        if not isinstance(layout, options.Layout):
            layout = options.Layout.parse(layout)
        object.__setattr__(self, "layout", layout)
        # Cast space attribute to _space_type type
        if not isinstance(space, self._space_type):
            space = self._space_type(space)  # type: ignore
        object.__setattr__(self, "space", space)
        # Cast input data to jax.numpy.ndarray
        if not isinstance(array, jax.Array):
            array = jax.numpy.asarray(array)
        object.__setattr__(self, "array", array)
        # Check space dimension matches array shape on feature axis.
        dim = self.space.dim
        axis = self.feature_axis
        if self.shape[axis] != dim:
            raise ValueError(f"Feature dimension {axis} is not of dimension {dim}")

    @property
    def feature_axis(self) -> int:
        match self.layout:
            case options.Layout.TRAILING_CHANNELS:
                return -2
            case options.Layout.LEADING_CHANNELS:
                return -1
            case options.Layout.E3NN:
                return -1
            case _:
                raise ValueError(f"Unknown layout: {self.layout}")

    def blocks(self) -> Iterable["Array"]:
        """Yield blocks of isomorphic irreducible subspaces.

        Equivariant linear operations can only mix irreducible representations
        from a same block, i.e. they are essentially block-diagonal operations,
        see :class:`e3j.linen.Linear`.

        This method is analogous to `e3nn.IrrepsArray.chunks` except that:
        * there is no notion of `None` chunks / `zero_flags`
        * the slice axis depends on the layout.
        """
        for (mul, ir), slc in zip(self.space, self.space.slices()):
            if self.layout == options.Layout.TRAILING_CHANNELS:
                arr = self.array[..., slc, :]
            else:
                arr = self.array[..., slc]
            yield self.__class__(self.space.__class__([(mul, ir)]), arr, self.layout)

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def _alike(self, array: jax.Array) -> "Array":
        return self.__class__(self.space, array, self.layout)

    def __getitem__(self, idx):
        ndim = self.ndim
        axis = self.feature_axis % ndim
        key = idx if isinstance(idx, tuple) else (idx,)
        if Ellipsis in key:
            ellipsis_pos = key.index(Ellipsis)
            pre, post = key[:ellipsis_pos], key[ellipsis_pos + 1 :]
            n_explicit = sum(1 for k in pre if k is not None) + sum(
                1 for k in post if k is not None
            )
            key = pre + (slice(None),) * (ndim - n_explicit) + post
        pos = 0
        for k in key:
            if k is None:
                continue
            # A boolean array consumes as many axes as its own rank (it is
            # matched against that many corresponding axes of self.array),
            # unlike every other index type which always consumes exactly
            # one axis.
            span = k.ndim if getattr(k, "dtype", None) == bool else 1
            if pos <= axis < pos + span and k != slice(None):
                raise ValueError(
                    f"Indexing the feature axis ({axis}) would silently "
                    f"reindex the irreps of {self.space}; got index {k!r} "
                    "at that axis."
                )
            pos += span
        return self._alike(self.array[idx])

    def _check_alike(self, other):
        if (
            not isinstance(other, self.__class__)
            or self.space != other.space
            or self.layout != other.layout
        ):
            raise ValueError(f"Can only add arrays that are alike {other}")

    def __add__(self, other):
        self._check_alike(other)
        return self._alike(self.array + other.array)

    def __sub__(self, other):
        self._check_alike(other)
        return self._alike(self.array - other.array)

    def __radd__(self, other):
        raise ValueError(f"Can only add arrays that are alike {other}")

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self._alike(other * self.array)
        if isinstance(other, jax.Array):
            axis = self.feature_axis
            feature_size = other.shape[axis] if other.ndim >= -axis else 1
            if other.size == 1 or feature_size == 1:
                return self._alike(other * self.array)
        raise ValueError(f"Cannot left multiply with non-scalar {other}")

    def __mul__(self, other):
        raise NotImplementedError(
            "Cannot multiply equivariant data, though scalar mixing may be "
            "eventually supported this way."
        )

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        cls = self.__class__.__name__
        out = f"{cls} '{self.space}'"
        if self.layout != config().layout:
            out += f" {self.layout}"
        return out + "\n" + str(self.array)


@dataclass(init=False)
class SO3Array(Array[SO3Space]):
    """Arrays of SO3-equivariant data."""


@dataclass(init=False)
class O3Array(Array[O3Space]):
    """Arrays of O3-equivariant data."""


@dataclass(init=False)
class IndexArray(Array[Finite]):
    """Arrays of integer indices."""
