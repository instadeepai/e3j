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

"""Hardware granularities and unroll helpers shared by the Mosaic GPU kernels."""

from typing import Iterable

import jax
import jax.numpy as jnp

#: Mosaic GPU spreads a vector across a warpgroup's 128 lanes, and only
#: vectorizes a value materialized from SMEM when its element count is a
#: multiple of this.
WARPGROUP_LANES = 128

#: An untiled `copy_gmem_to_smem` must transfer a multiple of this many *bytes*
#: (`primitives.py::_copy_gmem_to_smem_lowering`). A DMA granularity, unrelated
#: to `WARPGROUP_LANES` despite both being 128 today.
UNTILED_COPY_MIN_BYTES = 128

#: Most elements a tiled async copy moves along one axis
#: (`mosaic/gpu/launch_context.py::async_copy` refuses more), so the widest
#: block of any axis a kernel stages in SMEM -- channel blocks and coordinate
#: axes alike.
MAX_COPY_ELEMS_PER_AXIS = 256


def shared_memory_budget() -> int:
    """Shared memory one CTA may opt into on the local device, in bytes."""
    return getattr(jax.local_devices()[0], "shared_memory_per_block_optin", 227 * 1024)


def add_term(total: jax.Array | None, term: jax.Array) -> jax.Array:
    """`total + term`, with `None` standing for an empty sum.

    The unrolled Clebsch-Gordan sums are built term by term at trace time, and
    most of them are short. Seeding them with `None` rather than a zero row keeps
    the first term from paying an add against zero -- which the compiler would
    have to fold away -- and avoids needing a laid-out zero row wherever a sum
    starts.
    """
    return term if total is None else total + term


def sum_terms(terms: Iterable[jax.Array]) -> jax.Array | None:
    """Left-fold `terms` with :func:`add_term`; `None` if there are none.

    Left-folding pins the summation order, so the arithmetic a kernel emits is
    exactly the order its coefficients are unrolled in.
    """
    total = None
    for term in terms:
        total = add_term(total, term)
    return total


def used_input_coordinates(params) -> tuple[list[int], list[int]]:
    """The `x` and `y` coordinates some Clebsch-Gordan path touches, sorted.

    The kernels materialize one SMEM value per *used* input coordinate, which
    matters for a pruned target where whole `x` or `y` slices contribute
    nothing. Reads only `params.indices`, so it serves either kernel family.
    """
    return (
        sorted({int(xi) for xi in params.indices[:, 1]}),
        sorted({int(yi) for yi in params.indices[:, 2]}),
    )


def pad_trailing_axis_for_untiled_copy(array: jax.Array) -> jax.Array:
    """Zero-pad the trailing axis so a whole-row untiled copy is legal.

    A one-batch-row block of a small trailing axis -- e.g. `y_dim` in the
    OUTER-mode fast path, where `y` carries no channel axis to pad the
    transfer out for free -- can fall under `UNTILED_COPY_MIN_BYTES`. Pad it
    up to that floor.
    """
    min_elems = -(-UNTILED_COPY_MIN_BYTES // array.dtype.itemsize)
    remainder = array.shape[-1] % min_elems
    if not remainder:
        return array
    padding = [(0, 0)] * array.ndim
    padding[-1] = (0, min_elems - remainder)
    return jnp.pad(array, padding)
