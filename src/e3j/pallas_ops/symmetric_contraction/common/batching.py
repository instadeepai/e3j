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

"""`jax.vmap` and data parallelism for the fused symmetric contraction
kernels, shared by every backend.

Pallas kernels have no batching rule worth using here, and Mosaic kernels
cannot be partitioned by XLA. :func:`vmappable` instead folds the vmap axis
`V` into the kernel's own batch axis (and, for per-lane weights, into its
species axis), and splits `V` across the context mesh with a `shard_map`.
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec


def _merge(a: jax.Array, batched: bool, v: int) -> jax.Array:
    """`(V, n, ...)` (or `(n, ...)` broadcast over `V`) -> `(V * n, ...)`."""
    if not batched:
        a = jnp.broadcast_to(a, (v, *a.shape))
    return a.reshape((-1, *a.shape[2:]))


def _split(a: jax.Array, v: int) -> jax.Array:
    """Inverse of :func:`_merge`."""
    return a.reshape((v, -1, *a.shape[1:]))


def _fold(v, w, species, rows, w_b, s_b, rows_b, fold_w):
    """Fold the vmap axis `V` into the batch: `species` and `rows` to
    `(V * batch, ...)`. With `fold_w`, `w` becomes `(V * num_species, ...)`
    and every row's species is offset to its own `V` slice (after negative
    indices are wrapped and out-of-range ones clamped, as the kernels do)."""
    rows = [_merge(a, b, v) for a, b in zip(rows, rows_b)]
    if fold_w:
        num_species = w.shape[1 if w_b else 0]
        species = jnp.where(species < 0, species + num_species, species)
        species = jnp.clip(species, 0, num_species - 1)
        species = species + num_species * jnp.arange(v, dtype=species.dtype)[:, None]
        s_b = True
        w = _merge(w, w_b, v)
    return w, _merge(species, s_b, v), rows


def _most_lanes(v: int, num_species: int, fits: Callable[[int], bool] | None) -> int:
    """The most lanes, up to `v`, whose folded `lanes * num_species` species
    `fits` (assumed monotone); at least one."""
    if fits is None or fits(v * num_species):
        return v
    lo, hi = 1, v  # `lo` lanes fit (one is the unbatched call), `hi` do not.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid * num_species) else (lo, mid)
    return lo


def data_parallel(local, v: int, in_batched, *args):
    """Run `local(v, *args)` with the vmap axis `V` split across the context
    mesh.

    Mosaic kernels cannot be partitioned by XLA, so with a mesh set the
    batched operands are sharded along `V` by a `shard_map` (the others
    replicated) and every device runs `local` on its `V / devices` lanes.
    Mesh axes already manual (inside another `shard_map`) are left alone.

    A `V` the devices do not divide is folded on every device unsharded,
    for an outer vmap layer (whose rule then runs inside this one's lanes)
    to shard; with none, Mosaic refuses to be partitioned automatically.
    """
    mesh = jax.sharding.get_abstract_mesh()
    axes = tuple(a for a in mesh.axis_names if a not in mesh.manual_axes)
    d = math.prod(mesh.shape[a] for a in axes)
    if not axes or v % d:
        return local(v, *args)
    sharded = PartitionSpec(axes)
    return jax.shard_map(
        lambda *a: local(v // d, *a),
        mesh=mesh,
        in_specs=tuple(sharded if b else PartitionSpec() for b in in_batched),
        out_specs=sharded,
        check_vma=False,
    )(*args)


def vmappable(
    kernel: Callable,
    *,
    tile_w: bool = False,
    fits: Callable[[int], bool] | None = None,
) -> Callable:
    """`kernel(w, species, *rows)`, vmappable by folding into its batch.

    `w` is `(num_species, ...)` weights, `species` the `(batch,)` integer
    species of every row and each of `rows` a `(batch, ...)` array. Every
    output of `kernel` (a pytree, `None` leaves allowed) must be `(batch,
    ...)` or `(num_species, ...)`.

    Under `jax.vmap`, `V` is folded into the batch. Batched weights (or
    with `tile_w`, any) are folded into the species, every lane reading
    its own; outputs per species (e.g. `dw`) need `tile_w`, else they would
    sum over the lanes. Folded weights go `fits(num_species)` lanes' worth
    of species at a time (by default, all). `V` is split across the context
    mesh as by :func:`data_parallel`.
    """

    @jax.custom_batching.custom_vmap
    def call(w, species, *rows):
        return kernel(w, species, *rows)

    @call.def_vmap
    def rule(v, in_batched, w, species, *rows):
        w_b, s_b, *rows_b = in_batched
        fold_w = tile_w or w_b

        def local(v, w, species, *rows):
            num_species = w.shape[1 if w_b else 0]
            lanes = _most_lanes(v, num_species, fits) if fold_w else v
            outs = []
            for lo in range(0, v, lanes):
                n = min(lanes, v - lo)

                def chunk(a, b, lo=lo, n=n):
                    return a[lo : lo + n] if b and n < v else a

                cw, cs, crows = _fold(
                    n,
                    chunk(w, w_b),
                    chunk(species, s_b),
                    [chunk(a, b) for a, b in zip(rows, rows_b)],
                    w_b,
                    s_b,
                    rows_b,
                    fold_w,
                )
                # Call `call` again so that another vmap layer folds in the
                # same way.
                outs.append(
                    jax.tree.map(lambda a, n=n: _split(a, n), call(cw, cs, *crows))
                )
            if len(outs) == 1:
                return outs[0]
            return jax.tree.map(lambda *parts: jnp.concatenate(parts), *outs)

        out = data_parallel(local, v, in_batched, w, species, *rows)
        return out, jax.tree.map(lambda _: True, out)

    return call
