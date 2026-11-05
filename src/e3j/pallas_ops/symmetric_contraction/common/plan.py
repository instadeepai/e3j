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

"""Reduced evaluation plan of the fused SymmetricContraction.

Both backends' kernels walk this plan, forward and backward.

Per batch row and channel, every output block computes

```
y[mo] = sum_i W[mo, i] * F_i(x)
```

where the features `F_i` are irrep copies of the powers `x, x(x)x, ...`
(:class:`BlockPlan`). Most features are linearly dependent, by the rules
below: at `l_max=3`, correlation 3 -> `0e+1o`, the 28 + 58 features span
only 13 + 16 dimensions.

So for each `(block, leaf multiset)` the plan keeps a cheap spanning subset
of the features, and folds the rest into the weights:

```
y = sum_i W[i] F_i = sum_j (sum_i W[i] A[i, j]) F_sel_j = sum_j W_eff[j] F_sel_j
```

`A` is solved once at trace time, in float64, without evaluating anything:
each feature is a polynomial in `x` whose coefficients follow from the CG
tables alone (:func:`_polynomials`), and the dependencies are those of the
coefficients. They obey three rules:

* A feature's monomials are products of its own leaves, the source copies
  its CG path multiplies: features with different leaf multisets never mix,
  so `A` is block-diagonal by multiset, in blocks of a few features.
* A multiset `{a^n_a, b^n_b, ...}` holds as many independent `(L, p)`
  features as `L` appears in `Sym^n_a(V_la) (x) Sym^n_b(V_lb) (x) ...` (a
  plethysm), fewer if `l_max` pruned their paths.
* Reordering the leaves of a path only recouples it, e.g. `(b, a) -> L` is
  `(-1)^(la + lb + L) (a, b) -> L` and `(a, a) -> L` vanishes for odd `L`.
  Symmetrizing the path's coefficients over its equal leaves applies all of
  these at once.

The caller applies `W_eff = W @ A` (:func:`fold_weights`), and the kernel
only computes the selected features.

Last-power features sharing a CG block are then factorized:

```
sum_j W_eff[j] CG(P_prev[b_j], x_a) = CG(sum_j W_eff[j] P_prev[b_j], x_a)
```

so each group needs one CG contraction instead of one per feature.

The CG coefficients are float32, so dependencies only hold to ~1e-7; the
thresholds below sit well above that. The finished plan is checked against the
unreduced contraction in float64, at random samples, before use.
"""

import functools
from collections import Counter, defaultdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from e3j.pallas_ops.symmetric_contraction.common.params import (
    PallasSymmetricContractionStructure,
)

#: A feature is selected if the part of its coefficients outside the span of
#: those already selected exceeds this fraction of their norm. It is
#: identically zero, e.g. `(u, u) -> L` for odd `L`, if symmetrizing over
#: its equal leaves keeps less than this fraction of their norm
#: (:class:`Polynomial`). Dependent and zero features sit near 1e-7 (float32
#: rounding), independent and real ones at 1e-1 or above.
_INDEPENDENCE_THRESHOLD = 1e-4

#: Largest relative float64 error the finished plan may have.
_VERIFY_TOLERANCE = 1e-5

#: Fold coefficients this close to 0 or +-1 are snapped to it.
_SNAP_TOLERANCE = 1e-6


class Copy(NamedTuple):
    """One irrep `(2l+1)`-block of a power.

    `prev` and `src` are the copies of the previous power and of the source
    whose CG product makes this copy (-1 for the source itself).
    """

    stage: int
    index: int
    l: int
    p: int
    start: int
    prev: int
    src: int

    @property
    def dim(self) -> int:
        return 2 * self.l + 1


class Materialized(NamedTuple):
    """An intermediate copy the kernel computes.

    Coordinate `start + m` of power `stage` is the sum over `terms[m]` of
    `value * P[stage - 1][a] * x[b]`. Second-power self-pairs are
    symmetrized: `(a, b)` and `(b, a)` merge into `a <= b`.
    """

    stage: int
    start: int
    terms: tuple[tuple[tuple[int, int, float], ...], ...]


class Group(NamedTuple):
    """A factorized last-power contraction, `out += CG(R, x_a)`.

    `R = sum_j W_eff[column_j] * P[last - 1][start_j:]` over the `(column,
    start)` `members`, and each `(m, i, k, value)` of `terms` adds
    `value * R[i] * x[src_start + k]` to `out[m]`.
    """

    src_start: int
    prev_dim: int
    terms: tuple[tuple[int, int, int, float], ...]
    members: tuple[tuple[int, int], ...]


class ReducedBlock(NamedTuple):
    """One output block, mixed from its selected features.

    `weight_index` is the block's index in `params.blocks`; its folded
    weight for `(mo, column)` is row `weight_row + mo * columns + column` of
    the :func:`fold_weights` table. `offset` is `BlockPlan.offset`. Each
    column is used once, either by a `direct` feature `(column, stage,
    start)`, with `out[mo] += W_eff[mo, column] * P[stage][start:start +
    dim]`, or by a `groups` member. `fold` is the `(m_in, columns)` map `A`.
    """

    weight_index: int
    weight_row: int
    offset: int
    m_out: int
    dim: int
    columns: int
    direct: tuple[tuple[int, int, int], ...]
    groups: tuple[Group, ...]
    fold: np.ndarray


class ReducedPlan(NamedTuple):
    """Everything the kernel body unrolls.

    Attributes:
        x_coords: Source coordinates the body reads.
        materialized: Intermediate copies to compute, in dependency order.
        last: The power whose features are factorized into groups.
        blocks: The output blocks with at least one feature, in output order.
        weight_rows: Rows of the folded weight table (sum of every block's
            `m_out * columns`).
    """

    x_coords: tuple[int, ...]
    materialized: tuple[Materialized, ...]
    last: int
    blocks: tuple[ReducedBlock, ...]
    weight_rows: int


def fold_weights(weights: list[jax.Array], plan: ReducedPlan) -> jax.Array:
    """Every block's `W_eff = W @ A`, in one `(S, plan.weight_rows, C)` table
    of the weights' dtype.

    Written as sums of weight slices rather than a matmul: `A` is very
    sparse, so XLA fuses it into one elementwise pass, in exact float32 (no
    TF32). Lower-precision weights, e.g. bfloat16, are upcast first and the
    table is rounded back once at the end, so the fold only adds that one
    rounding.

    If every block was dropped (no feature survives), the table has no rows.

    Weights of identically-zero features are never read, so a non-finite
    one there does not give `NaN` as it would in the unfused contraction.
    """
    rows = []
    for block in plan.blocks:
        w = weights[block.weight_index].astype(jnp.float32)
        fold = block.fold
        for mo in range(block.m_out):
            for j in range(block.columns):
                row = None
                for i in np.flatnonzero(fold[:, j]):
                    term = w[:, :, mo, int(i)]
                    if fold[i, j] != 1.0:
                        term = term * np.float32(fold[i, j])
                    row = term if row is None else row + term
                rows.append(row)
    if not rows:
        s, c = weights[0].shape[:2]
        return jnp.zeros((s, 0, c), weights[0].dtype)
    # (S, C) rows -> (S, weight_rows, C)
    return jnp.stack(rows, axis=1).astype(weights[0].dtype)


@functools.lru_cache(maxsize=64)
def build_plan(params: PallasSymmetricContractionStructure) -> ReducedPlan:
    """The reduced plan of `params`, cached on its static content."""
    triples = _triples(params)
    copies, blocks_coef = _structure(params, triples)
    last = params.correlation - 1
    copy_at = [{c.start: c for c in stage} for stage in copies]

    # features[block][stage] = [(mi, copy), ...]
    features = []
    for block in params.blocks:
        by_stage = defaultdict(list)
        for mi in range(block.m_in):
            stage, coord = block.input_map[mi * block.dim_l]
            copy = copy_at[stage][coord]
            if (copy.l, copy.p) != (block.l, block.p):
                raise RuntimeError(
                    f"feature {mi} of block (l={block.l}, p={block.p}) is a "
                    f"({copy.l}, {copy.p}) copy"
                )
            by_stage[stage].append((mi, copy))
        features.append(dict(by_stage))

    poly = _polynomials(copies, blocks_coef)
    selected = _select(params, copies, blocks_coef, poly, features, last)

    reduced = []
    weight_rows = 0
    for bi, block in enumerate(params.blocks):
        fold, direct, groups = _fold_and_factorize(
            block, features[bi], selected[bi], poly, copies, blocks_coef, last
        )
        if fold.shape[1]:
            reduced.append(
                ReducedBlock(
                    weight_index=bi,
                    weight_row=weight_rows,
                    offset=block.offset,
                    m_out=block.m_out,
                    dim=block.dim_l,
                    columns=fold.shape[1],
                    direct=direct,
                    groups=groups,
                    fold=fold,
                )
            )
            weight_rows += block.m_out * fold.shape[1]

    materialized, x_coords = _materialize(reduced, triples, copies, copy_at, last)
    plan = ReducedPlan(
        x_coords=x_coords,
        materialized=materialized,
        last=last,
        blocks=tuple(reduced),
        weight_rows=weight_rows,
    )
    _verify(plan, params, triples, np.random.default_rng(1))
    return plan


def _triples(params) -> list:
    """Each power's `(zi, xi, yi, value)` CG arrays, stably sorted by `zi`
    (`None` for the source)."""
    out = [None]
    for idx, val in zip(params.stage_indices, params.stage_values):
        idx = np.asarray(idx).astype(np.int64)
        order = np.argsort(idx[:, 0], kind="stable")
        out.append((*idx[order].T, np.asarray(val, dtype=np.float64)[order]))
    return out


def _copy_triples(triples, copy):
    """The `(zi, xi, yi, value)` triples writing `copy`."""
    zi, xi, yi, val = triples[copy.stage]
    lo, hi = np.searchsorted(zi, [copy.start, copy.start + copy.dim])
    return zi[lo:hi], xi[lo:hi], yi[lo:hi], val[lo:hi]


def _structure(params, triples):
    """`(copies, coef)`: `copies[s]` lists power `s`'s copies in coordinate
    order, and `coef[s][index]` is that copy's dense `(dim, dim_prev,
    dim_src)` CG block."""
    copies = []
    for s, irreps in enumerate(params.stage_irreps):
        stage, start = [], 0
        for mul, l, p in irreps:
            for _ in range(mul):
                stage.append(Copy(s, len(stage), l, p, start, -1, -1))
                start += 2 * l + 1
        copies.append(stage)

    owner = [np.repeat(np.arange(len(st)), [c.dim for c in st]) for st in copies]
    coef = [None]
    for s in range(1, params.correlation):
        stage_blocks = {}
        resolved = []
        for c in copies[s]:
            zi, xi, yi, val = _copy_triples(triples, c)
            prev = set(owner[s - 1][xi].tolist())
            src = set(owner[0][yi].tolist())
            if len(prev) != 1 or len(src) != 1:
                raise RuntimeError(
                    f"copy {c.index} of power {s} reads {len(prev)} previous "
                    f"and {len(src)} source copies, expected one of each"
                )
            a, b = copies[s - 1][prev.pop()], copies[0][src.pop()]
            block = np.zeros((c.dim, a.dim, b.dim))
            np.add.at(block, (zi - c.start, xi - a.start, yi - b.start), val)
            stage_blocks[c.index] = block
            resolved.append(c._replace(prev=a.index, src=b.index))
        copies[s] = resolved
        coef.append(stage_blocks)
    return copies, coef


class Polynomial(NamedTuple):
    """A copy as a polynomial in the source.

    `leaves` are the source copies its CG path multiplies, sorted: its
    monomials are products of their coordinates. `coef` is the flattened
    `(dim, *leaf_dims)` coefficient tensor in that leaf order, symmetrized
    over equal leaves (as their product commutes), and `scale` is the
    tensor's norm before symmetrizing.
    """

    leaves: tuple[int, ...]
    coef: np.ndarray
    scale: float


def _polynomials(copies, coef):
    """`poly(stage, index)`: the :class:`Polynomial` of a copy, from the CG
    blocks `coef` alone."""
    # (stage, index) -> (leaves in path order, `(dim, *leaf_dims)` tensor).
    # A plain dict, not a recursive cached function: that would be a
    # reference cycle keeping every tensor alive until a full GC.
    paths = {}

    def path(stage, index):
        start = (stage, index)
        chain = []
        while stage > 0 and (stage, index) not in paths:
            chain.append((stage, index))
            stage, index = stage - 1, copies[stage][index].prev
        if stage == 0:
            paths.setdefault((0, index), ((index,), np.eye(copies[0][index].dim)))
        for key in reversed(chain):
            copy = copies[key[0]][key[1]]
            leaves, tensor = paths[(key[0] - 1, copy.prev)]
            # out[m, ..., b] = sum_a CG[m, a, b] prev[a, ...]
            tensor = np.einsum("mab,a...->m...b", coef[key[0]][key[1]], tensor)
            paths[key] = ((*leaves, copy.src), tensor)
        return paths[start]

    @functools.cache
    def poly(stage, index):
        leaves, tensor = path(stage, index)
        order = np.argsort(leaves, kind="stable")
        tensor = np.transpose(tensor, (0, *(1 + order)))
        leaves = tuple(leaves[i] for i in order)
        return Polynomial(
            leaves=leaves,
            coef=_symmetrize(tensor, leaves).ravel(),
            scale=float(np.linalg.norm(tensor)),
        )

    return poly


def _symmetrize(tensor: np.ndarray, leaves: tuple[int, ...]) -> np.ndarray:
    """Average `tensor` over the permutations of the axes of equal (sorted)
    `leaves`, axis `k + 1` holding leaf `k`.

    One leaf at a time, `Sym_k = (1 + sum_{i < k} (i k)) / k * Sym_{k-1}`:
    `O(n^2)` transposes for `n` equal leaves instead of `n!`.
    """
    lo = 0
    for n in Counter(leaves).values():  # in sorted order
        for k in range(1, n):
            total = tensor
            for i in range(k):
                axes = list(range(tensor.ndim))
                axes[1 + lo + i], axes[1 + lo + k] = axes[1 + lo + k], axes[1 + lo + i]
                total = total + np.transpose(tensor, axes)
            tensor = total / (k + 1)
        lo += n
    return tensor


def _evaluate(params, triples, x: np.ndarray) -> list:
    """Every power at the samples `x`, each `(n, stage_dims[s])` float64."""
    out = [x]
    for s in range(1, params.correlation):
        zi, xi, yi, val = triples[s]
        # `reduceat` over the sorted terms: ~10x faster than a scatter-add.
        starts = np.flatnonzero(np.r_[True, zi[1:] != zi[:-1]])
        prev = out[-1]
        cur = np.zeros((x.shape[0], params.stage_dims[s]))
        prod = val * prev[:, xi] * x[:, yi]
        cur[:, zi[starts]] = np.add.reduceat(prod, starts, axis=1)
        out.append(cur)
    return out


def _copy_cost(coef_block: np.ndarray, self_pair: bool) -> int:
    """Rough FP-op count of materializing one copy: per coordinate, one per
    `(a, b)` term plus one per distinct `a`."""
    nz = np.argwhere(coef_block != 0)
    if self_pair:
        terms = {(m, min(a, b), max(a, b)) for m, a, b in nz}
    else:
        terms = {(m, a, b) for m, a, b in nz}
    return len(terms) + len({(m, a) for m, a, _ in terms})


def _group_cost(coef_block: np.ndarray, m_out: int) -> int:
    """FP ops of `out += CG(R, x_a)` for `m_out` `R`s, best grouping."""
    nz = np.argwhere(coef_block != 0)
    by_mi = len({(m, i) for m, i, _ in nz})
    by_mk = len({(m, k) for m, _, k in nz})
    return min(len(nz) + m_out * by_mi, m_out * (len(nz) + by_mk))


def _select(params, copies, coef, poly, features, last):
    """Greedily pick a spanning subset of each `(block, leaf multiset)`'s
    features.

    Identically-zero features are dropped first. Then repeatedly take the
    remaining feature with the lowest marginal cost (its mix, any group it
    opens, and copies not yet materialized). Keep it if its coefficients are
    independent of the features already kept, else drop it for good.
    Returns `selected[block][stage] = [mi, ...]`.
    """
    materialized = {(0, c.index) for c in copies[0]}
    open_groups = set()

    @functools.cache
    def copy_cost(stage, index):
        c = copies[stage][index]
        return _copy_cost(coef[stage][index], stage == 1 and c.prev == c.src)

    @functools.cache
    def group_cost(stage, index, m_out):
        return _group_cost(coef[stage][index], m_out)

    def mat_cost(stage, index):
        cost = 0
        while stage > 0 and (stage, index) not in materialized:
            cost += copy_cost(stage, index)
            stage, index = stage - 1, copies[stage][index].prev
        return cost

    def mark(stage, index):
        while stage > 0 and (stage, index) not in materialized:
            materialized.add((stage, index))
            stage, index = stage - 1, copies[stage][index].prev

    def group_key(bi, copy):
        prev = copies[copy.stage - 1][copy.prev]
        return (bi, copy.src, prev.l, prev.p)

    def cost(bi, copy):
        m_out = params.blocks[bi].m_out
        if copy.stage == last and last > 0:
            prev = copies[copy.stage - 1][copy.prev]
            opening = group_key(bi, copy) not in open_groups
            return (
                m_out * prev.dim
                + (group_cost(copy.stage, copy.index, m_out) if opening else 0)
                + mat_cost(copy.stage - 1, copy.prev)
            )
        return m_out * copy.dim + mat_cost(copy.stage, copy.index)

    def is_zero(stage, copy):
        """Whether symmetrizing cancels the feature (`_INDEPENDENCE_THRESHOLD`
        of its norm before), e.g. `(u, u) -> L` for odd `L`."""
        p = poly(stage, copy.index)
        return np.linalg.norm(p.coef) <= _INDEPENDENCE_THRESHOLD * p.scale

    bases = {}  # (block, leaves) -> list of orthonormal rows
    selected = [defaultdict(list) for _ in params.blocks]
    remaining = [
        (bi, stage, mi, copy)
        for bi, by_stage in enumerate(features)
        for stage, feats in by_stage.items()
        for mi, copy in feats
        if not is_zero(stage, copy)
    ]
    while remaining:
        remaining.sort(key=lambda f: (cost(f[0], f[3]), f[0], f[1], f[2]))
        for i, (bi, stage, mi, copy) in enumerate(remaining):
            p = poly(stage, copy.index)
            basis = bases.setdefault((bi, p.leaves), [])
            r = p.coef / np.linalg.norm(p.coef)
            for q in basis:
                r = r - (q @ r) * q
            # Second Gram-Schmidt pass, for numerical stability.
            for q in basis:
                r = r - (q @ r) * q
            res = np.linalg.norm(r)
            if res > _INDEPENDENCE_THRESHOLD:
                basis.append(r / res)
                selected[bi][stage].append(mi)
                if stage == last and last > 0:
                    open_groups.add(group_key(bi, copy))
                    mark(stage - 1, copy.prev)
                else:
                    mark(stage, copy.index)
                # The features skipped before it are dependent: drop them.
                remaining = remaining[i + 1 :]
                break
        else:
            # Nothing left is independent.
            remaining = []
    return selected


def _fold_and_factorize(block, by_stage, selected, poly, copies, coef, last):
    """`(fold, direct, groups)` of one block, from its selected features."""
    columns = []  # (stage, mi, copy)
    for stage in sorted(by_stage):
        chosen = set(selected.get(stage, ()))
        for mi, copy in by_stage[stage]:
            if mi in chosen:
                columns.append((stage, mi, copy))
    fold = np.zeros((block.m_in, len(columns)))
    column_of = {mi: j for j, (_, mi, _) in enumerate(columns)}
    # Features of different leaf multisets share no monomial: express each
    # in terms of the selected features of its own multiset (none if they
    # are all identically zero).
    by_leaves = defaultdict(list)  # leaves -> [(mi, polynomial)]
    for stage, feats in by_stage.items():
        for mi, copy in feats:
            p = poly(stage, copy.index)
            by_leaves[p.leaves].append((mi, p))
    for feats in by_leaves.values():
        vecs = np.stack([p.coef for _, p in feats])
        chosen = [(column_of[mi], p.coef) for mi, p in feats if mi in column_of]
        cols = [j for j, _ in chosen]
        basis = np.array([v for _, v in chosen]).reshape(len(chosen), vecs.shape[1])
        coeff, *_ = np.linalg.lstsq(basis.T, vecs.T, rcond=None)
        resid = np.linalg.norm(basis.T @ coeff - vecs.T, axis=0)
        limit = _VERIFY_TOLERANCE * np.array([p.scale for _, p in feats])
        for (mi, _), bad in zip(feats, resid > limit):
            if bad:
                raise RuntimeError(
                    f"feature {mi} of block (l={block.l}, p={block.p}) is not "
                    f"spanned by the selected features"
                )
        # Snap rounding noise to exact 0 / +-1.
        for target in (0.0, 1.0, -1.0):
            coeff[np.abs(coeff - target) < _SNAP_TOLERANCE] = target
        for (mi, _), row in zip(feats, coeff.T):
            fold[mi, cols] = row

    def group_key(stage, copy):
        """The copy's group, or `None` if it is not a last-power feature.

        Members share their source copy, previous `(l, p)` and CG block. In
        e3j's tables the block only depends on the `l`s, so it never splits
        a group; it keeps hand-built tables correct.
        """
        if stage != last or last == 0:
            return None
        prev = copies[stage - 1][copy.prev]
        return (copy.src, prev.l, prev.p, coef[stage][copy.index].tobytes())

    # A group of one gains nothing and has a costlier backward: mix it
    # directly instead.
    group_size = defaultdict(int)
    for stage, _, copy in columns:
        group_size[group_key(stage, copy)] += 1

    direct, grouped = [], {}
    for j, (stage, _, copy) in enumerate(columns):
        key = group_key(stage, copy)
        if key is not None and group_size[key] > 1:
            prev = copies[stage - 1][copy.prev]
            _, members = grouped.setdefault(key, (coef[stage][copy.index], []))
            members.append((j, prev.start))
        else:
            direct.append((j, stage, copy.start))

    groups = []
    for key, (ref, members) in grouped.items():
        src = copies[0][key[0]]
        terms = tuple(
            (int(m), int(i), int(k), float(ref[m, i, k]))
            for m, i, k in np.argwhere(ref != 0)
        )
        groups.append(
            Group(
                src_start=src.start,
                prev_dim=ref.shape[1],
                terms=terms,
                members=tuple(members),
            )
        )
    return fold, tuple(direct), tuple(groups)


def _materialize(blocks, triples, copies, copy_at, last):
    """`(materialized, x_coords)`: the copies and source coordinates the
    blocks need."""
    needed = set()
    x_coords = set()

    def need(stage, start):
        copy = copy_at[stage][start]
        while stage > 0 and (stage, copy.index) not in needed:
            needed.add((stage, copy.index))
            x_coords.update(
                range(
                    copies[0][copy.src].start,
                    copies[0][copy.src].start + copies[0][copy.src].dim,
                )
            )
            stage, copy = stage - 1, copies[stage - 1][copy.prev]
        if stage == 0:
            x_coords.update(range(copy.start, copy.start + copy.dim))

    for block in blocks:
        for _, stage, start in block.direct:
            need(stage, start)
        for group in block.groups:
            x_coords.update(group.src_start + k for _, _, k, _ in group.terms)
            for _, start in group.members:
                need(last - 1, start)

    materialized = []
    for stage, index in sorted(needed):
        copy = copies[stage][index]
        zis, xis, yis, val = _copy_triples(triples, copy)
        per_coord = [defaultdict(float) for _ in range(copy.dim)]
        self_pair = stage == 1 and copy.prev == copy.src
        for zi, xi, yi, v in zip(
            zis.tolist(), xis.tolist(), yis.tolist(), val.tolist()
        ):
            key = (min(xi, yi), max(xi, yi)) if self_pair else (xi, yi)
            per_coord[zi - copy.start][key] += v
        # `x_a x_b` commutes: a self-pair's `(a, b)` and `(b, a)` merge
        # exactly, whatever their coefficients.
        terms = tuple(
            tuple((a, b, v) for (a, b), v in sorted(d.items()) if v != 0)
            for d in per_coord
        )
        materialized.append(Materialized(stage=stage, start=copy.start, terms=terms))
    return tuple(materialized), tuple(sorted(x_coords))


def _verify(plan: ReducedPlan, params, triples, rng) -> None:
    """Raise if the plan does not reproduce the unreduced contraction."""
    # avoid a circular import
    from e3j.pallas_ops.symmetric_contraction.common.unroll import emit_row_fwd

    n = 16
    x = rng.standard_normal((n, params.stage_dims[0]))
    evals = _evaluate(params, triples, x)
    w_full = [rng.standard_normal((n, b.m_out, b.m_in)) for b in params.blocks]
    ref = np.zeros((n, params.out_dim))
    for block, w in zip(params.blocks, w_full):
        for mo in range(block.m_out):
            for li in range(block.dim_l):
                acc = np.zeros(n)
                for mi in range(block.m_in):
                    stage, coord = block.input_map[mi * block.dim_l + li]
                    acc += w[:, mo, mi] * evals[stage][:, coord]
                ref[:, block.offset + mo * block.dim_l + li] = acc
    w_eff = {
        b.weight_index: np.einsum("nom,mj->noj", w_full[b.weight_index], b.fold)
        for b in plan.blocks
    }
    got = np.zeros((n, params.out_dim))
    rows = emit_row_fwd(
        plan,
        {k: x[:, k] for k in plan.x_coords},
        lambda block, mo, j: w_eff[block.weight_index][:, mo, j],
        np.zeros(n),
    )
    for k, row in rows.items():
        got[:, k] = row
    err = np.abs(got - ref).max()
    scale = max(np.abs(ref).max(), 1.0)
    if err > _VERIFY_TOLERANCE * scale:
        raise RuntimeError(
            f"reduced SymmetricContraction plan is off by {err:.3e} "
            f"(reference scale {scale:.3e})"
        )
