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
(:class:`BlockPlan`). Most features are linearly dependent: `x (x) x` is
symmetric, so `(v, u) -> L` is `+-(u, v) -> L` and `(u, u) -> L` vanishes
for odd `L`, and so on. At `l_max=3`, correlation 3 -> `0e+1o`, the 28 + 58
features span only 13 + 16 dimensions.

So for each `(block, degree)` the plan keeps a cheap spanning subset of the
features, and folds the rest into the weights:

```
y = sum_i W[i] F_i = sum_j (sum_i W[i] A[i, j]) F_sel_j = sum_j W_eff[j] F_sel_j
```

`A` is solved once at trace time, in float64, from random samples of the
features. The caller applies `W_eff = W @ A` (:func:`fold_weights`), and the
kernel only computes the selected features.

Last-power features sharing a CG block are then factorized:

```
sum_j W_eff[j] CG(P_prev[b_j], x_a) = CG(sum_j W_eff[j] P_prev[b_j], x_a)
```

so each group needs one CG contraction instead of one per feature.

The CG coefficients are float32, so dependencies only hold to ~1e-7; the
thresholds below sit well above that. The finished plan is checked against
the unreduced contraction in float64 before use.
"""

import functools
from collections import defaultdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from e3j.pallas_ops.symmetric_contraction.common.params import (
    PallasSymmetricContractionStructure,
)

#: A feature is selected if the part of it outside the span of those already
#: selected exceeds this fraction of its norm. Dependent features sit near
#: 1e-7 (float32 rounding), independent ones at 1e-1 or above.
_INDEPENDENCE_THRESHOLD = 1e-4

#: Largest relative float64 error the finished plan may have.
_VERIFY_TOLERANCE = 1e-5

#: After symmetrizing a self-pair, coefficients below this fraction of the
#: copy's largest are float32 residue and dropped.
_CANCELLATION_THRESHOLD = 1e-6

#: A feature whose norm is below this fraction of its norm without
#: cancellation (all terms taken positive) is identically zero, e.g.
#: `(u, u) -> L` for odd `L`. Exact zeros sit near 1e-16, real features at
#: 1e-2 and above.
_ZERO_THRESHOLD = 1e-6

#: Sample rows evaluated at once, to bound memory.
_EVAL_CHUNK = 256

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
    rng = np.random.default_rng(0)
    # Each sample gives `dim_l` values per feature: take enough samples for
    # twice the largest feature count.
    n_samples = max([64] + [-(-2 * b.m_in // b.dim_l) + 16 for b in params.blocks])
    x = rng.standard_normal((n_samples, params.stage_dims[0]))
    evals = _evaluate(params, triples, x)
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

    selected = _select(params, triples, copies, blocks_coef, evals, features, last)

    reduced = []
    weight_rows = 0
    for bi, block in enumerate(params.blocks):
        fold, direct, groups = _fold_and_factorize(
            block, features[bi], selected[bi], evals, copies, blocks_coef, last
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


def _evaluate(params, triples, x: np.ndarray, absolute: bool = False) -> list:
    """Every power at the samples `x`, each `(n, stage_dims[s])` float64.

    With `absolute`, inputs and coefficients are made positive, so nothing
    cancels.
    """
    if absolute:
        x = np.abs(x)
    out = [x]
    for s in range(1, params.correlation):
        zi, xi, yi, val = triples[s]
        if absolute:
            val = np.abs(val)
        # `reduceat` over the sorted terms: ~10x faster than a scatter-add.
        starts = np.flatnonzero(np.r_[True, zi[1:] != zi[:-1]])
        prev = out[-1]
        cur = np.zeros((x.shape[0], params.stage_dims[s]))
        for lo in range(0, x.shape[0], _EVAL_CHUNK):
            rows = slice(lo, lo + _EVAL_CHUNK)
            prod = val * prev[rows, xi] * x[rows, yi]
            cur[rows, zi[starts]] = np.add.reduceat(prod, starts, axis=1)
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


def _select(params, triples, copies, coef, evals, features, last):
    """Greedily pick a spanning subset of each `(block, degree)`'s features.

    Repeatedly take the remaining feature with the lowest marginal cost
    (its mix, any group it opens, and copies not yet materialized). Keep it
    if it is independent of the features already kept, else drop it for
    good. Returns `selected[block][stage] = [mi, ...]`.
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

    abs_evals = _evaluate(params, triples, evals[0], absolute=True)

    def is_zero(c):
        """Whether the copy is identically zero (`_ZERO_THRESHOLD`)."""
        span = slice(c.start, c.start + c.dim)
        scale = np.linalg.norm(abs_evals[c.stage][:, span])
        return np.linalg.norm(evals[c.stage][:, span]) <= _ZERO_THRESHOLD * scale

    bases = {}  # (block, stage) -> list of orthonormal rows
    selected = [defaultdict(list) for _ in params.blocks]
    remaining = [
        (bi, stage, mi, copy)
        for bi, by_stage in enumerate(features)
        for stage, feats in by_stage.items()
        for mi, copy in feats
        if not is_zero(copy)
    ]
    while remaining:
        remaining.sort(key=lambda f: (cost(f[0], f[3]), f[0], f[1], f[2]))
        for i, (bi, stage, mi, copy) in enumerate(remaining):
            vec = evals[stage][:, copy.start : copy.start + copy.dim].ravel()
            r = vec / np.linalg.norm(vec)
            for q in bases.get((bi, stage), ()):
                r = r - (q @ r) * q
            # Second Gram-Schmidt pass, for numerical stability.
            for q in bases.get((bi, stage), ()):
                r = r - (q @ r) * q
            res = np.linalg.norm(r)
            if res > _INDEPENDENCE_THRESHOLD:
                bases.setdefault((bi, stage), []).append(r / res)
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


def _fold_and_factorize(block, by_stage, selected, evals, copies, coef, last):
    """`(fold, direct, groups)` of one block, from its selected features."""
    columns = []  # (stage, mi, copy)
    for stage in sorted(by_stage):
        chosen = set(selected.get(stage, ()))
        for mi, copy in by_stage[stage]:
            if mi in chosen:
                columns.append((stage, mi, copy))
    fold = np.zeros((block.m_in, len(columns)))
    for stage, feats in by_stage.items():
        cols = [j for j, (s, _, _) in enumerate(columns) if s == stage]
        if not cols:
            continue

        def flat(copy, stage=stage):
            return evals[stage][:, copy.start : copy.start + copy.dim].ravel()

        basis = np.stack([flat(columns[j][2]) for j in cols])
        vecs = np.stack([flat(copy) for _, copy in feats])
        # Express every feature in terms of the selected ones.
        coeff, *_ = np.linalg.lstsq(basis.T, vecs.T, rcond=None)
        resid = np.linalg.norm(basis.T @ coeff - vecs.T, axis=0)
        limit = _VERIFY_TOLERANCE * np.maximum(np.linalg.norm(vecs, axis=1), 1.0)
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
        """The copy's group, or `None` if it is not a last-power feature."""
        if stage != last or last == 0:
            return None
        prev = copies[stage - 1][copy.prev]
        return (copy.src, prev.l, prev.p)

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
            cg = coef[stage][copy.index]
            if key not in grouped:
                grouped[key] = (cg, [])
            ref, members = grouped[key]
            alpha = float((cg * ref).sum() / (ref * ref).sum())
            if np.abs(cg - alpha * ref).max() > 1e-6 * np.abs(ref).max():
                # Not a multiple of the group's CG block: give it its own.
                key = (copy.src, prev.l, prev.p, j)
                grouped[key] = (cg, [])
                ref, members, alpha = cg, grouped[key][1], 1.0
            fold[:, j] *= alpha
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
        scale = float(np.abs(val).max())
        terms = tuple(
            tuple(
                (a, b, v)
                for (a, b), v in sorted(d.items())
                if abs(v) > _CANCELLATION_THRESHOLD * scale
            )
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
