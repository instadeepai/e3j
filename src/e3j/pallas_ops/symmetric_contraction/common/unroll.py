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

"""The reduced plan (:mod:`.plan`) for one batch row, unrolled at trace time.

:func:`emit_row_fwd` and :func:`emit_row_bwd` are shared by every kernel.
They work on whatever "row" values the kernel gives them (registers on
Mosaic GPU, `(batch, 128)` VMEM tiles on Mosaic TPU); loads and stores go
through callbacks.

They only add, subtract, negate and multiply, so they also run on
:class:`Dual` rows. The Mosaic GPU double backward uses this to
differentiate them in forward mode.
"""

import jax

from e3j.pallas_ops.symmetric_contraction.common.plan import ReducedPlan
from e3j.pallas_ops.utils.mosaic_gpu import add_term, sum_terms


class Dual:
    """A row and its forward-mode derivative.

    `tangent=None` is a zero derivative, and plain rows act as constants, so
    only derivatives that can be nonzero are computed.
    """

    __slots__ = ("primal", "tangent")

    def __init__(self, primal, tangent=None):
        self.primal = primal
        self.tangent = tangent

    @staticmethod
    def parts(value) -> tuple:
        """`(primal, tangent)` of a `Dual` or a plain row."""
        if isinstance(value, Dual):
            return value.primal, value.tangent
        return value, None

    def __add__(self, other):
        a, da = Dual.parts(self)
        b, db = Dual.parts(other)
        return Dual(a + b, add_term(da, db))

    __radd__ = __add__

    def __sub__(self, other):
        a, da = Dual.parts(self)
        b, db = Dual.parts(other)
        return Dual(a - b, add_term(da, None if db is None else -db))

    def __rsub__(self, other):
        return Dual(other) - self

    def __neg__(self):
        a, da = Dual.parts(self)
        return Dual(-a, None if da is None else -da)

    def __mul__(self, other):
        a, da = Dual.parts(self)
        if isinstance(other, (int, float)):
            return Dual(a * other, None if da is None else da * other)
        b, db = Dual.parts(other)
        return Dual(
            a * b,
            add_term(None if da is None else da * b, None if db is None else a * db),
        )

    __rmul__ = __mul__


def _scaled(value: float, row: jax.Array) -> jax.Array:
    """`value * row`, skipping a multiply by one."""
    return row if value == 1.0 else value * row


def _add_signed(total: jax.Array | None, term: jax.Array, sign: float) -> jax.Array:
    """`total + sign * term` for `sign` in {1, -1}; `None` is an empty sum.

    Adding or subtracting (rather than multiplying by the sign) lets the
    compiler fuse it into a single FMA.
    """
    if total is None:
        return term if sign > 0 else -term
    return total + term if sign > 0 else total - term


def _grouped(pairs):
    """Group `(key, item, value)` triples into `[(key, [(item, value), ...])]`,
    keys in first-seen order."""
    out: dict = {}
    for key, item, value in pairs:
        out.setdefault(key, []).append((item, value))
    return list(out.items())


class _LinearForms:
    """Cache of linear forms `sum_k value * x[k]` over one row.

    Forms are keyed up to sign, so `L` and `-L` share one entry; the sign is
    applied by the consumer (:func:`_add_signed`).
    """

    def __init__(self, x: dict):
        self.x = x
        self.cache: dict = {}

    @staticmethod
    def _key(terms) -> tuple[tuple, float]:
        terms = sorted(terms)
        sign = 1.0 if terms[0][1] > 0 else -1.0
        return tuple((k, v * sign) for k, v in terms), sign

    def cost(self, terms) -> int:
        """FP ops to build a form: 0 if cached or a single unit term."""
        key, _ = self._key(terms)
        if key in self.cache or (len(key) == 1 and key[0][1] == 1.0):
            return 0
        return len(key)

    def __call__(self, terms) -> tuple[jax.Array, float]:
        """`(form, sign)` with `sign * form == sum_k value * x[k]`."""
        key, sign = self._key(terms)
        if key not in self.cache:
            total = None
            for k, v in key:
                total = _add_signed(
                    total, _scaled(abs(v), self.x[k]), 1.0 if v > 0 else -1.0
                )
            self.cache[key] = total
        return self.cache[key], sign


def _second_power_grouping(terms) -> list:
    """Group a second-power coordinate's `(a, b, value)` terms by whichever
    operand gives fewer groups, each group one `x[a] * L(x[b])`."""
    by_a = _grouped((a, b, v) for a, b, v in terms)
    by_b = _grouped((b, a, v) for a, b, v in terms)
    return by_a if len(by_a) <= len(by_b) else by_b


def materialize_powers(
    plan: ReducedPlan, x: dict, zero_row: jax.Array
) -> tuple[list[dict], _LinearForms]:
    """Compute one row's intermediate copies; returns `(vals, forms)`.

    `vals[s]` maps each materialized coordinate of power `s` to its row
    (`vals[0]` is `x`). `forms` caches the linear forms of `x` built here,
    for the caller to reuse. Each coordinate is a sum of `prev[a] * L` over
    linear forms `L` of `x`.
    """
    vals = [x] + [{} for _ in range(plan.last)]
    forms = _LinearForms(x)
    for mat in plan.materialized:
        prev = vals[mat.stage - 1]
        for m, terms in enumerate(mat.terms):
            groups = (
                _second_power_grouping(terms) if mat.stage == 1 else _grouped(terms)
            )
            total = None
            for a, bv in groups:
                form, sign = forms(bv)
                total = _add_signed(total, prev[a] * form, sign)
            vals[mat.stage][mat.start + m] = zero_row if total is None else total
    return vals, forms


def _bilinear(pairs, left_forms: _LinearForms, right_forms: _LinearForms):
    """`sum v * left[p] * right[q]` over `pairs`, `None` if empty.

    Computed as either `sum_p left[p] * R_p` or `sum_q right[q] * L_q`
    (linear forms of the other side), whichever costs fewer FP ops.
    """
    by_p = _grouped(pairs)
    by_q = _grouped((q, p, v) for p, q, v in pairs)
    cost_p = sum(right_forms.cost(qv) + 1 for _, qv in by_p)
    cost_q = sum(left_forms.cost(pv) + 1 for _, pv in by_q)
    total = None
    if cost_p <= cost_q:
        for p, qv in by_p:
            form, sign = right_forms(qv)
            total = _add_signed(total, left_forms.x[p] * form, sign)
    else:
        for q, pv in by_q:
            form, sign = left_forms(pv)
            total = _add_signed(total, right_forms.x[q] * form, sign)
    return total


def _group_sum(group, ws, prev) -> dict:
    """`R[i] = sum_j ws[j] * prev[start_j + i]` over the group's members."""
    r = {}
    for (_, start), w in zip(group.members, ws):
        for i in range(group.prev_dim):
            r[i] = add_term(r.get(i), w * prev[start + i])
    return r


def emit_row_fwd(plan: ReducedPlan, x: dict, weight, zero_row) -> dict:
    """Unroll one row's forward; returns `{output coordinate: row}`.

    `weight(block, mo, j)` loads a folded weight row.
    """
    vals, forms = materialize_powers(plan, x, zero_row)
    out = {}
    for block in plan.blocks:
        for mo in range(block.m_out):
            acc = [None] * block.dim
            for j, stage, start in block.direct:
                w = weight(block, mo, j)
                for m in range(block.dim):
                    acc[m] = add_term(acc[m], w * vals[stage][start + m])
            for group in block.groups:
                ws = (weight(block, mo, j) for j, _ in group.members)
                r_forms = _LinearForms(_group_sum(group, ws, vals[plan.last - 1]))
                src = group.src_start
                # out[m] += sum v * R[i] * x[src + k]
                for m, ik in _grouped(
                    (m, (i, src + k), v) for m, i, k, v in group.terms
                ):
                    term = _bilinear([(i, k, v) for (i, k), v in ik], r_forms, forms)
                    acc[m] = add_term(acc[m], term)
            for m, total in enumerate(acc):
                coord = block.offset + mo * block.dim + m
                out[coord] = zero_row if total is None else total
    return out


def emit_row_bwd(plan: ReducedPlan, x: dict, ct, weight, emit_dw, zero_row) -> dict:
    """Unroll one row's backward; returns `dx` as `{source coordinate: row}`.

    `ct(block, mo, m)` loads an output cotangent, `weight(block, mo, j)` a
    folded weight row, and `emit_dw(block, mo, j, value)` receives that
    weight's cotangent. With `emit_dw=None`, weight cotangents are skipped.
    """
    vals, forms = materialize_powers(plan, x, zero_row)
    # adj[s]: cotangents of power `s` so far; `adj[0]` is `dx`.
    adj = [{} for _ in vals]

    def accumulate(stage, k, term, sign=1.0):
        adj[stage][k] = _add_signed(adj[stage].get(k), term, sign)

    for block in plan.blocks:
        for mo in range(block.m_out):
            cts = {m: ct(block, mo, m) for m in range(block.dim)}
            ct_forms = _LinearForms(cts)
            # Direct features: out[m] += w * P[start + m].
            for j, stage, start in block.direct:
                if emit_dw is not None:
                    emit_dw(
                        block,
                        mo,
                        j,
                        sum_terms(c * vals[stage][start + m] for m, c in cts.items()),
                    )
                w = weight(block, mo, j)
                for m, c in cts.items():
                    accumulate(stage, start + m, w * c)
            # Groups: out[m] += sum v * R[i] * x[src + k], with
            # R = sum_j w_j * P_prev[start_j:].
            prev = vals[plan.last - 1]
            for group in block.groups:
                ws = [weight(block, mo, j) for j, _ in group.members]
                r_forms = _LinearForms(_group_sum(group, ws, prev))
                src = group.src_start
                # dR[i] = sum v * ct[m] * x[src + k]
                d_r = {
                    i: _bilinear([(m, k, v) for (m, k), v in mk], ct_forms, forms)
                    for i, mk in _grouped(
                        (i, (m, src + k), v) for m, i, k, v in group.terms
                    )
                }
                # dx[src + k] += sum v * ct[m] * R[i]
                for k, mi in _grouped(
                    (src + k, (m, i), v) for m, i, k, v in group.terms
                ):
                    accumulate(
                        0,
                        k,
                        _bilinear([(m, i, v) for (m, i), v in mi], ct_forms, r_forms),
                    )
                for (j, start), w in zip(group.members, ws):
                    if emit_dw is not None:
                        emit_dw(
                            block,
                            mo,
                            j,
                            sum_terms(d * prev[start + i] for i, d in d_r.items()),
                        )
                    for i, d in d_r.items():
                        accumulate(plan.last - 1, start + i, w * d)

    # Walk the copies last power first, so each power's cotangent is
    # complete before it is propagated.
    stage_forms = {1: forms}
    for mat in reversed(plan.materialized):
        prev = vals[mat.stage - 1]
        if mat.stage not in stage_forms:
            stage_forms[mat.stage] = _LinearForms(prev)
        prev_forms = stage_forms[mat.stage]
        for m, terms in enumerate(mat.terms):
            g = adj[mat.stage].get(mat.start + m)
            if g is None:
                continue
            # coordinate = sum v * prev[a] * x[b]
            for a, bv in _grouped(terms):
                form, sign = forms(bv)
                accumulate(mat.stage - 1, a, g * form, sign)
            for b, av in _grouped((b, a, v) for a, b, v in terms):
                form, sign = prev_forms(av)
                accumulate(0, b, g * form, sign)
    return adj[0]
