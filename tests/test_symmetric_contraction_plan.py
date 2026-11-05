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

"""The reduced plan the fused kernels walk, on CPU.

:mod:`e3j.pallas_ops.symmetric_contraction.common.plan` is plain NumPy, so
the plan and its folded weights are checked against `UNFUSED` here without
an accelerator, through the kernels' own unroll in the pure-JAX reference.
"""

import functools
import itertools
from collections import Counter

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    SYMMETRIC_CONTRACTION_CONFIG_IDS,
    SYMMETRIC_CONTRACTION_CONFIGS,
    symmetric_contraction_case,
    unfused_symmetric_contraction,
)

from e3j.pallas_ops.symmetric_contraction.common import plan as plan_module
from e3j.pallas_ops.symmetric_contraction.common.plan import build_plan, fold_weights
from e3j.pallas_ops.symmetric_contraction.mosaic_tpu.fwd import (
    symmetric_contraction_reference,
)


@functools.cache
def _sym_weights(l: int, n: int) -> Counter:
    """How many basis states of `Sym^n(V_l)` have each total weight `m`."""
    return Counter(
        sum(ms) for ms in itertools.combinations_with_replacement(range(-l, l + 1), n)
    )


@functools.cache
def _multiplicity(ls: tuple[tuple[int, int], ...], l: int) -> int:
    """How many times `V_l` appears in `(x)_i Sym^n_i(V_l_i)` over the `(l_i,
    n_i)` of `ls`: the number of independent degree-`l` features of a leaf
    multiset."""
    weights = Counter({0: 1})
    for li, n in ls:
        nxt = Counter()
        for m, count in weights.items():
            for mi, ci in _sym_weights(li, n).items():
                nxt[m + mi] += count * ci
        weights = nxt
    # An irrep `V_L` contributes one state to each weight in `[-L, L]`.
    return weights[l] - weights[l + 1]


#: `l_max` prunes intermediate paths, so the features may span less than the
#: plethysm counts: at `l_max = 1`, `{2e, 2e, 3o}` and `{1o, 3o, 3o}` -> `1o`
#: keep 1 of 2 (:func:`test_pruned_plan_keeps_less_than_the_plethysm`).
PRUNED_CONFIGS = [
    ("0e+1o+2e", 3, "0e+1o", 1),
    ("0e+1o+2e+3o", 3, "0e+1o", 1),
    ("0e+1o+2e+3o", 3, "0e+1o", 2),
]


@pytest.mark.parametrize(
    "case",
    SYMMETRIC_CONTRACTION_CONFIGS + PRUNED_CONFIGS,
    ids=SYMMETRIC_CONTRACTION_CONFIG_IDS
    + [f"{s}_corr{c}_{o}_lmax{l}" for s, c, o, l in PRUNED_CONFIGS],
)
def test_reduced_plan_matches_unfused(case):
    source, correlation, target, *l_max = case
    core, weights, x, index, expected = symmetric_contraction_case(
        source, correlation, target, l_max=l_max[0] if l_max else None
    )
    params = core.mosaic_tpu_params
    kernel_weights = core._kernel_weights(
        [jnp.asarray(w, jnp.float32) for w in weights]
    )
    folded = fold_weights(kernel_weights, build_plan(params))
    got = symmetric_contraction_reference(
        jnp.asarray(x, jnp.float32), folded, jnp.asarray(index), params
    )
    # the tolerance is the float32 reference
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_reduced_plan_size():
    """At `l_max = 3 -> 0e + 1o`, correlation 3, the 28 + 58 features of the
    two blocks span 13 + 16 dimensions: the folded weight table's rows."""
    core = unfused_symmetric_contraction("0e+1o+2e+3o", 3, "0e+1o", 1, 1)
    assert build_plan(core.mosaic_gpu_params).weight_rows == 13 + 16


@pytest.mark.parametrize(
    "l, n, expected",
    [
        # Sym^2(V_l) holds the even L <= 2l.
        (1, 2, {0: 1, 2: 1}),
        (3, 2, {0: 1, 2: 1, 4: 1, 6: 1}),
        (1, 3, {1: 1, 3: 1}),
        (2, 3, {0: 1, 2: 1, 3: 1, 4: 1, 6: 1}),
        (1, 4, {0: 1, 2: 1, 4: 1}),
        # Multiplicities above one, already within `l <= 3`.
        (3, 3, {1: 1, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1, 9: 1}),
        (2, 4, {0: 1, 2: 2, 4: 2, 5: 1, 6: 1, 8: 1}),
        (4, 3, {0: 1, 2: 1, 3: 1, 4: 2, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1, 10: 1, 12: 1}),
    ],
)
def test_multiplicity_of_symmetric_powers(l, n, expected):
    got = {L: _multiplicity(((l, n),), L) for L in range(n * l + 1)}
    assert {L: m for L, m in got.items() if m} == expected


def test_multiplicity_of_mixed_multisets():
    """Sym^2(V_1) (x) V_2 = (V_0 + V_2) (x) V_2 = V_2 + (V_0 + ... + V_4)."""
    got = [_multiplicity(((1, 2), (2, 1)), L) for L in range(6)]
    assert got == [1, 1, 2, 1, 1, 0]


def _kept_and_plethysm(params) -> dict:
    """`{(block, leaves): (kept, plethysm count)}` over every leaf multiset of
    every block's features."""
    plan = build_plan(params)
    copies, coef = plan_module._structure(params, plan_module._triples(params))
    poly = plan_module._polynomials(copies, coef)
    copy_at = [{c.start: c for c in stage} for stage in copies]
    leaf_ls = {c.index: c.l for c in copies[0]}

    def leaves(block, mi):
        stage, coord = block.input_map[mi * block.dim_l]
        return poly(stage, copy_at[stage][coord].index).leaves

    out = {}
    for block in plan.blocks:
        source_block = params.blocks[block.weight_index]
        kept = Counter()
        for column in range(block.columns):
            # Fold columns never mix multisets: any feature a column reads
            # has the leaves of the column's own.
            mi = int(np.flatnonzero(block.fold[:, column])[0])
            kept[leaves(source_block, mi)] += 1
        for key in {leaves(source_block, mi) for mi in range(source_block.m_in)}:
            ls = tuple(sorted((leaf_ls[a], n) for a, n in Counter(key).items()))
            count = _multiplicity(ls, source_block.l)
            out[(block.weight_index, key)] = (kept[key], count)
    return out


@pytest.mark.parametrize(
    "source, correlation, target",
    [("0e+1o+2e+3o", 3, "0e+1o"), ("0e+1o+2e", 4, "0e+1o"), ("2x0e+1o", 3, "0e+1o")],
)
def test_reduced_plan_keeps_the_plethysm_counts(source, correlation, target):
    """Without `l_max`, every leaf multiset keeps exactly as many features as
    its symmetric power holds the block's irrep."""
    params = unfused_symmetric_contraction(
        source, correlation, target, 1, 1
    ).mosaic_tpu_params
    for key, (kept, count) in _kept_and_plethysm(params).items():
        assert kept == count, key


def test_pruned_plan_keeps_less_than_the_plethysm():
    """With `l_max = 1`, the paths left to `{2e, 2e, 3o}` and `{1o, 3o, 3o}`
    -> `1o` (source copies 2, 2, 3 and 1, 3, 3) span 1 of the 2 features:
    the rank test, not the count, stops their selection."""
    params = unfused_symmetric_contraction(
        "0e+1o+2e+3o", 3, "0e+1o", 1, 1, l_max=1
    ).mosaic_tpu_params
    short = {
        key: (kept, count)
        for key, (kept, count) in _kept_and_plethysm(params).items()
        if kept != count
    }
    assert short == {(1, (2, 2, 3)): (1, 2), (1, (1, 3, 3)): (1, 2)}


def test_plan_needs_no_samples(monkeypatch):
    """Selection and fold read the CG tables only: sampling the powers is
    left to the final check."""
    calls = []
    evaluate = plan_module._evaluate
    monkeypatch.setattr(
        plan_module,
        "_evaluate",
        lambda *args, **kwargs: calls.append(1) or evaluate(*args, **kwargs),
    )
    params = unfused_symmetric_contraction(
        "0e+1o+2e+3o", 3, "0e+1e", 1, 1
    ).mosaic_tpu_params
    build_plan.cache_clear()
    build_plan(params)
    # `_verify`'s one call.
    assert len(calls) == 1
