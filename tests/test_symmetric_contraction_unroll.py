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

"""The row unroller's backward and `Dual` rows, on CPU.

The plan's `_verify` only checks :func:`emit_row_fwd`. Both unrollers only
add, subtract, negate and multiply rows, so here they run on float64
`(n,)` arrays: :func:`emit_row_bwd` is checked against `jax.vjp` of the
forward, and :class:`Dual` rows against `jax.jvp` of the backward.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    SYMMETRIC_CONTRACTION_CONFIG_IDS,
    SYMMETRIC_CONTRACTION_CONFIGS,
    unfused_symmetric_contraction,
)

from e3j.pallas_ops.symmetric_contraction.common.params import (
    BlockPlan,
    PallasSymmetricContractionStructure,
)
from e3j.pallas_ops.symmetric_contraction.common.plan import build_plan
from e3j.pallas_ops.symmetric_contraction.common.unroll import (
    Dual,
    emit_row_bwd,
    emit_row_fwd,
)

N = 8


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64():
        yield


def _structure(core) -> PallasSymmetricContractionStructure:
    """The static structure of `core`: its powers' Clebsch-Gordan triples, and
    which power coordinate feeds each input of each non-degenerate block
    (the filters, concatenation and `Permutation.sigma` resolved)."""
    filters = core.expansion.filter_layers
    layers = core.expansion.otimes_layers
    sigma = np.asarray(core.permutation.sigma)
    filter_indices = [np.asarray(f.indices) for f in filters]
    # where each power's filtered coordinates start in what `sigma` indexes
    offsets = np.cumsum([0] + [int(f.target.dim) for f in filters])

    blocks = []
    out_offset = 0
    for block in core.linear_blocks:
        m_out, m_in = block.shape
        block_offset = out_offset
        out_offset += m_out * (2 * block.l + 1)
        if m_in == 0 or m_out == 0:
            continue
        raw = sigma[block.begin : block.end]
        # `side="right"` steps over empty powers
        stages = np.searchsorted(offsets, raw, side="right") - 1
        input_map = tuple(
            (int(s), int(filter_indices[s][r - offsets[s]]))
            for s, r in zip(stages, raw)
        )
        blocks.append(BlockPlan(block.l, block.p, m_out, input_map, block_offset))

    return PallasSymmetricContractionStructure(
        stage_irreps=tuple(
            tuple((int(mul), int(ir.l), int(ir.p)) for mul, ir in space)
            for space in (core.source, *(layer.target for layer in layers))
        ),
        stage_indices=tuple(np.asarray(layer.coef.indices) for layer in layers),
        stage_values=tuple(np.asarray(layer.coef.data) for layer in layers),
        blocks=tuple(blocks),
        num_species=core.num_species,
        out_dim=core.target.dim,
    )


def _plan(case):
    core = unfused_symmetric_contraction(*case, num_species=1, num_channels=1)
    return build_plan(_structure(core))


def _rows(rng, keys) -> dict:
    return {k: jnp.asarray(rng.standard_normal(N)) for k in keys}


def _weight_keys(plan) -> list:
    return [
        (b.weight_index, mo, j)
        for b in plan.blocks
        for mo in range(b.m_out)
        for j in range(b.columns)
    ]


def _fwd(plan, x, w) -> dict:
    return emit_row_fwd(
        plan, x, lambda b, mo, j: w[b.weight_index, mo, j], jnp.zeros(N)
    )


def _bwd(plan, x, w, ct, with_dw=True) -> tuple:
    """`(dx, dw)`; coordinates the unroller leaves out are absent."""
    dw = {}

    def emit_dw(b, mo, j, value):
        key = (b.weight_index, mo, j)
        assert key not in dw, f"weight {key} received two cotangents"
        dw[key] = value

    dx = emit_row_bwd(
        plan,
        x,
        lambda b, mo, m: ct[b.offset + mo * b.dim + m],
        lambda b, mo, j: w[b.weight_index, mo, j],
        emit_dw if with_dw else None,
        jnp.zeros(N),
    )
    return dx, dw


def _assert_rows_close(got: dict, expected: dict):
    """Rows match, a missing row standing for zero."""
    assert set(got) <= set(expected)
    for k, row in expected.items():
        np.testing.assert_allclose(
            got.get(k, np.zeros(N)), row, rtol=1e-10, atol=1e-10, err_msg=str(k)
        )


@pytest.mark.parametrize(
    "case", SYMMETRIC_CONTRACTION_CONFIGS, ids=SYMMETRIC_CONTRACTION_CONFIG_IDS
)
def test_backward_matches_vjp(case):
    plan = _plan(case)
    rng = np.random.default_rng(0)
    x = _rows(rng, plan.x_coords)
    w = _rows(rng, _weight_keys(plan))
    out, pullback = jax.vjp(lambda x, w: _fwd(plan, x, w), x, w)
    ct = _rows(rng, out)
    dx_ref, dw_ref = pullback(ct)

    dx, dw = _bwd(plan, x, w, ct)
    _assert_rows_close(dx, dx_ref)
    _assert_rows_close(dw, dw_ref)

    # skipping the weight cotangents leaves `dx` as it was
    dx_only, dw_none = _bwd(plan, x, w, ct, with_dw=False)
    assert not dw_none
    _assert_rows_close(dx_only, dx_ref)


@pytest.mark.parametrize(
    "case", SYMMETRIC_CONTRACTION_CONFIGS, ids=SYMMETRIC_CONTRACTION_CONFIG_IDS
)
def test_dual_backward_matches_jvp(case):
    plan = _plan(case)
    rng = np.random.default_rng(1)
    x, w = _rows(rng, plan.x_coords), _rows(rng, _weight_keys(plan))
    ct = _rows(rng, _fwd(plan, x, w))
    tx, tw, tct = _rows(rng, x), _rows(rng, w), _rows(rng, ct)
    (dx, dw), (tdx_ref, tdw_ref) = jax.jvp(
        lambda x, w, ct: _bwd(plan, x, w, ct), (x, w, ct), (tx, tw, tct)
    )

    def dual(rows, tangents):
        return {k: Dual(v, tangents[k]) for k, v in rows.items()}

    dx_dual, dw_dual = _bwd(plan, dual(x, tx), dual(w, tw), dual(ct, tct))
    for got, primal, tangent in ((dx_dual, dx, tdx_ref), (dw_dual, dw, tdw_ref)):
        assert set(got) == set(primal)
        parts = {k: Dual.parts(v) for k, v in got.items()}
        _assert_rows_close({k: p for k, (p, _) in parts.items()}, primal)
        _assert_rows_close(
            {k: t for k, (_, t) in parts.items() if t is not None}, tangent
        )
