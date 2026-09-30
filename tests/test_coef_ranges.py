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

"""Host-side coefficient partition in `e3j.ops.coef_ranges` (pure numpy)."""

import numpy
import pytest

from e3j.ops.coef_ranges import (
    _assign_rows,
    _critical_path,
    _pass_rows,
    balance_coef_ranges,
)


def _weights_summing_to(rng, total):
    """Random composition of `total` into positive row weights."""
    weights = []
    remaining = total
    while remaining > 0:
        w = int(rng.integers(1, remaining + 1))
        weights.append(w)
        remaining -= w
    return weights


def _random_passes(seed, num_coef, num_passes=3):
    """`num_passes` sorted `out_idx` arrays, each grouping `num_coef` coefs into rows.

    Mirrors the transposed backward coefficients (or a single-pass forward
    kernel, with `num_passes=1`): each output index is one contiguous, sorted
    run, but passes partition the coefs into rows independently.
    """
    rng = numpy.random.default_rng(seed)
    passes = []
    for _ in range(num_passes):
        weights = _weights_summing_to(rng, num_coef)
        out_idx = numpy.repeat(numpy.arange(len(weights)), weights)
        passes.append(out_idx.astype(numpy.int32))
    return passes


def _biggest_row(passes, num_coef):
    return max(w for oi in passes for _b, w in _pass_rows(oi, num_coef))


SEEDS = [0, 1, 2, 7, 42]
WARPS = [1, 2, 3, 4, 8]
NUM_PASSES = [1, 2, 3]
BARRIER = [False, True]
NUM_COEF = 40


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_warps", WARPS)
@pytest.mark.parametrize("num_passes", NUM_PASSES)
@pytest.mark.parametrize("barrier", BARRIER)
def test_permutations_are_bijections(seed, num_warps, num_passes, barrier):
    """Each per-pass permutation is a bijection of `range(num_coef)`."""
    passes = _random_passes(seed, NUM_COEF, num_passes)
    perms, _ = balance_coef_ranges(passes, NUM_COEF, num_warps, barrier)
    for perm in perms:
        assert perm.shape == (NUM_COEF,)
        numpy.testing.assert_array_equal(numpy.sort(perm), numpy.arange(NUM_COEF))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_warps", WARPS)
@pytest.mark.parametrize("num_passes", NUM_PASSES)
@pytest.mark.parametrize("barrier", BARRIER)
def test_endpoints_contiguous_and_cover(seed, num_warps, num_passes, barrier):
    """`coef_ranges` endpoints are non-decreasing and cover `num_coef` per pass."""
    passes = _random_passes(seed, NUM_COEF, num_passes)
    _, coef_ranges = balance_coef_ranges(passes, NUM_COEF, num_warps, barrier)
    assert coef_ranges.shape == (num_passes * num_warps,)
    for p in range(num_passes):
        endpoints = coef_ranges[p * num_warps : (p + 1) * num_warps]
        assert endpoints[-1] == NUM_COEF
        assert numpy.all(numpy.diff(endpoints) >= 0)
        assert endpoints[0] >= 0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_warps", WARPS)
@pytest.mark.parametrize("num_passes", NUM_PASSES)
@pytest.mark.parametrize("barrier", BARRIER)
def test_rows_contiguous_and_single_warp(seed, num_warps, num_passes, barrier):
    """Each output row stays contiguous and lands on exactly one warp.

    Reconstructs, per pass, the permuted `out_idx` and checks: (a) within-row
    order is preserved (each row maps to `arange(begin, begin + weight)`), and
    (b) each warp span in the permuted array is composed of whole rows.
    """
    passes = _random_passes(seed, NUM_COEF, num_passes)
    perms, coef_ranges = balance_coef_ranges(passes, NUM_COEF, num_warps, barrier)
    for p, out_idx in enumerate(passes):
        rows = _pass_rows(out_idx, NUM_COEF)
        perm = perms[p]
        endpoints = coef_ranges[p * num_warps : (p + 1) * num_warps]

        # Within-row order preserved: the permuted output index is constant
        # over each warp-major row block and equals the original run.
        permuted = out_idx[perm]
        cursor = 0
        for begin, weight in _row_blocks_in(perm, rows):
            numpy.testing.assert_array_equal(
                perm[cursor : cursor + weight],
                numpy.arange(begin, begin + weight),
            )
            block = permuted[cursor : cursor + weight]
            assert numpy.all(block == block[0])
            cursor += weight

        # Each warp span is a whole number of rows: its output indices are a
        # set disjoint from every other warp's.
        begins = numpy.concatenate([[0], endpoints[:-1]])
        seen = set()
        for b, e in zip(begins, endpoints):
            warp_rows = set(int(v) for v in permuted[b:e])
            assert warp_rows.isdisjoint(seen)
            seen |= warp_rows


def _row_blocks_in(perm, rows):
    """Yield `(begin, weight)` of each row in the order it appears in `perm`."""
    start_to_row = {begin: (begin, weight) for begin, weight in rows}
    i = 0
    while i < len(perm):
        begin = int(perm[i])
        block = start_to_row[begin]
        yield block
        i += block[1]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_warps", WARPS)
@pytest.mark.parametrize("num_passes", NUM_PASSES)
@pytest.mark.parametrize("barrier", BARRIER)
def test_critical_path_consistent_with_endpoints(seed, num_warps, num_passes, barrier):
    """The reported per-pass per-warp loads match the packed endpoints."""
    passes = _random_passes(seed, NUM_COEF, num_passes)
    _, endpoints, loads = _assign_rows(passes, NUM_COEF, num_warps, barrier)
    per_pass = numpy.diff(endpoints, prepend=0, axis=1)
    numpy.testing.assert_array_equal(loads, per_pass)
    expect = int(loads.max(axis=1).sum()) if barrier else int(loads.sum(axis=0).max())
    assert _critical_path(passes, NUM_COEF, num_warps, barrier) == expect


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_warps", WARPS)
@pytest.mark.parametrize("num_passes", NUM_PASSES)
def test_barrier_between_passes_balances_independently(seed, num_warps, num_passes):
    """Under a barrier, each pass is balanced as it would be alone."""
    passes = _random_passes(seed, NUM_COEF, num_passes)
    _, _, loads = _assign_rows(passes, NUM_COEF, num_warps, barrier_between_passes=True)
    for p, out_idx in enumerate(passes):
        _, _, alone = _assign_rows([out_idx], NUM_COEF, num_warps)
        numpy.testing.assert_array_equal(loads[p], alone[0])
    # Summing per-pass maxima cannot beat the maximum of summed loads.
    barrier_path = _critical_path(
        passes, NUM_COEF, num_warps, barrier_between_passes=True
    )
    assert barrier_path >= _critical_path(passes, NUM_COEF, num_warps)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_passes", NUM_PASSES)
def test_critical_path_approaches_ideal(seed, num_passes):
    """The critical path is bounded below by `max(total / num_warps, biggest_row)`,
    and reaches the biggest row with one warp per row."""
    passes = _random_passes(seed, NUM_COEF, num_passes)
    total = num_passes * NUM_COEF
    floor = _biggest_row(passes, NUM_COEF)

    for num_warps in [1, 2, 4, 8, 16, 32]:
        critical_path = _critical_path(passes, NUM_COEF, num_warps)
        assert critical_path >= floor
        assert critical_path >= -(-total // num_warps)  # ceil(total / num_warps)

    # Enough warps to isolate every row reaches the floor exactly.
    assert _critical_path(passes, NUM_COEF, num_passes * NUM_COEF) == floor
