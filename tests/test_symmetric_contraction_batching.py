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

"""The vmap folding shared by the fused symmetric contraction backends, on a
plain JAX stand-in for a kernel (so it runs on any backend)."""

import jax
import jax.numpy as jnp
import numpy.testing as testing
import pytest

from e3j.pallas_ops.symmetric_contraction.common.batching import vmappable

NUM_SPECIES, BATCH, CHANNELS = 3, 5, 4


def _kernel(w, species, x, ct):
    """A stand-in kernel: `y = x * w[species]`, and the backward `(dx, dw)`
    of `<y, ct>`, `dw` summed per species."""
    s = jnp.clip(
        jnp.where(species < 0, species + w.shape[0], species), 0, w.shape[0] - 1
    )
    y = x * w[s]
    dw = jax.ops.segment_sum(x * ct, s, num_segments=w.shape[0])
    return y, (ct * w[s], dw)


def _operands(v):
    keys = jax.random.split(jax.random.key(0), 3)
    w = jax.random.normal(keys[0], (v, NUM_SPECIES, CHANNELS))
    species = jax.random.randint(keys[1], (v, BATCH), -NUM_SPECIES, NUM_SPECIES)
    x, ct = jax.random.normal(keys[2], (2, v, BATCH, CHANNELS))
    return w, species, x, ct


@pytest.mark.parametrize(
    "in_axes",
    [(None, None, 0, 0), (0, None, None, None), (None, 0, None, 0), (0, 0, 0, 0)],
    ids=["rows", "weights", "species", "all"],
)
@pytest.mark.parametrize("max_species", [None, 2 * NUM_SPECIES, NUM_SPECIES])
def test_vmap_matches_plain_vmap(in_axes, max_species):
    """Any batched subset, the weights folded in chunks of lanes whose
    species `fits`, down to one lane per call."""
    v = 5
    args = [a if axis == 0 else a[0] for a, axis in zip(_operands(v), in_axes)]
    fits = None if max_species is None else (lambda n: n <= max_species)
    calls = []

    def kernel(*a):
        calls.append(a[0].shape[0])
        return _kernel(*a)

    got = jax.jit(jax.vmap(vmappable(kernel, tile_w=True, fits=fits), in_axes))(*args)
    expect = jax.vmap(_kernel, in_axes)(*args)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expect)):
        assert a.shape == b.shape
        testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    assert max(calls) <= (max_species or v * NUM_SPECIES)


def test_nested_vmap_folds_both_axes():
    w, species, x, ct = _operands(6)
    args = [a.reshape(2, 3, *a.shape[1:]) for a in (w, species, x, ct)]
    calls = []

    def kernel(*a):
        calls.append(a[2].shape[0])
        return _kernel(*a)

    got = jax.jit(jax.vmap(jax.vmap(vmappable(kernel, tile_w=True))))(*args)
    expect = jax.vmap(jax.vmap(_kernel))(*args)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expect)):
        testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    # The last call, the one lowered, runs on every row (the others trace
    # each layer's unbatched function).
    assert calls[-1] == 6 * BATCH


def test_unbatched_call_is_the_kernel():
    w, species, x, ct = (a[0] for a in _operands(1))
    got = vmappable(_kernel)(w, species, x, ct)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(_kernel(w, species, x, ct))):
        testing.assert_allclose(a, b)
