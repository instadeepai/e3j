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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from e3j.ops import scatter_add_1
from e3j.utils.sparse import narrow_index_dtype


def generate_test_data():
    """
    Generate common test inputs for the forward and backward tests.

    Returns:
        idx: A 1D index array.
        src: A 2D source array with shape (ROWS, num_idx).
        out_init: A 2D zero array with shape (ROWS, num_out), where num_out = max(idx) + 1.
    """
    ROWS = 8
    NUM_OUT = 240

    idx = jnp.arange(NUM_OUT)
    idx = jnp.repeat(idx, idx % 6)
    N = idx.shape[0]

    base_val = jnp.ones(N, dtype=jnp.float32)

    factors = 10 ** (np.arange(ROWS) % 3)
    factors = jnp.array(factors, dtype=jnp.float32).reshape((ROWS, 1))
    src = base_val * factors

    num_segments = int(jnp.max(idx)) + 1
    out_init = jnp.zeros((ROWS, num_segments), dtype=jnp.float32)

    return idx, src, out_init


def scatter_add_reference(idx, src, out):
    """
    Reference implementation of scatter-add using jax.ops.segment_sum row-wise.

    Args:
        idx: A 1D index array.
        src: A 2D source array of shape (ROWS, N).
        out: A 2D output array of zeros with shape (ROWS, num_out).

    Returns:
        A 2D array with the scatter-add result.
    """

    def row_segment_sum(row):
        return jax.ops.segment_sum(row, idx, num_segments=int(jnp.max(idx)) + 1)

    return jax.vmap(row_segment_sum)(src)


def test_forward_scatter_add():
    """
    Validate that the forward pass of scatter_add_1 matches the reference scatter-add.
    """
    idx, src, out_init = generate_test_data()

    out_custom = scatter_add_1(idx, src, out_init)
    out_ref = scatter_add_reference(idx, src, out_init)

    np.testing.assert_allclose(out_custom, out_ref, rtol=1e-5, atol=1e-5)


def test_backward_scatter_add():
    """
    Validate that the gradients computed by scatter_add_1 match those computed by the reference.

    Both functions simply sum their outputs so that the gradient with respect to each
    source element should be 1.
    """
    idx, src, out_init = generate_test_data()

    def f_custom(src):
        return jnp.sum(scatter_add_1(idx, src, out_init))

    def f_ref(src):
        return jnp.sum(scatter_add_reference(idx, src, out_init))

    grad_custom = jax.grad(f_custom)(src)
    grad_ref = jax.grad(f_ref)(src)

    np.testing.assert_allclose(grad_custom, grad_ref, rtol=1e-5, atol=1e-5)


def test_forward_scatter_add_narrow_idx():
    """Validate forward pass with narrow (uint8) index dtype."""
    idx, src, out_init = generate_test_data()
    idx_narrow = idx.astype(narrow_index_dtype((int(jnp.max(idx)) + 1,)))
    assert idx_narrow.dtype == jnp.uint8

    out_custom = scatter_add_1(idx_narrow, src, out_init)
    out_ref = scatter_add_reference(idx, src, out_init)
    np.testing.assert_allclose(out_custom, out_ref, rtol=1e-5, atol=1e-5)


def test_backward_scatter_add_narrow_idx():
    """Validate backward pass with narrow (uint8) index dtype."""
    idx, src, out_init = generate_test_data()
    idx_narrow = idx.astype(narrow_index_dtype((int(jnp.max(idx)) + 1,)))

    def f_custom(src):
        return jnp.sum(scatter_add_1(idx_narrow, src, out_init))

    def f_ref(src):
        return jnp.sum(scatter_add_reference(idx, src, out_init))

    grad_custom = jax.grad(f_custom)(src)
    grad_ref = jax.grad(f_ref)(src)
    np.testing.assert_allclose(grad_custom, grad_ref, rtol=1e-5, atol=1e-5)


def test_forward_scatter_add_multi_strip_single_index():
    """
    Regression test stale-prefetch OOB lanes in the last strip of a
    multi-strip row (num_idx > blockDim.x == 256) used to keep valid (idx, val) pairs
    from an earlier strip, corrupting the warp reduction and over-counting the result.

    Global sum-pooling of 300 elements into a single output index should give 300.
    """
    num_idx = 300  # > blockDim.x (256) => two strips, triggers the bug
    idx = jnp.zeros(num_idx, dtype=jnp.int32)
    src = jnp.ones((1, num_idx), dtype=jnp.float32)
    out_init = jnp.zeros((1, 1), dtype=jnp.float32)

    out_custom = scatter_add_1(idx, src, out_init)
    out_ref = scatter_add_reference(idx, src, out_init)

    np.testing.assert_allclose(out_custom, out_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out_custom, jnp.array([[float(num_idx)]]), rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
    test_forward_scatter_add()
    test_backward_scatter_add()
    test_forward_scatter_add_narrow_idx()
    test_backward_scatter_add_narrow_idx()
    print("All tests passed!")
