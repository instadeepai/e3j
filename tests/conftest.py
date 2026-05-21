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

from pathlib import Path

import jax.numpy as np
import numpy
import numpy.testing as testing
import pytest
from jax.experimental import sparse

E3J_OPS_TESTS = Path("tests/test_ops")


def pytest_collection_modifyitems(config, items):
    """Skip all e3j_ops tests by default, as they require cuda + nvcc."""
    for item in items:
        test_path = Path(item.fspath)
        if config.rootdir / E3J_OPS_TESTS in test_path.parents:
            item.add_marker(pytest.mark.e3j_ops)


def assert_allclose(expect, result, rtol=5e-6, atol=5e-6, debug: int = 1):
    """Show more on assertion errors to help diagnose addressing errors."""
    try:
        testing.assert_allclose(result, expect, rtol=rtol, atol=atol)
    except AssertionError as err:
        if debug >= 1:
            print("expect == result\n", abs(expect - result) < atol)
        if debug >= 2:
            print("expect\n", expect)
            print("result\n", result)
        raise err


class InBoundsBCOO(sparse.BCOO):
    """Subclass BCOO to enforce in-bound checks.

    JAX doesn't do this by itself, and the following is valid:

        >>> coef = np.ones(3)
        >>> idx = np.array([[0, 1], [2, 3], [20, 42]])
        >>> matrix = sparse.BCOO((coef, idx), shape=(4, 4))
        ### This works too
        >>> matrix.coalesce(), matrix.sort_indices()
    """

    def __init__(self, data, *, shape):
        super().__init__(data, shape=shape)
        # perform additional checks
        coef, idx = data
        idx_min = np.min(idx, axis=0)
        idx_max = np.max(idx, axis=0)
        bounds = np.array(shape)
        if not np.prod(idx_min >= 0):
            raise ValueError(f"Negative indices: {idx_min.tolist()} !>= 0")
        if not np.prod(idx_max < bounds):
            raise ValueError(f"Out of bounds: {idx_max.tolist()} !< {shape} ")


@pytest.fixture(autouse=True)
def _mock_BCOO(monkeypatch: pytest.MonkeyPatch):  # noqa: N802
    """Mock sparse.BCOO with in-bound checks."""
    monkeypatch.setattr(sparse, "BCOO", InBoundsBCOO)
