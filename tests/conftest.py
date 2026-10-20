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

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import pytest

E3J_OPS_TESTS = Path("tests/test_ops")

# Under `pytest -n N`, worker `gwK` gets TPU chip K to itself (TPU chips can't be shared).
_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
if _XDIST_WORKER is not None and "TPU_VISIBLE_DEVICES" not in os.environ:
    os.environ.update(
        TPU_VISIBLE_DEVICES=_XDIST_WORKER.removeprefix("gw"),
        TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1",
        TPU_PROCESS_BOUNDS="1,1,1",
    )


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
