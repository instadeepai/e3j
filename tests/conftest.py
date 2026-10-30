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

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as testing
import pytest

from e3j.core.symmetric_contraction import SymmetricContraction
from e3j.utils import config, options

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


def skip_unless_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("pallas_mosaic_tpu tests require a TPU backend")


# (source_irreps, correlation, keep_irrep_out): every block non-degenerate
SYMMETRIC_CONTRACTION_CONFIGS = [
    ("0e+1o+2e", 1, "0e+1o"),
    ("0e+1o+2e", 2, "0e+1o"),
    ("0e+1o", 2, "0e+1o+2e"),
    ("0e+1o+2e", 2, "0e+1o+2e"),
    ("0e+1o", 3, "0e+1o"),
    ("0e+1o+2e+3o", 3, "0e+1o"),
    ("0e+1o+2e+3o", 3, "0e+1o+2e"),
    ("0e+1o+2e+3o", 3, "0e"),
    ("1x0e+2x1o+1x2e", 2, "0e+1o"),
    ("0e+1o", 4, "0e+1o"),
]
SYMMETRIC_CONTRACTION_CONFIG_IDS = [
    f"{s}_corr{c}_{o}" for s, c, o in SYMMETRIC_CONTRACTION_CONFIGS
]


def unfused_symmetric_contraction(
    source_irreps, correlation, keep_irrep_out, num_species, num_channels
):
    with config.use(
        symmetric_contraction=options.SymmetricContraction.UNFUSED,
        tensor_product=options.TensorProduct.UNFUSED,
    ):
        return SymmetricContraction(
            source_irreps=source_irreps,
            correlation=correlation,
            keep_irrep_out=keep_irrep_out,
            num_species=num_species,
            num_channels=num_channels,
            layout="TRAILING_CHANNELS",
        )


def symmetric_contraction_case(
    source_irreps, correlation, keep_irrep_out, num_species=3, num_channels=2, batch=40
):
    """`(core, weights, x, index, expected)`: NumPy float64 operands and the
    float32 `UNFUSED` output."""
    core = unfused_symmetric_contraction(
        source_irreps, correlation, keep_irrep_out, num_species, num_channels
    )
    rng = np.random.default_rng(0)
    weights = [
        rng.normal(size=(num_species, num_channels, *b.shape))
        for b in core.linear_blocks
    ]
    x = rng.normal(size=(batch, core.source.dim, num_channels))
    index = rng.integers(0, num_species, batch)
    expected = np.asarray(
        core(
            jnp.asarray(x, jnp.float32),
            jnp.asarray(index),
            [jnp.asarray(w, jnp.float32) for w in weights],
        )
    )
    return core, weights, x, index, expected
