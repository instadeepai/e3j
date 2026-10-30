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

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    SYMMETRIC_CONTRACTION_CONFIG_IDS,
    SYMMETRIC_CONTRACTION_CONFIGS,
    symmetric_contraction_case,
    unfused_symmetric_contraction,
)

from e3j.pallas_ops.symmetric_contraction.common.plan import build_plan, fold_weights
from e3j.pallas_ops.symmetric_contraction.mosaic_tpu.fwd import (
    symmetric_contraction_reference,
)


@pytest.mark.parametrize(
    "case", SYMMETRIC_CONTRACTION_CONFIGS, ids=SYMMETRIC_CONTRACTION_CONFIG_IDS
)
def test_reduced_plan_matches_unfused(case):
    core, weights, x, index, expected = symmetric_contraction_case(*case)
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
