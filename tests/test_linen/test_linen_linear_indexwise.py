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

import e3nn_jax as e3nn
import jax
import pytest
from conftest import assert_allclose
from jax import numpy as np
from jax import random

from e3j.linen.linear_indexwise import LinearIndexwise


class _TestLinearIndexwise:
    """Base class for linear module tests."""

    out: str
    in1: str
    num_indices: int = 8
    num_channels: int = 16
    batch_size: int = 32
    layout: str = "LEADING_CHANNELS"
    _seed: int = 42

    @property
    def key(self) -> jax.Array:
        return random.key(self._seed)

    @pytest.fixture
    def inputs(self):
        """Arrange inputs for Linear modules."""
        rep_in = e3nn.Irreps(self.in1)
        input_features = random.normal(
            self.key, (self.batch_size, self.num_channels, rep_in.dim)
        )
        input_indices = random.choice(self.key, self.num_indices, (self.batch_size,))
        return (input_features, input_indices)

    @pytest.fixture(scope="class")
    def e3j_module(self):
        return LinearIndexwise(
            self.in1, self.out, self.num_indices, self.num_channels, layout=self.layout
        )

    @pytest.fixture
    def params(self, e3j_module, inputs):
        """Arrange parameters for Linear modules."""
        return e3j_module.init(self.key, *inputs)

    @pytest.fixture(scope="class")
    def rotations(self, e3j_module) -> list[np.ndarray]:
        """Prepare output and input rotations."""
        in_irreps = e3nn.Irreps(e3j_module.source)
        out_irreps = e3nn.Irreps(e3j_module.target)
        rotation = e3nn.rand_matrix(self.key)
        rotation_in = in_irreps.D_from_matrix(rotation)
        rotation_out = out_irreps.D_from_matrix(rotation)
        return [rotation_in, rotation_out]

    def rotate(self, g, x):
        return x @ g

    def assert_zero(self, data, tol: float = 5e-6):
        norm = float(np.sqrt(np.sum(data**2))) / data.size
        assert norm < tol

    # --- Test functions ---

    def test_equivariance(self, params, inputs, e3j_module, rotations):
        """Check that module commutes with SO3."""
        input_features, input_indices = inputs
        gfx = self.rotate(
            rotations[1],
            e3j_module.apply(params, input_features, input_indices),
        )
        fgx = e3j_module.apply(
            params,
            self.rotate(rotations[0], input_features),
            input_indices,
        )
        self.assert_zero(gfx - fgx)

    def test_layout_consistency(self, params, inputs, e3j_module):
        """Module commutes with LEADING <-> TRAILING layout cast."""
        if self.num_channels is None:
            pytest.skip("no channel axis to swap")

        other_layout = (
            "TRAILING_CHANNELS"
            if self.layout == "LEADING_CHANNELS"
            else "LEADING_CHANNELS"
        )
        other_module = LinearIndexwise(
            self.in1,
            self.out,
            self.num_indices,
            self.num_channels,
            layout=other_layout,
            kernel_init=e3j_module.kernel_init,
        )

        input_features, input_indices = inputs

        y1 = np.swapaxes(
            e3j_module.apply(params, input_features, input_indices), -1, -2
        )
        y2 = other_module.apply(
            params, np.swapaxes(input_features, -1, -2), input_indices
        )

        assert_allclose(y1, y2)


class TestLinear_1(_TestLinearIndexwise):

    out = "2x1e"
    in1 = "2x1e"


class TestLinear_012(_TestLinearIndexwise):

    out = "4x0e + 4x1o + 4x2e"
    in1 = "2x0e + 3x1o + 4x2e"


class TestLinear_012_123(_TestLinearIndexwise):

    out = "4x0e + 4x1o + 4x2e + 0x3o"
    in1 = "0x0e + 3x1o + 3x2e + 3x3o"


class TestLinear_0246_24(_TestLinearIndexwise):

    out = "0e + 2e + 4e + 6e"
    in1 = "0x0e + 8x2e + 16x4e + 0x6e"


class TestLinear_skip(_TestLinearIndexwise):
    """Source has only scalars, target has higher-order irreps (skip-connection)."""

    out = "4x0e + 4x1o + 4x2e"
    in1 = "8x0e"


class TestLinearIndexwise(_TestLinearIndexwise):

    out = "8x0e + 8x1o + 8x2e"
    in1 = "16x0e + 16x1o + 4x2e"

    num_channels = None

    @pytest.fixture
    def inputs(self):
        """Arrange inputs for Linear modules."""
        rep_in = e3nn.Irreps(self.in1)
        input_features = random.normal(self.key, (self.batch_size, rep_in.dim))
        input_indices = random.choice(self.key, self.num_indices, (self.batch_size,))
        return (input_features, input_indices)


# --- Leading channels tests (N, C, dim) ---


class TestLinearIndexwiseLeading(_TestLinearIndexwise):
    """Leading channels with mismatched source/target irreps."""

    out = "4x0e + 4x1o + 4x2e"
    in1 = "2x0e + 3x1o + 4x2e"
    num_channels = 32


class TestLinearIndexwiseLeading_skip(_TestLinearIndexwise):
    """Leading channels skip-connection: scalar source, higher-order target."""

    out = "4x0e + 4x1o + 4x2e"
    in1 = "8x0e"
    num_channels = 32


# --- Trailing channels tests (N, dim, C) ---


class _TestLinearIndexwiseTrailing(_TestLinearIndexwise):
    """Base class for trailing channels layout (N, dim, C)."""

    num_channels: int = 32
    layout: str = "TRAILING_CHANNELS"

    @pytest.fixture
    def inputs(self):
        rep_in = e3nn.Irreps(self.in1)
        input_features = random.normal(
            self.key, (self.batch_size, rep_in.dim, self.num_channels)
        )
        input_indices = random.choice(self.key, self.num_indices, (self.batch_size,))
        return (input_features, input_indices)

    def rotate(self, g, x):
        return np.einsum("nic,ij->njc", x, g)


class TestLinearIndexwiseTrailing(_TestLinearIndexwiseTrailing):

    out = "4x0e + 4x1o + 4x2e"
    in1 = "2x0e + 3x1o + 4x2e"


class TestLinearIndexwiseTrailing_skip(_TestLinearIndexwiseTrailing):
    """Trailing channels skip-connection: scalar source, higher-order target."""

    out = "4x0e + 4x1o + 4x2e"
    in1 = "8x0e"


# --- FAN_IN_FCTP normalization tests ---


class TestLinearIndexwiseFanInFCTP(_TestLinearIndexwise):
    """FAN_IN_FCTP normalization scales output variance by 1/num_indices."""

    out = "4x0e + 4x1o + 4x2e"
    in1 = "2x0e + 3x1o + 4x2e"
    num_indices = 16

    @pytest.fixture(scope="class")
    def e3j_module(self):
        return LinearIndexwise(
            self.in1,
            self.out,
            self.num_indices,
            self.num_channels,
            layout=self.layout,
            kernel_init="FAN_IN_FCTP",
        )

    def test_fctp_variance_ratio(self, inputs):
        """FAN_IN_FCTP output variance is 1/num_indices of FAN_IN output variance."""
        feats, indices = inputs

        module_none = LinearIndexwise(
            self.in1,
            self.out,
            self.num_indices,
            self.num_channels,
            layout=self.layout,
            kernel_init="FAN_IN",
        )
        module_fctp = LinearIndexwise(
            self.in1,
            self.out,
            self.num_indices,
            self.num_channels,
            layout=self.layout,
            kernel_init="FAN_IN_FCTP",
        )

        params_none = module_none.init(self.key, feats, indices)
        params_fctp = module_fctp.init(self.key, feats, indices)

        y_none = module_none.apply(params_none, feats, indices)
        y_fctp = module_fctp.apply(params_fctp, feats, indices)

        var_ratio = float(np.var(y_fctp)) / float(np.var(y_none))
        expected_ratio = 1.0 / self.num_indices
        assert_allclose(expected_ratio, var_ratio, rtol=0.15, atol=1e-4)
