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

import pickle

import e3nn_jax as e3nn
import jax
import pytest
from conftest import assert_allclose
from jax import numpy as np
from jax import random

import e3j
from e3j.core.symmetric_contraction import (
    SymmetricContraction as CoreSymmetricContraction,
)
from e3j.linen.symmetric_contraction import SymmetricContraction
from e3j.spaces import O3Space
from e3j.utils.config import Config
from e3j.utils.options import Layout, TensorProduct


class _TestSymmetricContraction:
    """Base class: features are built `(N, C, dim)` and swapped to `layout`."""

    in1: str
    out: str
    correlation: int = 3
    num_species: int = 4
    num_channels: int = 8
    batch_size: int = 16
    layout: str = "TRAILING_CHANNELS"
    _seed: int = 42

    @property
    def key(self) -> jax.Array:
        return random.key(self._seed)

    @pytest.fixture(scope="class")
    def e3j_module(self):
        return SymmetricContraction(
            source_irreps=self.in1,
            correlation=self.correlation,
            keep_irrep_out=self.out,
            num_species=self.num_species,
            num_channels=self.num_channels,
            layout=self.layout,
        )

    @pytest.fixture
    def inputs(self):
        """Arrange ``(node_feats, index)`` inputs for the module."""
        source = O3Space(self.in1)
        feats = random.normal(
            self.key, (self.batch_size, self.num_channels, source.dim)
        )
        index = random.choice(self.key, self.num_species, (self.batch_size,))
        return (feats, index)

    @pytest.fixture
    def params(self, e3j_module, inputs):
        """Arrange parameters for the module."""
        feats, index = inputs
        return e3j_module.init(self.key, self._to_layout(feats, self.layout), index)

    @pytest.fixture(scope="class")
    def rotations(self, e3j_module) -> list[np.ndarray]:
        """Input and output rotations in the e3j basis."""
        source = O3Space(self.in1)
        target = e3j_module.target
        with jax.default_matmul_precision("highest"):
            rotation = e3nn.rand_matrix(self.key)
            return [source.action(rotation), target.action(rotation)]

    @staticmethod
    def _to_layout(feats: np.ndarray, layout: str) -> np.ndarray:
        """Swap `(N, C, dim)` to and from `layout` (an involution)."""
        if Layout.parse(layout) == Layout.TRAILING_CHANNELS:
            return np.swapaxes(feats, -1, -2)
        return feats

    def assert_zero(self, data, tol: float = 5e-6):
        norm = float(np.sqrt(np.sum(data**2))) / data.size
        assert norm < tol

    # --- Test functions ---

    def test_output_shape(self, e3j_module, params, inputs):
        """Output is sized for the regrouped target irreps."""
        feats, index = inputs
        out = e3j_module.apply(params, self._to_layout(feats, self.layout), index)
        y = self._to_layout(out, self.layout)
        assert y.shape == (self.batch_size, self.num_channels, e3j_module.target.dim)

    def test_equivariance(self, e3j_module, params, inputs, rotations):
        """Check that the module commutes with O(3) on the feature axis."""
        feats, index = inputs
        rot_in, rot_out = rotations
        with jax.default_matmul_precision("highest"):
            y = self._to_layout(
                e3j_module.apply(params, self._to_layout(feats, self.layout), index),
                self.layout,
            )
            gfx = np.einsum("nci,ij->ncj", y, rot_out)
            feats_rot = np.einsum("nci,ij->ncj", feats, rot_in)
            fgx = self._to_layout(
                e3j_module.apply(
                    params, self._to_layout(feats_rot, self.layout), index
                ),
                self.layout,
            )
        self.assert_zero(gfx - fgx)

    def test_param_structure(self, e3j_module, params):
        """Weights live under `LinearIndexwise_0/`, one per block, in order."""
        core = CoreSymmetricContraction(
            self.in1,
            self.correlation,
            self.out,
            self.num_species,
            self.num_channels,
            self.layout,
        )
        weights = params["params"]["LinearIndexwise_0"]
        assert sorted(weights) == sorted(
            f"weight{i}_{b.l}_{b.p}" for i, b in enumerate(core.linear_blocks)
        )
        for i, block in enumerate(core.linear_blocks):
            m_out, m_in = block.shape
            assert weights[f"weight{i}_{block.l}_{block.p}"].shape == (
                self.num_species,
                self.num_channels,
                m_out,
                m_in,
            )

    def test_core_reuse_after_jit(self, inputs):
        """A core first built under `jit` must not leak tracers.

        Weight shapes come from another instance, so nothing is built before.
        """
        args = (
            self.in1,
            self.correlation,
            self.out,
            self.num_species,
            self.num_channels,
            self.layout,
        )
        weights = [
            random.normal(self.key, (self.num_species, self.num_channels, *block.shape))
            for block in CoreSymmetricContraction(*args).linear_blocks
        ]
        core = CoreSymmetricContraction(*args)
        feats, index = inputs
        x = self._to_layout(feats, self.layout)
        apply = jax.jit(core)
        apply(x[1:], index[1:], weights)
        apply(x, index, weights)
        core(x, index, weights)


class TestSymmetricContractionGolden:
    """Pin the output, which equivariance alone does not: init, scaling, order."""

    in1 = "0e + 1o + 2e + 3o"
    out = "0e + 1o"
    correlation = 3
    num_species = 4
    num_channels = 8
    batch_size = 16

    def test_golden_values(self):
        key = random.key(42)
        module = SymmetricContraction(
            source_irreps=self.in1,
            correlation=self.correlation,
            keep_irrep_out=self.out,
            num_species=self.num_species,
            num_channels=self.num_channels,
            layout="LEADING_CHANNELS",
        )
        feats = random.normal(
            key, (self.batch_size, self.num_channels, O3Space(self.in1).dim)
        )
        index = random.choice(key, self.num_species, (self.batch_size,))
        with jax.default_matmul_precision("highest"):
            y = module.apply(module.init(key, feats, index), feats, index)

        assert y.shape == (self.batch_size, self.num_channels, 4)
        assert_allclose(float(np.sum(y)), 10.716981, atol=1e-3)
        assert_allclose(float(np.sum(y**2)), 337.169312, atol=1e-2)
        assert_allclose(
            y[0, 0], np.array([0.691399, -0.223694, -0.04847, -0.097783]), atol=1e-4
        )


class TestSymmetricContraction_0123_01(_TestSymmetricContraction):
    in1 = "0e + 1o + 2e + 3o"
    out = "0e + 1o"


class TestSymmetricContraction_corr2(_TestSymmetricContraction):
    in1 = "0e + 1o + 2e"
    out = "0e + 1o + 2e"
    correlation = 2


class TestSymmetricContraction_1o_0e(_TestSymmetricContraction):
    """The source holds no output irrep: only even powers contribute."""

    in1 = "1o"
    out = "0e"

    def test_output_nonzero(self, e3j_module, params, inputs):
        """The squared norm reaches the output."""
        feats, index = inputs
        y = e3j_module.apply(params, self._to_layout(feats, self.layout), index)
        assert np.all(np.abs(y) > 0)


@pytest.mark.parametrize("source, correlation, out", [("1o", 1, "0e"), ("0e", 2, "1o")])
def test_unreachable_output(source, correlation, out):
    """No power holds an output irrep: a clear error, not a parse failure."""
    core = CoreSymmetricContraction(source, correlation, out, 2, 4, "LEADING_CHANNELS")
    with pytest.raises(ValueError, match="No power of"):
        core.linear_blocks


class TestSymmetricContractionConfig:
    """The contraction's own config drives its power-expansion products."""

    in1 = "0e + 1o"
    out = "0e + 1o"
    correlation = 3
    num_species = 4
    num_channels = 8
    batch_size = 16

    def core(self, **kwargs) -> CoreSymmetricContraction:
        return CoreSymmetricContraction(
            self.in1,
            self.correlation,
            self.out,
            self.num_species,
            self.num_channels,
            "LEADING_CHANNELS",
            **kwargs,
        )

    def module(self, **kwargs) -> SymmetricContraction:
        return SymmetricContraction(
            self.in1,
            self.correlation,
            self.out,
            self.num_species,
            self.num_channels,
            "LEADING_CHANNELS",
            **kwargs,
        )

    def test_tensor_products_use_module_config(self):
        """A per-module config is not overridden by the global one."""
        core = self.core(config=Config(tensor_product=TensorProduct.DENSE))
        impls = [
            otimes.config.tensor_product for otimes in core.expansion.otimes_layers
        ]
        assert impls == [TensorProduct.DENSE, TensorProduct.DENSE]

    def test_global_config_change_after_first_call(self):
        """A global config change does not reach already-built products."""
        core = self.core()
        key = random.key(0)
        feats = random.normal(
            key, (self.batch_size, self.num_channels, O3Space(self.in1).dim)
        )
        index = random.choice(key, self.num_species, (self.batch_size,))
        weights = [
            random.normal(key, (self.num_species, self.num_channels, *block.shape))
            for block in core.linear_blocks
        ]
        y = core(feats, index, weights)
        with e3j.config.use(tensor_product="DENSE"):
            y_dense = core(feats, index, weights)
        assert_allclose(y, y_dense)

    def test_core_config_frozen_at_construction(self):
        """A core built under `config.use` keeps that config once it exits."""
        with e3j.config.use(tensor_product="DENSE"):
            core = self.core()
        assert core.config.tensor_product == TensorProduct.DENSE
        impls = [
            otimes.config.tensor_product for otimes in core.expansion.otimes_layers
        ]
        assert impls == [TensorProduct.DENSE, TensorProduct.DENSE]

    def test_core_config_snapshots_live_global(self):
        """Passing the live global config explicitly also freezes it."""
        core = self.core(config=e3j.config())
        assert core.config is not e3j.config()
        with e3j.config.use(tensor_product="DENSE"):
            impls = [
                otimes.config.tensor_product for otimes in core.expansion.otimes_layers
            ]
        assert TensorProduct.DENSE not in impls

    def test_module_config_frozen_at_construction(self):
        """A module built under `config.use` keeps that config once it exits."""
        with e3j.config.use(tensor_product="DENSE"):
            module = self.module()
        assert module.config.tensor_product == TensorProduct.DENSE
        assert module.config is not e3j.config()

    def test_module_snapshots_live_global(self):
        """An explicit live global config is frozen: hashable, safe to unpickle."""
        module = self.module(config=e3j.config())
        assert module.config is not e3j.config()
        assert module.config == e3j.config.state()
        assert hash(module) == hash(self.module())
        data = pickle.dumps(module)
        with e3j.config.use(tensor_product="DENSE"):
            module = pickle.loads(data)
            assert e3j.config().tensor_product == TensorProduct.DENSE
        assert module.config.tensor_product != TensorProduct.DENSE

    def test_module_is_hashable(self):
        """The module hashes, so it can be a static argument of `jax.jit`."""
        module = self.module()
        assert hash(module) == hash(self.module())
        key = random.key(0)
        feats = random.normal(
            key, (self.batch_size, self.num_channels, O3Space(self.in1).dim)
        )
        index = random.choice(key, self.num_species, (self.batch_size,))
        params = module.init(key, feats, index)
        apply = jax.jit(lambda m, *args: m.apply(*args), static_argnums=0)
        assert_allclose(
            apply(module, params, feats, index), module.apply(params, feats, index)
        )


class TestSymmetricContractionWeights:
    """The core contraction rejects weights that do not match its blocks."""

    num_species = 3
    num_channels = 4

    @pytest.fixture(params=["LEADING_CHANNELS", "TRAILING_CHANNELS"])
    def core(self, request) -> CoreSymmetricContraction:
        return CoreSymmetricContraction(
            "0e + 1o + 2e",
            2,
            "0e + 1o + 2e",
            self.num_species,
            self.num_channels,
            request.param,
        )

    @pytest.fixture
    def args(self, core):
        key = random.key(0)
        feats = random.normal(key, (5, self.num_channels, core.source.dim))
        feats = _TestSymmetricContraction._to_layout(feats, core.layout)
        index = random.choice(key, self.num_species, (5,))
        shape = (self.num_species, self.num_channels)
        weights = [random.normal(key, (*shape, *b.shape)) for b in core.linear_blocks]
        return feats, index, weights

    @pytest.mark.parametrize("jit", [False, True])
    def test_short_weights(self, core, args, jit):
        """A missing weight is an error, not a silently dropped block."""
        feats, index, weights = args
        apply = jax.jit(core) if jit else core
        with pytest.raises(ValueError, match="Expected 3 weights"):
            apply(feats, index, weights[:-1])

    def test_extra_weights(self, core, args):
        feats, index, weights = args
        with pytest.raises(ValueError, match="Expected 3 weights"):
            core(feats, index, weights + weights[:1])

    def test_wrong_species_count(self, core, args):
        feats, index, weights = args
        weights[0] = weights[0][:1]
        with pytest.raises(ValueError, match="Weight 0 has shape"):
            core(feats, index, weights)

    def test_misshapen_weight(self, core, args):
        feats, index, weights = args
        weights[0] = weights[0][:, :2]
        with pytest.raises(ValueError, match="Weight 0 has shape"):
            core(feats, index, weights)


class TestSymmetricContractionBatchAxes:
    """Features may carry any number of leading batch axes, as documented."""

    in1 = "0e + 1o + 2e"
    out = "0e + 1o + 2e"
    correlation = 2
    num_species = 3
    num_channels = 4

    @pytest.fixture(params=["LEADING_CHANNELS", "TRAILING_CHANNELS"])
    def layout(self, request) -> str:
        return request.param

    @pytest.fixture(params=["core", "linen"])
    def apply(self, request, layout):
        """`(features, indices) -> output` for the core or linen contraction."""
        args = (
            self.in1,
            self.correlation,
            self.out,
            self.num_species,
            self.num_channels,
            layout,
        )
        key = random.key(0)
        if request.param == "core":
            core = CoreSymmetricContraction(*args)
            shape = (self.num_species, self.num_channels)
            weights = [
                random.normal(key, (*shape, *b.shape)) for b in core.linear_blocks
            ]
            return lambda x, i: core(x, i, weights)
        module = SymmetricContraction(*args)
        x = random.normal(key, (5, *self.feat_shape(layout)))
        params = module.init(key, x, np.zeros(5, dtype=int))
        return lambda x, i: module.apply(params, x, i)

    def feat_shape(self, layout: str) -> tuple[int, int]:
        dim = O3Space(self.in1).dim
        if Layout.parse(layout) == Layout.TRAILING_CHANNELS:
            return (dim, self.num_channels)
        return (self.num_channels, dim)

    @pytest.mark.parametrize("batch", [(), (5,), (1, 1), (2, 5), (2, 1, 3)])
    def test_batch_axes(self, apply, layout, batch):
        """The output matches the flattened-batch one, batch axes restored."""
        key = random.key(1)
        feats = random.normal(key, (*batch, *self.feat_shape(layout)))
        index = random.choice(key, self.num_species, batch)
        y = apply(feats, index)
        y_flat = apply(feats.reshape(-1, *feats.shape[-2:]), index.reshape(-1))
        assert y.shape == (*batch, *y_flat.shape[1:])
        assert_allclose(y, y_flat.reshape(y.shape))

    def test_index_shape_mismatch(self, apply, layout):
        feats = np.ones((2, 5, *self.feat_shape(layout)))
        with pytest.raises(ValueError, match="do not match the batch axes"):
            apply(feats, np.zeros(10, dtype=int))
