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

import dataclasses

import pytest
from jax import random

from e3j.core.tensor_product import TensorProduct
from e3j.utils.config import Config, config

# CPU-runnable options, given as strings.
STRINGS = dict(
    layout="LEADING_CHANNELS",
    tensor_product="UNFUSED",
    aggregation="SCATTER",
    convolution="UNFUSED",
)


def yaml_round_trip(tmp):
    Config(**STRINGS).to_yaml(tmp / "e3j.yaml")
    return Config.from_yaml(tmp / "e3j.yaml")


@pytest.mark.parametrize(
    "make",
    [
        lambda tmp: Config(**STRINGS),
        lambda tmp: dataclasses.replace(Config(), **STRINGS),
        yaml_round_trip,
    ],
    ids=["init", "replace", "yaml"],
)
def test_strings_are_cast(make, tmp_path):
    cfg = make(tmp_path)
    for key, value in STRINGS.items():
        assert getattr(cfg, key) == Config.fields()[key][value]
    assert hash(cfg) == hash(Config(**STRINGS))
    feats = random.normal(random.key(0), (5, 2, 4))
    tp = TensorProduct(("0e + 1o", "0e + 1o"), None, config=cfg)
    assert tp(feats[:, 0], feats[:, 1]).shape == (5, tp.target.dim)


def test_invalid_string():
    with pytest.raises(ValueError, match="NOT_AN_OPTION"):
        Config(tensor_product="NOT_AN_OPTION")


def test_global_setter():
    with config.use(layout="LEADING_CHANNELS"):
        assert config().layout.name == "LEADING_CHANNELS"
        frozen = config.snapshot(config())
    assert frozen.layout.name == "LEADING_CHANNELS"
    assert config.snapshot(frozen) is frozen
    with pytest.raises(ValueError, match="not an e3j configuration field"):
        config(not_a_field=1)


def test_use_yields_full_state():
    with config.use(debug_level=3):
        with config.use(layout="LEADING_CHANNELS") as cfg:
            assert cfg == config.state()
            assert cfg.debug_level == 3
