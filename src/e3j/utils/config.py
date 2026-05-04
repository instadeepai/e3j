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
from contextlib import contextmanager
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

import yaml

from e3j.utils._yaml_config import YamlConfig
from e3j.utils.options import Aggregation, Layout, TensorProduct

E3J_CONFIG = Path(os.environ.get("E3J_CONFIG") or "e3j.yaml")

E3J_OPS_AVAILABLE = True
try:
    import e3j_ops  # noqa
except ModuleNotFoundError:
    E3J_OPS_AVAILABLE = False


@dataclass
class Config(YamlConfig):
    """Global configuration for e3j.

    Controls evaluation strategy, aggregation method, array layout, and
    debug verbosity. Values can be set programmatically or loaded from a
    YAML file (see ``E3J_CONFIG`` environment variable).

    While :class:`Config` objects are plain dataclasses with additional
    I/O and serialization support, the global mutable state is of type
    :class:`config` (lower case).

    Attributes:
        layout: Default array layout for equivariant features.
            See :class:`~e3j.utils.options.Layout`.
        tensor_product: Evaluation strategy for tensor products.
            See :class:`~e3j.utils.options.TensorProduct`.
        aggregation: Aggregation method for sparse reduction steps.
            Only used if `tensor_product` option is "SPARSE".
            See :class:`~e3j.utils.options.Aggregation`.
        debug_level: Verbosity level (0 = silent).

    Example::

        >>> cfg: Config = config.state()
    """

    layout: Layout = Layout.TRAILING_CHANNELS
    tensor_product: TensorProduct = (
        TensorProduct.FUSED if E3J_OPS_AVAILABLE else TensorProduct.SPARSE
    )
    tensor_product_bwd: bool = False
    aggregation: Aggregation = Aggregation.SCATTER
    debug_level: int = 0


class config(Config):
    """Singleton config class for e3j.

    This class manages a global :class:`Config` instance. See
    `help(e3j.config.state())` for more details on the actual configuration options.

    Usage:

        .. code::python

            # context manager:
            with e3j.use(**kwargs):
                ...
            # get mutable global state
            cfg = e3j.config()
            # get copy of current global state
            cfg = e3j.config.state()
            # set permanently
            e3j.config(**kwargs)

    """

    _state: type["config"] = None

    _path = E3J_CONFIG if E3J_CONFIG.exists() else None

    _is_initialized: bool = False

    def __new__(cls, **kwargs):
        """Get/set the global configuration object."""
        # Initialize global configuration
        if not cls._is_initialized:
            _state = super().__new__(cls)
            cls._state = _state
        # Getter
        if not len(kwargs):
            return cls._state
        # Setter
        for key, value in kwargs.items():
            setattr(cls._state, key, value)
        return cls._state

    def __init__(self, **kwargs):
        if self.__class__._is_initialized:
            return None
        super().__init__(**kwargs)
        self.__class__._is_initialized = True

    @classmethod
    def fields(cls):
        return Config.__annotations__

    def __setattr__(self, key, value):
        fields = self.fields()
        if not key in fields:
            raise ValueError(
                f"{key} is not an e3j configuration field. Valid keys are:\n"
                f"{list(fields.keys())}"
            )
        if not isinstance(value, fields[key]):
            value = fields[key](value)
        super().__setattr__(key, value)

    @classmethod
    @contextmanager
    def use(cls, **kwargs):
        """Override configuration within context scope."""
        _state = cls()
        old = {key: getattr(_state, key) for key in cls.fields()}
        try:
            cls(**kwargs)
            new = Config(**kwargs)
            yield new

        finally:
            cls(**old)

    @classmethod
    def state(cls) -> Config:
        """Return a frozen copy of current state."""
        _state = cls()
        fields = {key: getattr(_state, key) for key in cls.fields()}
        return Config(**fields)


# ==== Read configuration from file ====

if E3J_CONFIG.exists():
    config.from_yaml(E3J_CONFIG)
