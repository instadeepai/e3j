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
from enum import Enum
from typing import Any

import yaml


class YamlConfig:
    """A helper base class for yaml I/O.

    Keeps `Config` field options separate in `config.py`.
    Also helps (de)serializing Enum fields.
    """

    @classmethod
    def fields(cls):
        return cls.__annotations__

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for key, key_t in self.fields().items():
            val = getattr(self, key)
            if issubclass(key_t, Enum):
                val = val.value
            out[key] = val
        return out

    def to_yaml(self, path: os.PathLike | None = None) -> None | str:
        if path is not None:
            with open(path, "w") as stream:
                yaml.dump(self.to_dict(), stream)
            return None
        else:
            return yaml.dump(self.to_dict())

    @classmethod
    def from_yaml(cls, path: os.PathLike | str) -> type["YamlConfig"]:
        with open(path, "r") as stream:
            dct = yaml.load(stream, yaml.FullLoader)
        return cls(**dct)

    @classmethod
    def options(cls):
        out = {}
        for key, key_t in cls.fields().items():
            if issubclass(key_t, Enum):
                out[key] = " | ".join(key_t.__members__.keys())
            else:
                out[key] = key_t
        return out
