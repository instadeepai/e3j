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

from enum import Enum
from typing import Any, Self


class EnumOption(Enum):
    """Helper Enum subclass accepting member names as values.

    Instances can be built from their (case-insensitive) member name, e.g.
    `Layout("LEADING_CHANNELS")`, in addition to the standard value lookup.
    The `parse` and `parse_value` methods add a more informative error message
    than the default `Enum` lookup.
    """

    @classmethod
    def _missing_(cls, key: Any) -> Self | None:
        # fall back to case-insensitive name lookup, so member names are
        # accepted wherever the enum value is expected.
        if isinstance(key, str):
            return cls.__members__.get(key.upper())
        return None

    @classmethod
    def parse(cls, key: str | Self) -> Self:
        try:
            return cls(key)
        except ValueError:
            options = " | ".join(cls.__members__)
            raise KeyError(
                f"expected one of ({options}) for {cls.__name__}, got {key!r}"
            )

    @classmethod
    def parse_value(cls, key: str | Self) -> int | Any:
        return cls.parse(key).value
