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

from e3j.spaces.o3 import O3Space
from e3j.spaces.so3 import SO3Space
from e3j.spaces.space import Space


def to_space(space: Space | str) -> Space:
    """Helper dispatching to O3Space or SO3Space."""
    if isinstance(space, Space):
        return space
    if isinstance(space, str):
        if "e" in space or "o" in space:
            return O3Space(space)
        else:
            return SO3Space(space)
