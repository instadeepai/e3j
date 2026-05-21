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

"""Workaround to support JAX >= 0.4.31 bindings.

The `jax.extend.ffi` is now deprecated and has been moved to `jax.ffi`, since version 0.X.Y.
"""

try:
    from jax import ffi
except ImportError:
    from jax.extend import ffi
