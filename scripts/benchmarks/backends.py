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

"""Provide (mock) references to E3 backends for benchmarks."""

# torch
try:
    import torch

    torch.Tensor

except (ModuleNotFoundError, AttributeError):

    class torch:
        Tensor = type

        class cuda:
            def is_available():
                return False

            def synchronize(): ...
            def empty_cache(): ...


# e3nn-torch
try:
    import e3nn as e3nn_torch
except ModuleNotFoundError:
    e3nn_torch = object()

# cuequivariance-torch
try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
except ModuleNotFoundError:
    cue = object()
    cuet = object()

# openequivariance
try:
    import openequivariance as openeq
except ModuleNotFoundError:
    openeq = object()
