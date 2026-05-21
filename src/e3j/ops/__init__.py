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

try:
    import e3j_ops

    from .jax_ffi import ffi
    from .scatter_add import scatter_add_1
    from .tensor_product import TensorProductParams, tensor_product

    for _name, _value in e3j_ops.get_registrations().items():
        ffi.register_ffi_target(_name, _value, platform="CUDA")

except ModuleNotFoundError:

    def scatter_add_1(*xs, **ks):
        raise NotImplementedError(
            "Module `e3j_ops` not found.\n"
            "Run `make e3j_ops` and add `e3j/bin` to your $PYTHONPATH."
        )

    def tensor_product(*xs, **ks):
        raise NotImplementedError(
            "Module `e3j_ops` not found.\n"
            "Run `make e3j_ops` and add `e3j/bin` to your $PYTHONPATH."
        )

    TensorProductParams = object
