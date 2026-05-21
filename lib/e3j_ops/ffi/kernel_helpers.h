/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// NOTE: we merged the two boilerplate header files from:
//
//     https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html
//
// as it doesn't make any sense to '(Un)PackDescriptor' arguments from C++.
// This bit-cast machinery is only meant for XLA calls within Python, so
// the pybind11 code can rest in our `lib/ffi` dir safely.

#ifndef _E3J_OPS_KERNEL_HELPERS_H_
#define _E3J_OPS_KERNEL_HELPERS_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <pybind11/pybind11.h>

#include "xla/ffi/api/c_api.h"

namespace py = pybind11;

namespace e3j_ops {

template <typename T> pybind11::capsule pyEncapsulateFunction(T *fn) {
    static_assert(
        std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
        "Encapsulated function must be an XLA FFI handler"
    );
    return pybind11::capsule(reinterpret_cast<void *>(fn));
}


} // namespace e3j_ops

#endif
