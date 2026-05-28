/* Copyright (c) 2026 InstaDeep Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_helpers.h" // already includes pybind
#include "ffi/e3j_ops.h"

namespace py = pybind11;

namespace {

py::dict GetRegistrations () {
    // NOTE: Encapsulation of XLA ops MUST be postponed to the evaluation
    // of such a function, otherwise pybind and the compiler complain about
    // the incomplete definition of `cudaStream_t`.
    py::dict ops;
    ops["scatter_add_1"] = e3j_ops::pyEncapsulateFunction(e3j_ops::xla_scatter_add_1);
    ops["tensor_product"] = e3j_ops::pyEncapsulateFunction(e3j_ops::xla_tensor_product);
    ops["tensor_product_bwd"] = e3j_ops::pyEncapsulateFunction(e3j_ops::xla_tensor_product_bwd);
    return ops;
}

PYBIND11_MODULE(e3j_ops, m){

    m.doc() = "Custom XLA ops";

    m.def("get_registrations", &GetRegistrations);

}

} // namespace
