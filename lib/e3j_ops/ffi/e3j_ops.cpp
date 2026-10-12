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

#include <stdexcept>
#include <string>

#include "kernel_helpers.h" // already includes pybind
#include "cuda/device_properties.h"
#include "ffi/e3j_ops.h"

namespace py = pybind11;

namespace {

// Query device properties (SMEM budgets, occupancy limits) so Python launch
// heuristics (see `e3j.ops.coef_ranges.choose_block_y`) can size `blockDim.y`
// hardware-aware instead of against hardcoded H100 constants.
py::dict GetDeviceProperties () {
    e3j::utils::DeviceProperties props = e3j::utils::DeviceProperties::query();
    py::dict out;
    out["device"] = props.device;
    out["smem_max"] = props.smem_max;
    out["smem_opt_in"] = props.smem_opt_in;
    out["smem_per_sm"] = props.smem_per_sm;
    out["max_threads_per_sm"] = props.max_threads_per_sm;
    out["max_blocks_per_sm"] = props.max_blocks_per_sm;
    out["regs_per_sm"] = props.regs_per_sm;
    return out;
}

// Registers/thread for one of the fused convolution kernels, via
// `cudaFuncGetAttributes` (see `e3j::convolution::get_registers{,_bwd}`), so
// Python's `choose_block_y` can cap `resident_blocks` by the register budget
// too. `idx_dtype`/`val_dtype` mirror the pairs in `dispatch_macros.h`.
template<typename Idx, typename Val>
int DispatchKernelRegisters(const std::string &kernel, int N) {
    if (kernel == "convolution")
        return e3j::convolution::get_registers<Idx, Val>(N);
    if (kernel == "convolution_bwd")
        return e3j::convolution::get_registers_bwd<Idx, Val>(N);
    throw std::invalid_argument(
        "get_kernel_registers: unsupported kernel '" + kernel + "'");
}

int GetKernelRegisters (
    const std::string &kernel,
    const std::string &idx_dtype,
    const std::string &val_dtype,
    int N
) {
    if (idx_dtype == "uint8" && val_dtype == "float32")
        return DispatchKernelRegisters<std::uint8_t, float>(kernel, N);
    if (idx_dtype == "int32" && val_dtype == "float32")
        return DispatchKernelRegisters<std::int32_t, float>(kernel, N);
    if (idx_dtype == "uint8" && val_dtype == "float16")
        return DispatchKernelRegisters<std::uint8_t, __half>(kernel, N);
    if (idx_dtype == "int32" && val_dtype == "float16")
        return DispatchKernelRegisters<std::int32_t, __half>(kernel, N);
    if (idx_dtype == "uint8" && val_dtype == "float64")
        return DispatchKernelRegisters<std::uint8_t, double>(kernel, N);
    if (idx_dtype == "int32" && val_dtype == "float64")
        return DispatchKernelRegisters<std::int32_t, double>(kernel, N);
    throw std::invalid_argument(
        "get_kernel_registers: unsupported (idx_dtype, val_dtype) pair");
}

py::dict GetRegistrations () {
    // NOTE: Encapsulation of XLA ops MUST be postponed to the evaluation
    // of such a function, otherwise pybind and the compiler complain about
    // the incomplete definition of `cudaStream_t`.
    py::dict ops;
    ops["scatter_add_1"] = e3j_ops::pyEncapsulateFunction(e3j_ops::xla_scatter_add_1);
    ops["tensor_product"] = e3j_ops::pyEncapsulateFunction(e3j_ops::xla_tensor_product);
    ops["tensor_product_bwd"] = e3j_ops::pyEncapsulateFunction(e3j_ops::xla_tensor_product_bwd);
    ops["convolution"] = e3j_ops::pyEncapsulateFunction(e3j_ops::xla_convolution);
    ops["convolution_bwd"] = e3j_ops::pyEncapsulateFunction(e3j_ops::xla_convolution_bwd);
    return ops;
}

PYBIND11_MODULE(e3j_ops, m){

    m.doc() = "Custom XLA ops";

    m.def("get_registrations", &GetRegistrations);
    m.def("get_device_properties", &GetDeviceProperties);
    m.def("get_kernel_registers", &GetKernelRegisters);

}

} // namespace
