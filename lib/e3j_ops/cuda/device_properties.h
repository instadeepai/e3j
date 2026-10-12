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

#ifndef _E3J_CUDA_DEVICE_PROPERTIES_H_
#define _E3J_CUDA_DEVICE_PROPERTIES_H_

// Runtime API only: also compiled by g++ for the pybind module.
#include <cuda_runtime_api.h>

namespace e3j {
namespace utils {

// Helper to query useful device properties
struct DeviceProperties {

    int device = 0;
    int smem_max = 0;            // max shared per block (opt-in)
    int smem_opt_in = 0;         // max shared per block without opt-in
    int smem_per_sm = 0;         // max shared per SM
    int max_threads_per_sm = 0;
    int max_blocks_per_sm = 0;
    int regs_per_sm = 0;

    static DeviceProperties query() {
        DeviceProperties out;
        cudaGetDevice(&out.device);
        cudaDeviceGetAttribute(
            &out.smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, out.device
        );
        cudaDeviceGetAttribute(
            &out.smem_opt_in, cudaDevAttrMaxSharedMemoryPerBlock, out.device
        );
        cudaDeviceGetAttribute(
            &out.smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, out.device
        );
        cudaDeviceGetAttribute(
            &out.max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, out.device
        );
        cudaDeviceGetAttribute(
            &out.max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, out.device
        );
        cudaDeviceGetAttribute(
            &out.regs_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, out.device
        );
        return out;
    }
};

} // namespace utils
} // namespace e3j

#endif // _E3J_CUDA_DEVICE_PROPERTIES_H_
