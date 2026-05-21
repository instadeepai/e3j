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

#ifndef E3J_CUDA_SCATTER_ADD_H_
#define E3J_CUDA_SCATTER_ADD_H_

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda/dtype.cuh"
#include "ffi/error.h"

namespace e3j {

namespace scatter_add_1 {

using int32 = std::int32_t;

struct Params {
    int32 num_rows;
    int32 num_idx;
    int32 num_out;
};

template<typename Idx, typename Val>
__global__ void kernel(
    const Idx *idx,
    const Val *val,
    Val *out,
    Params p
);

template<typename Idx, typename Val>
e3j::Error launch(
    const Idx *idx,
    const Val *val,
    Val *out,
    Params p,
    cudaStream_t stream
);

} // namespace scatter_add_1
} // namespace e3j

#endif // E3J_CUDA_SCATTER_ADD_H_
