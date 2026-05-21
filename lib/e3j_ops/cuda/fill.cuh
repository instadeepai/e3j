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

#ifndef E3J_CUDA_FILL_H_
#define E3J_CUDA_FILL_H_

#include <cstdint>
#include "cuda.h"
#include "cuda_runtime_api.h"

#include "cuda/dtype.cuh"
#include "ffi/error.h"

namespace e3j {
namespace fill {

struct Params {
    std::int32_t num_val;
    Dtype val_t;
};

template <typename Val>
__global__ void kernel(Val *out, Val value, Params p);

template <typename Val>
e3j::Error launch(Val *out, Val value, Params p, cudaStream_t stream);

} // namespace fill
} // namespace e3j

#endif
