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

#include "cuda/fill.cuh"
#include <cuda_fp16.h>
#include <iostream>

namespace e3j {
namespace fill {

// TODO: Compute occupancy depending on device
#define WARPS_PER_BLOCK 2
#define THREADS_PER_BLOCK 64

template<typename Val>
__global__ void kernel(Val* out, Val value, Params p) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < p.num_val) {
        out[idx] = value;
    }
}

template<typename Val>
e3j::Error launch(Val *out, Val value, Params p, cudaStream_t stream) {

    std::size_t sharedMem = 0;
    int numBlocks = 1 + (p.num_val - 1) / THREADS_PER_BLOCK;

    kernel<<<numBlocks, THREADS_PER_BLOCK, sharedMem, stream>>>(
        out, value, p
    );

    return e3j::Error::FromCudaLaunch(cudaGetLastError());
}

#define DISPATCH_VAL_DTYPE  \
    X(float)                \
    X(double)               \
    X(__half)

#define X(Val)                                       \
template e3j::Error launch<Val> (                    \
    Val *out, Val value, Params p, cudaStream_t stream  \
);
DISPATCH_VAL_DTYPE
#undef X

} // namespace fill
} // namespace e3j
