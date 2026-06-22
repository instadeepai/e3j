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

#ifndef E3J_CUDA_CONVOLUTION_BWD_H_
#define E3J_CUDA_CONVOLUTION_BWD_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda/convolution.cuh"
#include "ffi/error.h"

namespace e3j {
namespace convolution {

template <typename Idx, typename Val>
e3j::Error launch_bwd(
    const Coef4D<Idx, Val> *coef,
    const Val *x,
    const Val *y,
    const Val *dz,
    const Val *mix,
    const AdjacencyCSR adj,
    const int32_t *edge_perm,
    Val *dx,
    Val *dy,
    Val *dmix,
    Params p,
    cudaStream_t stream,
    int debug
);

} // namespace convolution
} // namespace e3j

#endif // E3J_CUDA_CONVOLUTION_BWD_H_
