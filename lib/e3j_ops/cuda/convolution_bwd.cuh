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

/*************************************************
 *  Convolution backward kernel launcher
 *
 *  The backward coefficients should be packed as a
 *  triple (coef_dx, coef_dy, coef_ds) such that:
 *
 *   - dx = bigotimes(coef_dx, dm, y, s)
 *   - dy = bigotimes(coef_dy, dm, x, s)
 *   - ds = bigotimes(coef_ds, dm, y, x)
 *
 *     @param coef packed coefficients for the backward pass
 *     @param x primal node features
 *     @param y primal edge features
 *     @param s primal edge scalars
 *     @param dm cotangent of messages
 *     @param adj CSR adjacency matrix of transposed graph
 *     @param edge_perm transposition of edges
 *     @param dx output buffer for node feature cotangents
 *     @param dy output buffer for edge feature cotangents
 *     @param ds output buffer for edge scalar cotangents
 *     @param p convolution parameters / problem sizes
 *     @param stream CUDA stream
 *     @param debug debug level

 *************************************************/
template <typename Idx, typename Val>
e3j::Error launch_bwd(
    const Coef4D<Idx, Val> *coef,
    const Val *x,
    const Val *y,
    const Val *s,
    const Val *dz,
    const AdjacencyCSR adj,
    const int32_t *edge_perm,
    Val *dx,
    Val *dy,
    Val *ds,
    Params p,
    cudaStream_t stream,
    int debug
);

} // namespace convolution
} // namespace e3j

#endif // E3J_CUDA_CONVOLUTION_BWD_H_
