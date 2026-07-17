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

#include "cuda/convolution.cuh"
#include "cuda/convolution/launch_fwd.cuh"
#include "cuda/dispatch_macros.h"

namespace e3j {
namespace convolution {

#define FOR_EACH_DTYPE_PAIR(Idx, Val)              \
template e3j::Error launch<Idx, Val>(              \
    const Coef4D<Idx,Val> *coef,                   \
    const Val *x,                                  \
    const Val *y,                                  \
    const Val *s,                                  \
    const AdjacencyCSR adj,                        \
    Val *m,                                        \
    Params p,                                      \
    cudaStream_t stream,                           \
    int debug                                      \
);
__FOR_EACH_DTYPE_PAIR
#undef FOR_EACH_DTYPE_PAIR

} // namespace convolution
} // namespace e3j
