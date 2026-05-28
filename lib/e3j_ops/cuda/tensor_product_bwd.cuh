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

#ifndef _E3J_TENSOR_PRODUCT_BWD_H_
#define _E3J_TENSOR_PRODUCT_BWD_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda/tensor_product.cuh"
#include "ffi/error.h"
#include <iostream>

namespace e3j {
namespace tensor_product {

namespace trailing_channels {

template<typename Idx, typename Val, Mode kxMode, Mode kyMode>
e3j::Error launch_bwd(
    const Coef<Idx, Val> *coef,
    const Val *x,
    const Val *y,
    const Val *dz,
    Val *dx,
    Val *dy,
    const Params p,
    cudaStream_t stream,
    int debug
);

template<typename Idx, typename Val>
e3j::Error launch_bwd(
    const Coef<Idx, Val> *coef,
    const Val *x,
    const Val *y,
    const Val *dz,
    Val *dx,
    Val *dy,
    const Params p,
    cudaStream_t stream,
    int debug
);

}// namespace trailing_channels

struct ModeBwd {
    Mode lhs;
    Mode rhs;
};

struct ParamsBwd {
    Params lhs;
    Params rhs;
};

// Infer modes of codifferential on dx and dy
ModeBwd GetModeBwd (Mode fwd, int channels_lhs, int channels_rhs);

// Infer params of codifferential on dx and dy
ParamsBwd GetParamsBwd(Params p);


} // namespace tensor_product
} // namespace e3j

#endif // _E3J_TENSOR_PRODUCT_BWD_H_
