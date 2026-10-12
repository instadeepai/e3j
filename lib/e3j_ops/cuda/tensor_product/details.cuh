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

#ifndef _E3J_TENSOR_PRODUCT_DETAILS_H_
#define _E3J_TENSOR_PRODUCT_DETAILS_H_

#include "cuda/tensor_product.cuh"
#include "cuda/utils.cuh"
#include <iostream>

namespace e3j {

namespace tensor_product {

    // Container for I/O buffers.
    //
    // Helps summarize function signatures in device code, while
    // encapsulating some of the opaque pointer offset logic.
    //
    // Some of the (optional) copy boilerplate could also delegated
    // this way (e.g. to handle striding or leverage L2 in larger calls),
    // if e.g. a `Buffer` class was used as template parameter.
    template<typename Val>
    struct SMEM {
        Val *x;
        Val *y;
        Val *z;

        template<int VEC=1, typename T>
        static __device__ SMEM from_extern(T* smem_, Params p) {
            Val* smem_x   = reinterpret_cast<Val*>(smem_);
            // Round up to the next VEC multiple so smem_y is VEC-float aligned
            // for vectorized loads (LDS.64 / LDS.128): ceil(n / VEC) * VEC
            Val* smem_y   = smem_x + ((p.unroll_x * p.num_x + VEC-1) / VEC) * VEC;
            Val* smem_z   = smem_y + p.unroll_y * p.num_y;
            return SMEM {
                .x=smem_x,
                .y=smem_y,
                .z=smem_z
            };
        }
    };

    using std::int32_t;

    struct CoefRange {
        int begin;
        int end;
    };

namespace leading_channels {

    template<typename Idx, typename Val, Mode kMode>
    e3j::Error launch (
        const Coef<Idx,Val> *coef,
        const Val *x,
        const Val *y,
        Val *out,
        Params p,
        cudaStream_t stream,
        int debug
    );

} // namespace leading_channels


namespace trailing_channels {

    template<typename Idx, typename Val, Mode kMode>
    e3j::Error launch (
        const Coef<Idx,Val> *coef,
        const Val *x,
        const Val *y,
        Val *out,
        Params p,
        cudaStream_t stream,
        int debug
    );

} // namespace trailing_channels

} // namespace tensor_product
} // namespace e3j

#endif
