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

#ifndef E3J_CUDA_TENSOR_PRODUCT_H_
#define E3J_CUDA_TENSOR_PRODUCT_H_

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "ffi/error.h"

constexpr size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

namespace e3j {

namespace tensor_product {

using int32 = std::int32_t;
using uint8_t = std::uint8_t;

/********************************************************************
 * Mode : OUTER | INNER
 *
 * - `OUTER` multiplies channels in outer-product fashion
 * - `INNER` implies that both LHS and RHS have same channel
 *   dimensions for now and aggregates them in SMEM.
 *******************************************************************/
enum Mode { OUTER, INNER, MAP };

/******************************************************************
 *  Channel layout: leading (k, lm) or trailing (lm, k).
 *****************************************************************/
enum Layout { LEADING_CHANNELS, TRAILING_CHANNELS };

/******************************************************************
 *  Parameters a.k.a. problem sizes.
 *
 *  This descriptor will be passed to the kernel and therefore be
 *  stored within the registers.
 *****************************************************************/
struct Params {
    int32 num_rows;
    int32 num_idx;
    int32 num_x;
    int32 num_y;
    int32 num_out;
    int32 channels_x = 1;
    int32 channels_y = 1;
    Mode mode = Mode::OUTER;
    int32 unroll_x = 1;
    int32 unroll_y = 1;
    int32 unroll_z = 1;
    Layout layout = Layout::LEADING_CHANNELS;

    __host__ __device__
    int32 channels_z() const {
        switch(mode) {
            case Mode::OUTER:
                return channels_x * channels_y;
            case Mode::INNER:
                // TODO: should be one for I/O layouts,
                //       although working channels may be looked for.
                return channels_x;
            case Mode::MAP:
                return channels_x;
        }
    }

    __host__ __device__
    int32 channels_out() const {
        switch(mode) {
            case Mode::OUTER:
                return channels_x * channels_y;
            case Mode::INNER:
                return 1;
            case Mode::MAP:
                return channels_x;
        }
    }
};


/******************************************************************
 *  Packed coefficient: (val, i, j, k) per nonzero CG entry.
 *
 *  Passed from JAX as an opaque idx_t vector, reinterpret_cast
 *  to Coef* at the FFI boundary.
 *
 *  Alignment is computed from the raw field sizes (sizeof(Val) + 3*sizeof(Idx)),
 *  rounded up to the next power of two.  This gives:
 *    Idx=int32, Val=float  ->  4+12=16  -> alignas(16) -> LDS.128
 *    Idx=uint8, Val=float  ->  4+ 3= 7  -> alignas( 8) -> LDS.64
 *****************************************************************/
template<typename Idx, typename Val>
struct alignas(next_pow2(sizeof(Val) + 3*sizeof(Idx))) Coef {
    Val val;
    Idx i;
    Idx j;
    Idx k;
};


template<typename Idx, typename Val, Mode kMode>
e3j::Error launch(
    const Coef<Idx,Val> *coef,
    const Val *x,
    const Val *y,
    Val *out,
    Params p,
    cudaStream_t stream,
    int debug
);

template<typename Idx, typename Val>
e3j::Error launch(
    const Coef<Idx,Val> *coef,
    const Val *x,
    const Val *y,
    Val *out,
    Params p,
    cudaStream_t stream,
    int debug
);

} // namespace tensor_product
} // namespace e3j

#endif // E3J_CUDA_TENSOR_PRODUCT_H_
