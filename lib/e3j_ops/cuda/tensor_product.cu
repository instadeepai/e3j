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

#include "cuda/tensor_product.cuh"
#include "cuda/tensor_product/details.cuh"
#include "cuda/tensor_product/launch_trailing.cuh"
#include "cuda/tensor_product/leading_channels.cuh"

#include "cuda/dispatch_macros.h"

namespace e3j {
namespace tensor_product {

template <typename Idx, typename Val, Mode kMode>
e3j::Error launch(
    const Coef<Idx,Val> *coef,
    const Val *x,
    const Val *y,
    Val *out,
    Params p,
    cudaStream_t stream,
    int debug
) {
    switch(p.layout) {

        case Layout::TRAILING_CHANNELS: {
            return trailing_channels::launch<Idx,Val,kMode>
                (coef, x, y, out, p, stream, debug);
        }

        case Layout::LEADING_CHANNELS:
            return leading_channels::launch<Idx,Val,kMode>
                (coef, x, y, out, p, stream, debug);

        default:
            return e3j::Error::InvalidArgument("Invalid layout.");
    }
}


template<typename Idx, typename Val>
e3j::Error launch(
    const Coef<Idx,Val> *coef,
    const Val *x,
    const Val *y,
    Val *out,
    Params p,
    cudaStream_t stream,
    int debug
) {
    #define DISPATCH_MODE(MODE)                      \
    return launch<Idx, Val, MODE>(                   \
        coef, x, y, out, p, stream, debug            \
    );

    #define DISPATCH_MODE_ERROR(MODE) \
    return e3j::Error::InvalidArgument("Invalid mode: " #MODE " is not supported.");

    __DISPATCH_MODE(p.mode)
    #undef DISPATCH_MODE
    #undef DISPATCH_MODE_ERROR
}

/* Specialize template for all registered dtype pairs */
#define FOR_EACH_DTYPE_PAIR(Idx, Val)              \
template e3j::Error launch<Idx, Val>(              \
    const Coef<Idx,Val> *coef,                     \
    const Val *x,                                  \
    const Val *y,                                  \
    Val *out,                                      \
    Params p,                                      \
    cudaStream_t stream,                           \
    int debug                                      \
);
__FOR_EACH_DTYPE_PAIR
#undef FOR_EACH_DTYPE_PAIR

} // namespace tensor_product
} // namespace e3j
