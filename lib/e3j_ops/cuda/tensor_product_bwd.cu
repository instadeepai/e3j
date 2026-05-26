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
#include "cuda/tensor_product_bwd.cuh"
#include "cuda/tensor_product_bwd/trailing_channels.cuh"
#include "cuda/tensor_product_bwd/launch_trailing.cuh"

#include "cuda/dispatch_macros.h"
#include "ffi/error.h"


namespace e3j {
namespace tensor_product {

namespace trailing_channels {

template <typename Idx, typename Val>
e3j::Error launch_bwd(
    const Coef<Idx, Val> *coef,
    const Val *x,
    const Val *y,
    const Val *z,
    Val *dx,
    Val *dy,
    Params p,
    cudaStream_t stream,
    int debug
) {
    ModeBwd mode = GetModeBwd(
        p.mode, p.channels_x, p.channels_y
    );
    #define DISPATCH_MODE_PAIR(MODE_LHS, MODE_RHS)       \
    return launch_bwd<Idx, Val, MODE_LHS, MODE_RHS>(     \
        coef, x, y, z, dx, dy, p, stream, debug          \
    );

    #define DISPATCH_MODE_PAIR_ERROR(MODE_LHS, MODE_RHS) \
    return e3j::Error::InvalidArgument("incompatible modes");

    __DISPATCH_MODE_PAIR(mode.lhs, mode.rhs)
    #undef DISPATCH_MODE_PAIR
};


/* Specialize template for dtype pairs and mode using X macro. */

#define FOR_EACH_DTYPE_PAIR(Idx, Val)               \
template e3j::Error launch_bwd<Idx, Val>(           \
    const Coef<Idx, Val> *coef,                     \
    const Val *x,                                   \
    const Val *y,                                   \
    const Val *z,                                   \
    Val *dx,                                        \
    Val *dy,                                        \
    Params p,                                       \
    cudaStream_t stream,                            \
    int debug                                       \
);
__FOR_EACH_DTYPE_PAIR
#undef FOR_EACH_DTYPE_PAIR


} // namespace trailing_channels


ModeBwd GetModeBwd (Mode fwd, int channels_lhs, int channels_rhs) {
    switch(fwd) {
        case Mode::OUTER:
            if (channels_lhs == 1) {          /* (1, v) -> v */
                return {Mode::INNER, Mode::OUTER};
            } else if (channels_rhs == 1) {   /* (u, 1) -> u */
                return {Mode::OUTER, Mode::INNER};
            } else {
                printf("ERROR: Cannot backpropagate Mode::OUTER"
                       "if both inputs have multiple channels.");
                exit(1);
            }
            break;
        case Mode::INNER:   /* (u, u) -> 1 */
            return {Mode::OUTER, Mode::OUTER};
            break;
        case Mode::MAP:     /* (u, u) -> u */
            return {Mode::MAP, Mode::MAP};
            break;
        default:
            printf("ERROR: could not parse backward mode.");
            exit(1);
    }
}

ParamsBwd GetParamsBwd(Params p) {
    ModeBwd mode_bwd = GetModeBwd(
        p.mode, p.channels_x, p.channels_y
    );

    Params p_dx = {
        .num_rows = p.num_rows,
        .num_idx = p.num_idx,
        .num_x = p.num_out,
        .num_y = p.num_y,
        .num_out = p.num_x,
        .channels_x = p.channels_out(),
        .channels_y = p.channels_y,
        .mode = mode_bwd.lhs,
        .unroll_x = p.channels_out(),
        .unroll_y = p.channels_y,
        .unroll_z = 0, // PLACEHOLDER
        .layout = Layout::TRAILING_CHANNELS,
    };

    Params p_dy = {
        .num_rows = p.num_rows,
        .num_idx = p.num_idx,
        .num_x = p.num_out,
        .num_y = p.num_x,
        .num_out = p.num_y,
        .channels_x = p.channels_out(),
        .channels_y = p.channels_x,
        .mode = mode_bwd.rhs,
        .unroll_x = p.channels_out(),
        .unroll_y = p.channels_x,
        .unroll_z = 0, // PLACEHOLDER
        .layout = Layout::TRAILING_CHANNELS,
    };

    return ParamsBwd({
        .lhs = p_dx,
        .rhs = p_dy
    });

}


} // namespace tensor_product
} // namespace e3j
