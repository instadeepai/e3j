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

    // Cooperative find_coef_bounds: all threads race via atomicMin
    // to claim the nearest cut at each output-index boundary.
    // smem_cuts: int[blockDim.y + 1] scratch in shared memory.
    // Assumes num_coef < 2^16.
    template<typename Idx, typename Val, template<typename,typename> class CoefT=Coef>
    __device__ CoefRange
    find_coef_bounds(
        int *smem_cuts,
        const CoefT<Idx, Val> *coef,
        int num_coef,
        int num_out
    ) {
        int tid = threadIdx.x + threadIdx.y * blockDim.x;
        int stride = blockDim.x * blockDim.y;
        int num_warps = blockDim.y;
        int coefs_per_warp = (num_coef + num_warps - 1) / num_warps;

        if (num_warps == 1)
            return {0, num_coef};

        #pragma unroll 1
        for (int w = tid; w <= num_warps; w += stride)
            smem_cuts[w] = (w == 0) ? 0 : (w == num_warps) ? num_coef : INT_MAX;
        __syncthreads();

        #pragma unroll 1
        for (int i = tid; i < num_coef - 1; i += stride) {
            if (coef[i].i != coef[i + 1].i) {
                int cut = i + 1;
                int w = (cut + coefs_per_warp / 2) / coefs_per_warp;
                if (w >= 1 && w < num_warps
                    && coef[cut].i < num_out - (num_warps - w)) {
                    int dist = abs(cut - w * coefs_per_warp);
                    atomicMin(&smem_cuts[w], (dist << 16) | cut);
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            #pragma unroll 1
            for (int w = 1; w < num_warps; w++) {
                int packed = smem_cuts[w];
                int pos = (packed == INT_MAX) ? smem_cuts[w - 1] : (packed & 0xFFFF);
                smem_cuts[w] = max(pos, smem_cuts[w - 1]);
            }
        }
        __syncthreads();

        CoefRange result = {
            smem_cuts[threadIdx.y],
            smem_cuts[threadIdx.y + 1],
        };
        __syncthreads();
        return result;

    }

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
