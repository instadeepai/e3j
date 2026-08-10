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

#ifndef _E3J_TENSOR_PRODUCT_DEBUG_H_
#define _E3J_TENSOR_PRODUCT_DEBUG_H_

#include "cuda/tensor_product/details.cuh"
#include "tests/vec.h"

#include <cuda.h>
#include <vector>
#include <iostream>

namespace e3j {
namespace tensor_product {

namespace debug {

    // printf("%f", ...) takes a double, and __half has no vararg conversion.
    static inline double to_double(float v)  { return static_cast<double>(v); }
    static inline double to_double(double v) { return v; }
    static inline double to_double(__half v) { return __half2float(v); }

    template<typename Idx, typename Val>
    int check_coefs(
        Coef<Idx,Val> *coef,
        Params p,
        bool strict = false
    ) {
        printf("Checking %d coefficients:\n", p.num_idx);

        using Coef = Coef<Idx, Val>;
        std::vector<Coef> coef_h = std::vector<Coef>(p.num_idx);
        cudaMemcpy(coef_h.data(), coef, p.num_idx * sizeof(Coef),
                   cudaMemcpyDeviceToHost);

        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess)
            printf("%s", cudaGetErrorString(error));

        bool all_in_bounds = true;

        for (int i=0; i < p.num_idx; i++) {
            bool in_bounds = true;
            Coef c = coef_h[i];
            printf("(%d, %d, %d) -> %f.0\n", c.i, c.j, c.k, to_double(c.val));
            in_bounds &= (c.i < p.num_out);
            in_bounds &= (c.j < p.num_x);
            in_bounds &= (c.k < p.num_y);
            if (not in_bounds) {
                all_in_bounds = false;
                printf("coef[%d] indices (%d, %d, %d) "
                       "is out of bounds (%d, %d, %d)\n",
                       i, c.i, c.j, c.k, p.num_out, p.num_x, p.num_y);
            }
        }
        if (not all_in_bounds and strict) {
            exit(1);
        }
        return all_in_bounds ? 0 : 1;
    }

}// namespace debug
}// namespace tensor_product
}// namespace e3j

#endif // _E3J_TENSOR_PRODUCT_DEBUG_H_
