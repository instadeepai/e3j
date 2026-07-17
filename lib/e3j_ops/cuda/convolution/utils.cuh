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

#ifndef _E3J_CONVOLUTION_UTILS_H_
#define _E3J_CONVOLUTION_UTILS_H_

#include <algorithm>
#include <numeric>
#include <vector>
#include "cuda/convolution.cuh"

namespace e3j {
namespace convolution {

/******************************************************************
 *  Host-side conversion: (Coef, mix_idx) -> Coef4D (forward).
 *
 *  Expands the 3-index CG coefficient {val, i, j, k} into a
 *  4-index {val, i, j, k, l} where l = mix_idx[i] is the scalar
 *  mixing index for the output coordinate i.
 *****************************************************************/
template<typename Idx, typename Val>
std::vector<Coef4D<Idx, Val>> otimes_mix_coefficients(
    const std::vector<Coef<Idx, Val>> &coef,
    const Idx *mix_idx
) {
    int nnz = (int)coef.size();
    std::vector<Coef4D<Idx, Val>> out(nnz);
    for (int n = 0; n < nnz; n++) {
        out[n] = {
            .val = coef[n].val,
            .i   = coef[n].i,
            .j   = coef[n].j,
            .k   = coef[n].k,
            .l   = mix_idx[coef[n].i]
        };
    }
    return out;
}


/******************************************************************
 *  Host-side backward coefficient transpose.
 *
 *  Permutes forward Coef4D indices according to the given ordering,
 *  producing a new sorted Coef4D by output index.
 *
 *  The ordering quadruple {a, b, c, d} selects which forward index
 *  goes into each Coef4D slot: out = {val, fwd[a], fwd[b], fwd[c], fwd[d]},
 *  then stable-sorted by out.i = fwd[a].
 *****************************************************************/
template<typename Idx, typename Val>
std::vector<Coef4D<Idx, Val>> transpose_coef4D(
    const std::vector<Coef4D<Idx, Val>> &coef,
    int order[4]
) {
    int nnz = (int)coef.size();
    std::vector<Coef4D<Idx, Val>> out(nnz);
    for (int n = 0; n < nnz; n++) {
        Idx fwd[4] = { coef[n].i, coef[n].j, coef[n].k, coef[n].l };
        out[n] = {
            coef[n].val,
            fwd[order[0]], fwd[order[1]], fwd[order[2]], fwd[order[3]]
        };
    }
    std::stable_sort(out.begin(), out.end(),
        [](const Coef4D<Idx, Val> &a, const Coef4D<Idx, Val> &b) {
            return a.i < b.i;
        }
    );
    return out;
}

} // namespace convolution
} // namespace e3j

#endif // _E3J_CONVOLUTION_UTILS_H_
