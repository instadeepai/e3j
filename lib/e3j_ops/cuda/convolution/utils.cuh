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
#include <climits>
#include <cmath>
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


/******************************************************************
 *  Per-warp cuts of one coefficient pass, for the C++ tests.
 *
 *  Splits the coefficients into `num_warps` slices of similar size,
 *  cutting only where the output index `out_idx` (sorted ascending)
 *  changes, so that slices own disjoint output rows. Returns
 *  num_warps + 1 cut positions.
 *****************************************************************/
inline std::vector<int> find_coef_bounds_host(
    const int *out_idx, int num_coef, int num_warps
) {
    std::vector<int> cuts(num_warps + 1);
    cuts[0] = 0;
    cuts[num_warps] = num_coef;
    if (num_warps == 1)
        return cuts;
    int coefs_per_warp = (num_coef + num_warps - 1) / num_warps;
    for (int w = 1; w < num_warps; w++) {
        int target = w * coefs_per_warp;
        int previous_cut = cuts[w - 1];
        int best_cut = num_coef, best_distance = INT_MAX;
        for (int cut = previous_cut + 1; cut < num_coef; cut++) {
            if (out_idx[cut] == out_idx[cut - 1])
                continue;
            int distance = std::abs(cut - target);
            if (distance >= best_distance)
                break;                     // distance is convex: past the nearest
            best_distance = distance;
            best_cut = cut;
        }
        cuts[w] = best_cut;
    }
    return cuts;
}


/******************************************************************
 *  Backward per-warp coefficient ranges (kernel_bwd `coef_ranges`).
 *
 *  Cuts the dx, dy, ds passes independently with `find_coef_bounds_host`
 *  and returns 3 * num_warps end offsets laid out as [dx | dy | ds]. A
 *  warp's range begins at the previous warp's end offset, or at 0.
 *
 *  `coef_bwd` is the concatenated [dx | dy | ds] array of 3 * num_coef
 *  coefficients, each sorted by output index (`.i`).
 *****************************************************************/
template<typename Idx, typename Val>
std::vector<int> pack_coef_ranges(
    const Coef4D<Idx, Val> *coef_bwd, int num_coef, int num_warps
) {
    std::vector<int> ix(num_coef), iy(num_coef), is(num_coef);
    for (int c = 0; c < num_coef; c++) {
        ix[c] = coef_bwd[c].i;
        iy[c] = coef_bwd[num_coef + c].i;
        is[c] = coef_bwd[2*num_coef + c].i;
    }
    std::vector<int> cx = find_coef_bounds_host(ix.data(), num_coef, num_warps);
    std::vector<int> cy = find_coef_bounds_host(iy.data(), num_coef, num_warps);
    std::vector<int> cs = find_coef_bounds_host(is.data(), num_coef, num_warps);

    // Drop the leading 0 of each cut list: entry w is warp w's endpoint.
    std::vector<int> ranges(3 * num_warps);
    for (int w = 0; w < num_warps; w++) {
        ranges[0*num_warps + w] = cx[w + 1];
        ranges[1*num_warps + w] = cy[w + 1];
        ranges[2*num_warps + w] = cs[w + 1];
    }
    return ranges;
}

} // namespace convolution
} // namespace e3j

#endif // _E3J_CONVOLUTION_UTILS_H_
