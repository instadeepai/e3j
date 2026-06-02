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
            .k   = mix_idx[coef[n].i],
            .l   = coef[n].k,
        };
    }
    return out;
}


/******************************************************************
 *  Host-side backward coefficient transpose: (Coef, mix_idx) -> Coef4D.
 *
 *  Permutes forward Coef4D indices (i, j, k, l) = (m, x, s, y)
 *  according to the given ordering, producing a new sorted Coef4D.
 *
 *  The ordering quadruple {a, b, c, d} selects which forward index
 *  goes into each Coef4D slot: out = {val, fwd[a], fwd[b], fwd[c], fwd[d]},
 *  then stable-sorted by out.i (= fwd[a]).
 *
 *  Backward orderings (dm first, broadcast operand last):
 *    dmix (2,0,1,3): {k, i, j, l}  for bigotimes(dm, x, y)  sorted by k (=s)
 *    dx   (1,0,2,3): {j, i, k, l}  for bigotimes(dm, s, y)  sorted by j (=x)
 *    dy   (3,0,2,1): {l, i, k, j}  for bigotimes(dm, s, x)  sorted by l (=y)
 *****************************************************************/
template<typename Idx, typename Val>
std::vector<Coef4D<Idx, Val>> transpose_coef4D(
    const std::vector<Coef<Idx, Val>> &coef,
    const Idx *mix_idx,
    int order[4]
) {
    int nnz = (int)coef.size();
    std::vector<Coef4D<Idx, Val>> out(nnz);
    for (int n = 0; n < nnz; n++) {
        Idx fwd[4] = { coef[n].i, coef[n].j, mix_idx[coef[n].i], coef[n].k };
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
