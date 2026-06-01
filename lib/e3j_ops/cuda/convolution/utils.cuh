#ifndef _E3J_CONVOLUTION_UTILS_H_
#define _E3J_CONVOLUTION_UTILS_H_

#include <vector>
#include "cuda/convolution.cuh"

namespace e3j {
namespace convolution {

/******************************************************************
 *  Host-side conversion: (Coef, mix_idx) -> Coef4D.
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

} // namespace convolution
} // namespace e3j

#endif // _E3J_CONVOLUTION_UTILS_H_
