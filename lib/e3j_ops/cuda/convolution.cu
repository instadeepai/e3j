#include "cuda/convolution.cuh"
#include "cuda/convolution/launch_fwd.cuh"
#include "cuda/dispatch_macros.h"

namespace e3j {
namespace convolution {

#define FOR_EACH_DTYPE_PAIR(Idx, Val)              \
template e3j::Error launch<Idx, Val>(              \
    const Coef<Idx,Val> *coef,                     \
    const Val *x,                                  \
    const Val *y,                                  \
    const Val *r,                                  \
    const Idx *irrep_out,                          \
    const AdjacencyCSR adj,                        \
    Val *out,                                      \
    Params p,                                      \
    cudaStream_t stream,                           \
    int debug                                      \
);
__FOR_EACH_DTYPE_PAIR
#undef FOR_EACH_DTYPE_PAIR

} // namespace convolution
} // namespace e3j
