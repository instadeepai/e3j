#include "cuda/convolution_bwd.cuh"
#include "cuda/convolution/launch_bwd.cuh"
#include "cuda/dispatch_macros.h"

namespace e3j {
namespace convolution {

#define FOR_EACH_DTYPE_PAIR(Idx, Val)              \
template e3j::Error launch_bwd<Idx, Val>(          \
    const Coef<Idx,Val> *coef,                     \
    const Val *x,                                  \
    const Val *y,                                  \
    const Val *dz,                                 \
    const Val *mix,                                \
    const Idx *irrep_out,                          \
    const AdjacencyCSR adj,                         \
    const int32_t *edge_perm,                      \
    Val *dx,                                       \
    Val *dy,                                       \
    Val *dmix,                                     \
    Params p,                                      \
    cudaStream_t stream,                           \
    int debug                                      \
);
__FOR_EACH_DTYPE_PAIR
#undef FOR_EACH_DTYPE_PAIR

} // namespace convolution
} // namespace e3j
