#ifndef E3J_CUDA_CONVOLUTION_BWD_H_
#define E3J_CUDA_CONVOLUTION_BWD_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda/convolution.cuh"
#include "ffi/error.h"

namespace e3j {
namespace convolution {

template <typename Idx, typename Val>
e3j::Error launch_bwd(
    const Coef<Idx, Val> *coef,
    const Val *x,
    const Val *y,
    const Val *dz,
    const Val *mix,
    const Idx *irrep_out,
    const AdjacencyCSR adj,
    const int32_t *edge_perm,
    Val *dx,
    Val *dy,
    Val *dmix,
    Params p,
    cudaStream_t stream,
    int debug
);

} // namespace convolution
} // namespace e3j

#endif // E3J_CUDA_CONVOLUTION_BWD_H_
