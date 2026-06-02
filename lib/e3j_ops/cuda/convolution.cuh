#ifndef E3J_CUDA_CONVOLUTION_H_
#define E3J_CUDA_CONVOLUTION_H_

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda/tensor_product.cuh"

namespace e3j {
namespace convolution {

using std::int32_t;
using e3j::tensor_product::Coef;
using e3j::tensor_product::Mode;

template<typename Idx, typename Val>
struct alignas(next_pow2(sizeof(Val) + 4*sizeof(Idx)))
Coef4D {
    Val val;
    Idx i; Idx j; Idx k; Idx l;
};

// Note: only Layout::TRAILING_CHANNELS supported.
struct Params {
    int32_t num_nodes;
    int32_t num_coef;
    int32_t num_x;
    int32_t num_y;        // (L+1)^2
    int32_t num_out;
    int32_t num_scalars;  // radial_embedding.numel
    int32_t channels_x;   // channels_y = 1
};

/****************************************************************************
 *  Compressed Sparse Row (CSR) format for the adjacency matrix.
 *
 *  Edges are grouped by receivers (rows of the adjacency format) to aggregate
 *  messages without atomics. The CSR format allows thread blocks to efficiently
 *  distribute groups of edges without having to scan receiver indices.
 *
 *  The number of nodes is not stored here to avoid redundancies with the
 *  `Params` struct.
 *****************************************************************************/
struct AdjacencyCSR {
    int32_t *sender;
    int32_t *receiver_ptr;
};

/****************************************************************************
 *  Convolution of node features with equivariant edge embeddings.
 *
 *  Sender node features `x_i` are first gathered from node features `x`.
 *  On each edge, the tensor product of `x_i` with spherical embeddings `y` is then
 *  computed, and multiplied by the radial embedding `r` (scalar mixing).
 *  The messages are finally scatter-reduced by receiver indices to produce the
 *  output node features.
 *
 *  @param coef Clebsch-Gordan coefficients for the edge-wise tensor product.
 *  @param x node features.
 *  @param y spherical embeddings of edge vectors.
 *  @param r radial embeddings of edge vectors.
 *  @param adj adjacency matrix in CSR format.
 *  @param out output node features.
 *  @param p additional problem parameters.
 *  @param stream CUDA stream (default 0)
 *  @param debug flag for stricter error catching and verbose logging.
 *
 *****************************************************************************/
template <typename Idx, typename Val>
e3j::Error launch(
    const Coef4D<Idx, Val> *coef,
    const Val *x,
    const Val *y,
    const Val *r,
    const AdjacencyCSR adj,
    Val *out,
    Params p,
    cudaStream_t stream,
    int debug
);

} // namespace convolution
} // namespace e3j

#endif // E3J_CUDA_CONVOLUTION_H_
