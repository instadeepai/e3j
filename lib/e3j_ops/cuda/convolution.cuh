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

// Same numpy byte-compatibility check as on `Coef`, see cuda/tensor_product.cuh.
static_assert(sizeof(Coef4D<std::int32_t, float>)  == 32, "Coef4D(int32,float)");
static_assert(sizeof(Coef4D<std::uint8_t, float>)  ==  8, "Coef4D(uint8,float)");
static_assert(sizeof(Coef4D<std::int32_t, double>) == 32, "Coef4D(int32,double)");
static_assert(sizeof(Coef4D<std::uint8_t, double>) == 16, "Coef4D(uint8,double)");
static_assert(sizeof(Coef4D<std::int32_t, __half>) == 32, "Coef4D(int32,half)");
static_assert(sizeof(Coef4D<std::uint8_t, __half>) ==  8, "Coef4D(uint8,half)");

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
 *  Computes a trilinear mixing of node features `x` (gathered by senders),
 *  edge features `y` and edge scalars `s`, accumulated on receiver nodes:
 *
 *          mⱼ= ∑ᵢ (xᵢ⊗ yᵢⱼ) ⊙  sᵢⱼ
 *
 *  The receiver messages `m` are returned, without materializing the
 *  intermediate edge-wise messages, and without resorting to atomic
 *  operations.
 *
 *  @param coef Clebsch-Gordan coefficients for the edge-wise tensor product.
 *  @param x node features.
 *  @param y edge features (typically harmonic embeddings of edge vectors).
 *  @param s edge scalars (typically MLP of RBF radial embeddings).
 *  @param adj adjacency matrix in CSR format.
 *  @param m output node features.
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
    const Val *s,
    const AdjacencyCSR adj,
    Val *m,
    Params p,
    cudaStream_t stream,
    int debug
);

} // namespace convolution
} // namespace e3j

#endif // E3J_CUDA_CONVOLUTION_H_
