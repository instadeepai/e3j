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

#ifndef _E3J_CONVOLUTION_BWD_H_
#define _E3J_CONVOLUTION_BWD_H_

#include "cuda/convolution.cuh"
#include "cuda/convolution/fwd.cuh"
#include "cuda/tensor_product/details.cuh"
#include "cuda/tensor_product/trailing_channels.cuh"
#include "cuda/utils.cuh"

#define BUFFER_CONV_BWD_COEFS_IN_SMEM true

namespace e3j {
namespace convolution {

using utils::copy;
using utils::copy_strided;
using utils::copy_pipe;
using utils::copy_pipe_strided;
using utils::wait_pipe;
using utils::fill;
using tensor_product::CoefRange;

namespace tp = tensor_product::trailing_channels;


// Align to 16B: sizeof(T) should divide 16.
template<typename T>
__device__ unsigned int align16(unsigned int n) {
    constexpr int A = 16 / sizeof(T);
    return (n + A - 1) / A * A;
}


/*****************************************************************
 *  Buffers for backward convolution:
 *
 *      [ dm | x | y | mix | dy_scratch ]
 *
 *  - dm:   cotangent receiver messages
 *  - x:    primal node features
 *  - y:    primal edge features
 *  - mix:  edge scalars
 *  - dy_scratch:  cotangent edge features scratch memory (Mode::INNER)
 *
 *  Note dx is accumulated directly into GMEM via L2, since direct SMEM
 *  accumulation would be too greedy in the case of large channel counts.
 *
 *****************************************************************/
template <typename Idx, typename Val>
struct BuffersBwd {
    CuArray2D<Val> dm;
    CuArray2D<Val> x;
    CuArray2D<Val> y;
    CuArray2D<Val> mix;
    Val* dy_scratch;

    static __device__ BuffersBwd init (
        CuArray2D<const Val> x,
        CuArray2D<const Val> y,
        CuArray2D<const Val> dm,
        CuArray2D<const Val> mix,
        dim3 unroll
    ) {
        return BuffersBwd {
            .dm  = { nullptr, dm.shape[0],  unroll.z },
            .x   = { nullptr, x.shape[0],   unroll.x },
            .y   = { nullptr, y.shape[0],   1u       },
            .mix = { nullptr, mix.shape[0], unroll.z },
            .dy_scratch = nullptr,
        };
    }

    template<int Stages=1>
    __device__ void to_shared (char *smem_) {
        dm.data  = reinterpret_cast<Val*>(smem_);
        x.data   = dm.data  + align16<Val>(dm.size());
        y.data   = x.data   + align16<Val>(x.size());
        mix.data = y.data   + align16<Val>(y.size());
        dy_scratch = mix.data + align16<Val>(mix.size());
    }
};


/*****************************************************************
 *  Convolution backward kernel.
 *
 *  Sender-grouped loop over the transposed CSR adjacency:
 *    - x[sender] loaded once per sender group
 *    - dx accumulated in SMEM across edges (OUTER), one GMEM
 *      store per sender — no atomics
 *    - dy written per-edge to GMEM (INNER warp reduction)
 *
 *  The backward coefficients should be packed as a
 *  triple (coef_dx, coef_dy, coef_ds) such that:
 *
 *   - dx = bigotimes(coef_dx, dm, y, s)
 *   - dy = bigotimes(coef_dy, dm, x, s)
 *   - ds = bigotimes(coef_ds, dm, y, x)
 *
 *****************************************************************/
template <typename Idx, typename Val, int N=1>
__global__ void kernel_bwd (
    const Coef4D<Idx, Val> *coef,
    CuArray2D<const Val> x,
    CuArray2D<const Val> y,
    CuArray2D<const Val> s,
    CuArray2D<const Val> dm,
    const AdjacencyCSR adj,
    const int32_t *edge_perm,
    Val *gmem_dx,
    Val *gmem_dy,
    Val *gmem_ds,
    unsigned int num_nodes,
    unsigned int num_edges,
    unsigned int num_coef,
    dim3 unroll,
    const int *coef_ranges
) {

    using Coef = Coef4D<Idx,Val>;

    int size_x   = x.size();
    int num_y    = y.shape[0];
    int num_warps = blockDim.x / 32;

    // --- Shared memory prolog ----------------------------------------

    extern __shared__ __align__(16) char smem__[];
    char* smem_ = reinterpret_cast<char*>(smem__);
    BuffersBwd<Idx, Val> smem = BuffersBwd<Idx, Val>::init(
        x, y, dm, s, unroll
    );

    if (BUFFER_CONV_BWD_COEFS_IN_SMEM) {
        Coef* smem_coef = reinterpret_cast<Coef*>(smem_);
        copy_pipe<1>(smem_coef, coef, 3 * num_coef);
        coef = smem_coef;
        // Advance past the coef buffer, rounding up to 16 B so the dm/x/y/
        // mix/dx buffers stay 16-byte aligned for LDS.128 and the int4
        // cp.async copies (copy_pipe_strided always issues 16 B transactions
        // regardless of N).
        size_t coef_bytes = (size_t)(3 * num_coef) * sizeof(Coef);
        coef_bytes = utils::smem_align(coef_bytes);
        smem_ = (char*)smem_coef + coef_bytes;
        __pipeline_commit();
        wait_pipe();
    }
    smem.to_shared(smem_);

    const Coef *coef_dx = coef;
    const Coef *coef_dy = coef + num_coef;
    const Coef *coef_ds = coef + 2 * num_coef;

    // Assign coefficient along blockDim.y from host-computed coefficient ranges
    unsigned int tidy = threadIdx.y;
    const int *coef_ranges_dx = coef_ranges,
              *coef_ranges_dy = coef_ranges + blockDim.y,
              *coef_ranges_ds = coef_ranges + blockDim.y * 2;
    CoefRange range_dx = { tidy ? coef_ranges_dx[tidy - 1] : 0, coef_ranges_dx[tidy] },
              range_dy = { tidy ? coef_ranges_dy[tidy - 1] : 0, coef_ranges_dy[tidy] },
              range_ds = { tidy ? coef_ranges_ds[tidy - 1] : 0, coef_ranges_ds[tidy] };

    // Dummy view: only shape[0] = num_y is used (scratch indexing).
    CuArray2D<Val> dy_view = {
        nullptr, (unsigned int)num_y, smem.dm.shape[1]
    };

    int num_strides = 1 + (max((int)x.shape[1], (int)dm.shape[1]) - 1)
                         / (blockDim.x * N);

    int size_ds = s.shape[0] * dm.shape[1];

    // --- Grid-stride loop over senders -------------------------------

    for (unsigned int sender = blockIdx.x;
         sender < num_nodes;
         sender += gridDim.x)
    {
        int first_edge = adj.receiver_ptr[sender],
            last_edge  = adj.receiver_ptr[sender + 1];

        // Zero this sender's dx row before accumulating edges in place.
        // Covers coef-less ("jumped") output rows that dx never writes.
        fill<4>(gmem_dx + (size_t)sender * size_x, Val(0), size_x);
        __syncthreads();

        for (int edge = first_edge; edge < last_edge; edge++) {

            // Gather edge features by `edge_t` in the backward pass
            // since transposing the graph's CSR adjacency requires to
            // order edges by senders (instead of receivers).
            //
            // NOTE: We could also assume edge transposition acts by (-1)^p
            //       by equivariance of edge features. It is worth checking
            //       first how much overhead is incurred by the permuted
            //       gathering of edge features.
            size_t edge_t = (size_t)(edge_perm ? edge_perm[edge] : edge);

            int receiver = adj.sender[edge];

            // Skip padding edges
            if (receiver < 0 or receiver >= (int)num_nodes)
                continue;

            // Zero shared dy accumulator for INNER mode channel reduction
            fill<4>(smem.dy_scratch, Val(0), num_warps * num_y);
            __syncthreads();

            //=== Channel stride loop ===
            for (int channel0 = 0; channel0 < num_strides; channel0++) {

                int offset_x = (x.shape[1]  > unroll.x) ? channel0 * (int)unroll.x : 0;
                int offset_z = (dm.shape[1] > unroll.z) ? channel0 * (int)unroll.z : 0;

                // Translated pointers to current channel slice
                CuArray2D<const Val> x_  = { x.data  + offset_x, x.shape[0],  x.shape[1]  };
                CuArray2D<const Val> dm_ = { dm.data + offset_z, dm.shape[0], dm.shape[1] };
                CuArray2D<const Val> s_  = { s.data  + offset_z, s.shape[0],  s.shape[1]  };

                smem.x.load<4>(x_, sender);
                smem.dm.load<4>(dm_, receiver);
                smem.y.load<1>(y, edge_t);
                smem.mix.load<4>(s_, edge_t);
                __pipeline_commit();
                wait_pipe<0>();

                // Edge feature cotangents (dy)
                // Accumulates over channel in SMEM before writing out.
                bigotimes<Idx,Val,Mode::INNER,N,true>(
                    coef_dy, range_dy,
                    smem.dm, smem.x, smem.mix,
                    dy_view, smem.dy_scratch
                );

                // Edge scalar cotangents (ds)
                // Writes directly to GMEM.
                CuArray2D<Val> ds_view = {
                    gmem_ds + edge_t * size_ds + offset_z, s.shape[0], dm.shape[1]
                };
                bigotimes<Idx,Val,Mode::OUTER,N,false>(
                    coef_ds, range_ds,
                    smem.dm, smem.y, smem.x,
                    ds_view, nullptr
                );

                // Sender feature cotangents (dx)
                // Accumulates in GMEM via L2 because of the channel stride loop.
                // No race conditions given the nested loops over nodes and neighbours.
                CuArray2D<Val> dx_view = {
                    gmem_dx + (size_t)sender * size_x + offset_x,
                    x.shape[0], x.shape[1]
                };
                bigotimes<Idx,Val,Mode::OUTER,N,true>(
                    coef_dx, range_dx,
                    smem.dm, smem.y, smem.mix,
                    dx_view, nullptr
                );

                // Finish SMEM reads before the next slice overwrites them.
                __syncthreads();

            }//=== Channel stride loop ===

            // Reduce dy over the channel axis
            int tid = threadIdx.x + blockDim.x * threadIdx.y,
                blocksize = blockDim.x * blockDim.y;
            for (int i = tid; i < num_y; i += blocksize) {
                Val sum = 0;
                for (int w = 0; w < num_warps; w++)
                    sum += smem.dy_scratch[w * num_y + i];
                gmem_dy[edge_t * num_y + i] = sum;
            }
            __syncthreads();

        }//=== Edge loop ===

    }//=== Grid-stride loop ===

    // Zero cotangents of padding edges
    for (unsigned int edge = blockIdx.x; edge < num_edges; edge += gridDim.x) {
        int receiver = adj.sender[edge];
        if (receiver >= 0 and receiver < (int)num_nodes)
            continue;
        size_t edge_t = (size_t)(edge_perm ? edge_perm[edge] : edge);
        fill<1>(gmem_dy + edge_t * num_y, Val(0), num_y);
        fill<1>(gmem_ds + edge_t * size_ds, Val(0), size_ds);
    }

}

} // namespace convolution
} // namespace e3j


#endif // _E3J_CONVOLUTION_BWD_H_
