#ifndef _E3J_CONVOLUTION_FWD_H_
#define _E3J_CONVOLUTION_FWD_H_

#include "cuda/convolution.cuh"
#include "cuda/tensor_product/details.cuh"
#include "cuda/tensor_product/trailing_channels.cuh"
#include "cuda/utils.cuh"

#define BUFFER_CONV_FWD_COEFS_IN_SMEM true

namespace e3j {
namespace convolution {

using utils::copy;          // LDG + STS mix_idx
using utils::copy_strided;  // STG summed messages
using utils::copy_pipe;
using utils::copy_pipe_strided;
using utils::wait_pipe;
using utils::fill;

using tensor_product::Coef;
using tensor_product::CoefRange;
using tensor_product::Mode;
using tensor_product::find_coef_bounds;

namespace tp = tensor_product::trailing_channels;

template <typename Idx, typename Val>
struct Buffers {
    CuArray2D<Val> lhs;
    CuArray2D<Val> rhs;
    CuArray2D<Val> out;
    CuArray2D<Val> mix;
    Idx* mix_idx;

    static __device__ Buffers init (
        CuArray2D<const Val> x,
        CuArray2D<const Val> y,
        CuArray2D<Val> z,
        CuArray2D<const Val> r,
        dim3 unroll
    ) {
        return Buffers {
            .lhs = { nullptr, x.shape[0], unroll.x },
            .rhs = { nullptr, y.shape[0], unroll.y },
            .out = { nullptr, z.shape[0], unroll.z },
            .mix = { nullptr, r.shape[0], unroll.z },
            .mix_idx = nullptr,
        };
    }

    template<int Stages=1>
    __device__ void to_shared (char *smem_) {
        // Round up to the next 128 bit multiple so buffers are
        // always aligned. Next N-multiple would only exceeds this
        // with double precision.
        constexpr int A = 16 / sizeof(Val);
        lhs.data = reinterpret_cast<Val*>(smem_);
        rhs.data = lhs.data + (Stages * lhs.size() + A-1) / A * A;
        out.data = rhs.data + (Stages * rhs.size() + A-1) / A * A;
        mix.data = out.data + (Stages * out.size() + A-1) / A * A;
        mix_idx = reinterpret_cast<Idx*>(mix.data + Stages * mix.size());
    }
};

/*****************************************************************
 *  Tensor product - scalar mixing FMADD fusion.
 *
 *  Output scalars are loaded from mix[mix_idx[z.i]] where z is
 *  computed with the same `accumulate_products` routine as the
 *  forward tensor product kernel and `otimes`.
 *
 *  Note: in addition to scalar mixing fusion, this function also
 *        updates its output buffer in place with utils::fmadd,
 *        since the message-passing kernel needs to accumulate
 *        messages.
 *
 *        Not-incrementing output buffer (direct STG instead of
 *        LDS + fmadd + STS) is not needed nor supported for now,
 *        though we could easily e.g. via template <bool ADD=true>.
 *****************************************************************/
template<typename Idx, typename Val, int N>
__device__ void otimes_mix(
    const Coef<Idx,Val> *coef,
    const CoefRange range,
    CuArray2D<Val> lhs,
    CuArray2D<Val> rhs,
    CuArray2D<Val> out,
    CuArray2D<Val> mix,
    const Idx *mix_idx
) {
    using Coef = Coef<Idx,Val>;
    using IdxVal = tp::IdxVal<Idx, Val, N>;

    int col = range.begin;
    Coef c = coef[col];
    IdxVal z_ab,
           acc = {c.i, broadcast<N,Val>(Val(0))};

    int channel_offset = threadIdx.x * N;
    lhs.data += channel_offset;
    mix.data += channel_offset;

    if (channel_offset < out.shape[1]) {

        Vect<N,Val> *out_lane = reinterpret_cast<Vect<N,Val>*>(out.data) + threadIdx.x;

        int stride_out = out.shape[1] / N;

        while (col <= range.end) {

            Idx idx_out = acc.i;
            Vect<N,Val> s_ab = load<N,Val>(&mix.data[mix_idx[idx_out] * mix.shape[1]]);
            Vect<N,Val> message_b = out_lane[idx_out * stride_out];

            // NOTE: z_ab.i == acc.i before the `accumulate_products` call,
            //       which returns the sum of products over i and updates
            //       its accumulator in place to i+1 when returning.

            z_ab = tp::accumulate_products<Idx,Val,Mode::OUTER,N>(
                acc, coef, col, range.end, lhs.data, rhs.data, lhs.shape[1], rhs.shape[1]
            );

            // STS fused output / current message_b.
            fmadd<N,Val>(message_b, s_ab, z_ab.val);
            out_lane[idx_out * stride_out] = message_b;
        }
    }
}


template <typename Idx, typename Val, int N=1>
__global__ void kernel (
    const Coef<Idx, Val> *coef,
    CuArray2D<const Val> x,
    CuArray2D<const Val> y,
    CuArray2D<const Val> r,
    const Idx *irrep_out,
    const AdjacencyCSR adj,
    CuArray2D<Val> z,
    int num_nodes,
    int num_coef,
    dim3 unroll
) {

    using Coef = Coef<Idx,Val>;

    int size_x = x.size();
    int size_y = y.size();
    int size_out = z.size();

    int num_strides = 1 + (max((int)x.shape[1], (int)z.shape[1]) - 1)
                         / (blockDim.x * N);

    // Shared memory layout: [ coef | x | y | z ] or [ x | y | z ]
    extern __shared__ __align__(16) char smem__[];
    char* smem_ = reinterpret_cast<char*>(smem__);
    Buffers<Idx, Val> smem = Buffers<Idx, Val>::init(
        x, y, z, r, unroll
    );

    if (BUFFER_CONV_FWD_COEFS_IN_SMEM) {
        Coef* smem_coef = reinterpret_cast<Coef*>(smem_);
        copy_pipe<1>(smem_coef, coef, num_coef);
        // Overwrite reference to GMEM `coef` with SMEM buffer.
        coef = smem_coef;
        // Align `smem_` to Vect<N, Val> for LDS.64 / LDS.128
        constexpr size_t align = 16;
        size_t coef_bytes = (size_t)num_coef * sizeof(Coef);
        coef_bytes =  (coef_bytes + align - 1) & ~(align - 1);
        smem_ = (char*)smem_coef + coef_bytes;
        // Wait for coef copy.
        __pipeline_commit();
        wait_pipe();
    }
    // Allocate SMEM buffer addresses.
    smem.to_shared(smem_);

    // Load irrep_out into SMEM (constant across edges).
    // Regular copy — cp.async requires size >= 4 bytes, Idx may be uint8_t.
    copy(smem.mix_idx, irrep_out, z.shape[0]);
    __syncthreads();

    // Distribute coefficients across blockDim.y, without z-index overlap.
    // Reuses lhs buffer (not yet populated) as int[blockDim.y + 1] scratch.
    CoefRange coef_range = find_coef_bounds<Idx, Val>(
        reinterpret_cast<int*>(smem.lhs.data), coef, num_coef, z.shape[0]
    );

    // First receiver for this block.
    z.data += blockIdx.x * size_out;

    //=== Grid stride loop ===
    for (unsigned int receiver = blockIdx.x; receiver < num_nodes; receiver += gridDim.x) {

        // Stride-local GMEM views, advanced per channel slice.
        CuArray2D<const Val> x_s = { x.data, x.shape[0], x.shape[1] };
        CuArray2D<const Val> r_s = { r.data, r.shape[0], r.shape[1] };
        CuArray2D<Val> z_s = { z.data, z.shape[0], z.shape[1] };

        //=== Channel stride loop ===
        for (int s = 0; s < num_strides; s++) {

            // Zero the SMEM accumulator for this channel slice.
            fill(smem.out.data, Val(0), smem.out.size());
            __syncthreads();

            int first_edge = adj.receiver_ptr[receiver],
                last_edge  = adj.receiver_ptr[receiver + 1];

            for (int edge = first_edge; edge < last_edge; edge++) {

                int sender = adj.sender[edge];

                // Load sender features, edge embeddings, and radial scalars.
                if (x.shape[1] > unroll.x)
                    copy_pipe_strided(
                        smem.lhs.data, x_s.data + (sender * size_x),
                        x.shape[0], unroll.x, x.shape[1]
                    );
                else
                    copy_pipe<N>(smem.lhs.data, x_s.data + (sender * size_x), smem.lhs.size());

                // y has no channel dimension (channels_y = 1), never strided.
                copy_pipe<1>(smem.rhs.data, y.data + (edge * size_y), size_y);

                if (r.shape[1] > unroll.z)
                    copy_pipe_strided(
                        smem.mix.data, r_s.data + (edge * r.size()),
                        r.shape[0], unroll.z, r.shape[1]
                    );
                else
                    copy_pipe<N>(smem.mix.data, r_s.data + (edge * r.size()), smem.mix.size());

                __pipeline_commit();
                wait_pipe();

                // Fused TP reduction + scalar mixing, accumulates into smem.out.
                otimes_mix<Idx,Val,N>(
                    coef, coef_range, smem.lhs, smem.rhs,
                    smem.out, smem.mix, smem.mix_idx
                );
                __syncthreads();
            }

            // Store aggregated output to GMEM once per receiver.
            if (z.shape[1] > unroll.z)
                copy_strided(
                    z_s.data, smem.out.data,
                    z.shape[0], smem.out.shape[1],
                    z.shape[1], smem.out.shape[1]
                );
            else
                copy(z_s.data, smem.out.data, smem.out.size());
            __syncthreads();

            // Advance to next channel slice (y is never strided).
            if (x.shape[1] > unroll.x) x_s.data += unroll.x;
            if (z.shape[1] > unroll.z) z_s.data += unroll.z;
            if (r.shape[1] > unroll.z) r_s.data += unroll.z;

        }//=== Channel stride loop ===

        // Advance to next receiver for this block.
        z.data += gridDim.x * size_out;

    }//=== Grid-stride loop ===

}


} // namespace convolution
} // namespace e3j


#endif // _E3J_CONVOLUTION_FWD_H_
