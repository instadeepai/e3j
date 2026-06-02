#ifndef _E3J_CONVOLUTION_FWD_H_
#define _E3J_CONVOLUTION_FWD_H_

#include "cuda/convolution.cuh"
#include "cuda/tensor_product/details.cuh"
#include "cuda/tensor_product/trailing_channels.cuh"
#include "cuda/utils.cuh"

#define BUFFER_CONV_FWD_COEFS_IN_SMEM true

namespace e3j {
namespace convolution {

using utils::copy;          // STG messages
using utils::copy_strided;  // STG messages
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
    }
};


/****************************************************************
 *  Accumulate trilinear products above a given output coordinate.
 *
 *  Returns current index and value for STG write-out.
 *  Updates accumulator in place for next call, i.e. next output
 *  index and zeroed-out value.
 *
 *  Caller is responsible for applying thread-wise pointer offsets,
 *  for each thread to process its own channels.
 ****************************************************************/
template<typename Idx, typename Val, Mode kMode, int N>
__device__ tp::IdxVal<Idx,Val,N> accumulate_trilinear(
    tp::IdxVal<Idx, Val, N> &acc,
    const Coef4D<Idx,Val> *coef,
    int &col,
    const int col_end,
    const Val* x1,
    const Val* x2,
    const Val* x3,
    unsigned int channels_x1,
    unsigned int channels_x2,
    unsigned int channels_x3
) {
    using IdxVal = tp::IdxVal<Idx,Val,N>;
    using Coef = Coef4D<Idx,Val>;
    IdxVal out = acc;

    // Sum all c.val * x1[c.j] * x2[c.k] * x3[c.l] above output index c.i.
    while (col < col_end) {

        // Load a sparse coefficient {val, i, j, k, l}
        Coef c = coef[col];
        col++;

        // Evaluate one summand `c.val * x1[c.j] * x2_[c.k] * x3[c.l]`,
        // then branch on output index c.i:
        //  - Reset accumulator with FMUL and return current sum if new
        //  - Update accumulator with FFMA otherwise
        //
        // OUTER:     Channels of last operand only are broadcast for now.
        //            => Call on (x, s, y) with channels_x = channels_s
        //               (node features, scalars) and channels_y = 1
        //               (harmonic embeddings).
        //
        // INNER/MAP: All operands are vectorized.
        if constexpr (kMode == Mode::OUTER) {
            Val x3_l = x3[c.l * channels_x3];
            Vect<N,Val> x2_k = load<N,Val>(x2 + c.k * channels_x2);
            Vect<N,Val> x1_j = load<N,Val>(x1 + c.j * channels_x1);
            // Multiply twice
            Vect<N,Val> prod = broadcast<N, Val>(c.val * x3_l);
            prod = mul<N, Val>(prod, x2_k);
            // Branch on output index to apply last mul
            if (c.i != acc.i) {
                acc = {c.i, mul<N,Val>(prod, x1_j)};
                return out;
            }
            fmadd<N,Val>(out.val, prod, x1_j);

        } else {
            Vect<N,Val> x1_j = load<N,Val>(x1 + c.j * channels_x1);
            Vect<N,Val> x2_k = load<N,Val>(x2 + c.k * channels_x2);
            Vect<N,Val> x3_l = load<N,Val>(x3 + c.l * channels_x3);
            // Multiply twice
            Vect<N,Val> prod = mul<N,Val>(broadcast<N, Val>(c.val), x1_j);
            prod = mul<N,Val>(prod, x2_k);
            if (c.i != acc.i) {
                acc = {c.i, mul<N,Val>(prod, x3_l)};
                return out;
            }
            fmadd<N,Val>(out.val, prod, x3_l);
        }
    }
    // Reached last coef, update counter to exit loop from caller.
    col++;
    return out;
}


/*************************************************************
 *  Reduce products over an input row and write out.
 *
 *  Because loading multiple rows in SMEM simultaneously may
 *  be too heavy on memory, this kernel part assumes that a
 *  block may contain multiple warps along blockDim.y processing
 *  the same inputs together, splitting coefficients over non-
 *  overlapping output coordinates.
 *
 *  Note: we have 64 warps / SM on H100. Therefore with K=64,
 *        we'd have 32 blocks / SM requesting shared memory if
 *        warps would only work on distinct channels.
 *        We could also bypass SMEM and load from GMEM.
 *
 ************************************************************/
template<typename Idx, typename Val, Mode kMode, int N, bool accumulate=false>
__device__ void bigotimes(
    const Coef4D<Idx,Val> *coef,
    const CoefRange range,
    CuArray2D<Val> x1,
    CuArray2D<Val> x2,
    CuArray2D<Val> x3,
    CuArray2D<Val> out,
    Val *scratch
) {

    // NOTE: consumes p.num_out, p.unroll_{x,y}, p.channels_x.
    using Coef = Coef4D<Idx, Val>;
    using IdxVal = tp::IdxVal<Idx, Val, N>;

    int col = range.begin;
    Coef c = coef[col];
    IdxVal zi,
           acc = {c.i, broadcast<N,Val>(Val(0))};

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    // Number of warps needed to sum all channels in INNER mode.
    int num_warps_x = 1 + (x1.shape[1] - 1) / (32 * N);

    // Apply threadwise channel offsets:
    //  - OUTER:        3rd operand only is broadcast
    //  - INNER | MAP:  all operands have same number of channels
    x1.data += threadIdx.x * N;
    x2.data += threadIdx.x * N;
    if constexpr (kMode == Mode::INNER || kMode == Mode::MAP)
        x3.data += threadIdx.x * N;

    if constexpr (kMode == Mode::OUTER || kMode == Mode::MAP) {
        // Prevent OOB threads from writing out.
        // With N chosen so channels_z/N >= 32, all threads are active.
        if (threadIdx.x * N < out.shape[1]) {
            Vect<N,Val> *out_lane =
                reinterpret_cast<Vect<N,Val>*>(out.data) + threadIdx.x;
            int stride_out = out.shape[1] / N;
            // For each i, accumulate outputs (i, z[i]) above i.
            // Jumped indices should be zeroed out from the outside.
            //
            // NOTE: col <= range.end important to not skip the case where
            //       last output index has only one coefficient. In that
            //       case, previous call to `accumulate_trilinear` has already
            //       read the coefficient. Accumulation then returns without
            //       loading out-of-bounds coefficient.
            while (col <= range.end) {
                // Compute z[i] = ∑ c[ijk] * x[j] * y[k] by channel.
                zi = accumulate_trilinear<Idx,Val,kMode,N>(
                    acc, coef, col, range.end,
                    x1.data, x2.data, x3.data,
                    x1.shape[1], x2.shape[1], x3.shape[1]
                );
                if constexpr (accumulate)
                    fmadd<N,Val>(
                        out_lane[zi.i * stride_out],
                        broadcast<N,Val>(Val(1)),
                        zi.val
                    );
                else
                    out_lane[zi.i * stride_out] = zi.val;
            }
        }

    } else if constexpr (kMode == Mode::INNER) {
        // NOTE: partial warp sums have to be accumulated in SMEM
        //       as we can't synchronize over blockDim.y without
        //       reaching deadlock within the loop over coefficients.
        while (col <= range.end) {
            // Sum z[i] over 32 channels at a time
            zi = accumulate_trilinear<Idx,Val,kMode,N>(
                acc, coef, col, range.end,
                x1.data, x2.data, x3.data,
                x1.shape[1], x2.shape[1], x3.shape[1]
            );
            // Horizontally sum N channels within each lane,
            // then reduce the 32-lane scalar across the warp.
            Val zi_scalar = tp::sum_warp(hsum<N,Val>(zi.val), 32);
            // STS for leading threads
            if (lane == 0) {
                // column-major to prevent bank conflicts
                if constexpr (accumulate)
                    scratch[warp * out.shape[0] + zi.i] += zi_scalar;
                else
                    scratch[warp * out.shape[0] + zi.i] = zi_scalar;
            }
        }
        if constexpr (!accumulate) {
            __syncthreads();
            // Combine warp-wide partial sums and write out
            for (unsigned int i = threadIdx.x; i < out.shape[0]; i += blockDim.x) {
                Val sum_zi = 0;
                for (int w = 0; w < num_warps_x; w++)
                    sum_zi += scratch[w * out.shape[0] + i];
                out.data[i] = sum_zi;
            }
        }
    }
}


template <typename Idx, typename Val, int N=1>
__global__ void kernel (
    const Coef4D<Idx, Val> *coef,
    CuArray2D<const Val> x,
    CuArray2D<const Val> y,
    CuArray2D<const Val> r,
    const AdjacencyCSR adj,
    CuArray2D<Val> z,
    int num_nodes,
    int num_coef,
    dim3 unroll
) {

    using Coef = Coef4D<Idx,Val>;

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

    // Distribute coefficients across blockDim.y, without z-index overlap.
    // Reuses lhs buffer (not yet populated) as int[blockDim.y + 1] scratch.
    CoefRange coef_range = find_coef_bounds<Idx, Val, Coef4D>(
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
                bigotimes<Idx,Val,Mode::OUTER,N,true>(
                    coef, coef_range,
                    smem.lhs, smem.mix, smem.rhs,
                    smem.out, nullptr
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
