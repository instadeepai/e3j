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

#ifndef _E3J_TENSOR_PRODUCT_TRAILING_H_
#define _E3J_TENSOR_PRODUCT_TRAILING_H_

#include "cuda/tensor_product.cuh"
#include "cuda/tensor_product/details.cuh"
#include "cuda/utils.cuh"
#include "cuda/array.cuh"

#define WARP_MASK 0xffffffff
#define BUFFER_COEFS_IN_SMEM true

namespace e3j {
namespace tensor_product {

namespace trailing_channels {

    using utils::copy_pipe;
    using utils::copy_pipe_strided;
    using utils::wait_pipe;
    using utils::fill;
    using utils::DeviceProperties;

    // Alias of int4, mapping threads to I/O channels.
    //
    // N.B. `channel.z` may reflect a temporary output channel index,
    // prior to eventual aggregation in the final output array.
    //
    // TODO: get rid of `.total`
    struct Channels {
        int x;
        int y;
        int z;
        int total;

        // Compute thread-wise channel addresses from I/O channel dimensions.
        template<Mode kMode, int N>
        static __device__ Channels get(
            int tid,
            unsigned int channels_x,
            unsigned int channels_y,
            unsigned int channels_z
        ) {
            // Base output-channel index for this lane.
            // Each lane owns N consecutive channels, so lane (tid+offset)
            // starts at channel (tid+offset)*N.
            // `offset` shifts the base when blockIdx.y > 0 (multi-block striding).
            int k_z = tid * N;
            if constexpr (kMode == Mode::OUTER) {
                return {
                    .x = k_z % channels_x,
                    .y = k_z / channels_x,
                    .z = k_z,
                    .total = channels_x * channels_y,
                };
            } else if constexpr (kMode == Mode::INNER) {
                return {
                    .x = k_z,
                    .y = k_z,
                    .z = k_z,
                    .total = channels_x,
                };
            } else if constexpr (kMode == Mode::MAP) {
                return {
                    .x = k_z,
                    .y = k_z,
                    .z = k_z,
                    .total = channels_x,
                };
            }
        }
    };

    // Shared memory buffers layout.
    //
    // Use `.to_shared(char* smem_)` to initialize addresses
    // on the extern dynamically allocated char* SMEM pointer:
    //
    //   Buffers gmem {x_gmem, y_gmem, z_gmem};
    //
    //   extern __shared__ char* smem_;
    //   Buffers smem = gmem.to_shared(smem_);
    //   CuArray2D<T> x_smem = smem.lhs;
    //   CuArray2D<T> y_smem = smem.rhs;
    //   CuArray2D<T> z_smem = smem.out;
    //
    // From the kernel launcher, use `gmem.size_shared(stages)` to
    // infer SMEM footprint.
    template <typename T>
    struct Buffers {

        CuArray2D<T> lhs;
        CuArray2D<T> rhs;
        CuArray2D<T> out;

        // Initialize array shapes from GMEM arrays and unroll constraints.
        // NOTE: unroll.{x,y,z} already contains the vectorization parameter N.
        static __device__ Buffers init(
            CuArray2D<const T> x, CuArray2D<const T> y, CuArray2D<T> z, dim3 unroll
        ) {
            return Buffers {
                .lhs = { nullptr, x.shape[0], unroll.x },
                .rhs = { nullptr, y.shape[0], unroll.y },
                .out = { nullptr, z.shape[0], unroll.z },
            };
        }

        // Prepare pointers in SMEM memory given I/O dimension passed
        // e.g. as an initial pointer of GMEM arrays.
        // Prepare pointers in SMEM memory given I/O dimension passed
        // e.g. as an initial pointer of GMEM arrays.
        template <int Stages=1>
        __device__ void to_shared(char* smem_) {
            // Round up to the next N multiple so buffers are Vect<N,T>
            // aligned for vectorized loads (LDS.64 / LDS.128)
            constexpr int A = 16 / sizeof(T);
            lhs.data = reinterpret_cast<T*>(smem_);
            rhs.data = lhs.data + (Stages * lhs.size() + A-1) / A * A;
            out.data = rhs.data + (Stages * rhs.size() + A-1) / A * A;
        }

        // Get SMEM footprint.
        // N.B. `stages` doesn't have to be a template parameter in host code.
        size_t size_shared(Mode mode, int stages=1) {
            constexpr int A = 16 / sizeof(T);
            size_t size = mode == Mode::INNER
                ? sizeof(T) * out.size() / 32
                : 0;
            size += sizeof(T) * (stages * lhs.size() + A - 1) / A * A;
            size += sizeof(T) * (stages * rhs.size() + A - 1) / A * A;
            return size;
        }

    };


    // Pair type (idx, value) for output accumulator.
    //
    // This is the return type of the `accumulate_products` routine,
    // which sums tensor-product summand until a new output coordinate
    // is encountered.
    //
    // The `IdxVal` also serves as a stateful accumulator passed by
    // reference, so that the first load with distinct output coordinate
    // is stored for next call, while the computed sum on current coordinate
    // is returned.
    template<typename Idx, typename Val, int N>
    struct IdxVal {
        Idx i;
        Vect<N, Val> val;
    };

    /****************************************************************
     *  Accumulate all products above a given output coordinate.
     *
     *  Returns current index and value for STG write-out.
     *  Updates accumulator in place for next call, i.e. next output
     *  index and zeroed-out value.
     *
     *  Caller is responsible for applying thread-wise pointer offsets,
     *  for each thread to process its own channels.
     ****************************************************************/
    template<typename Idx, typename Val, Mode kMode, int N>
    __device__ IdxVal<Idx,Val,N> accumulate_products(
        IdxVal<Idx, Val, N> &acc,
        const Coef<Idx,Val> *coef,
        int &col,
        const int col_end,
        const Val* lhs,
        const Val* rhs,
        unsigned int channels_lhs,
        unsigned int channels_rhs
    ) {
        using IdxVal = IdxVal<Idx,Val,N>;
        using Coef = Coef<Idx,Val>;
        IdxVal out = acc;
        // Sum all c_ijk * xj * yk above output index out.i
        // Each lane loads N channels via Vect (guaranteed active by caller).
        while (col < col_end) {
            // Load a sparse coefficient {i, j, k, c_ijk} (LDS.128)
            Coef c = coef[col];
            col++;
            // Evaluate one summand `c.val * x_j * y_k`.
            // Branch on c.i is hoisted before the accumulation so the common path
            // (same output index) hits a single FFMA per channel instead of FMUL+FADD.
            // The rare path (new output index) falls back to plain FMUL to init acc.
            //
            // OUTER:     channels_x % N == 0 keeps all N z-channels in one y-channel,
            //            so y_k is scalar and x_j is Vect<N> (scalar-vector FFMA).
            // TODO:      load Vect<N> y_k per lane to compute N*N outputs at once,
            //            eliminating redundant x_j reloads across y-channels (WIP).
            // INNER/MAP: xy = x_j*y_k (Vect<N>),  FFMA: out[v] += c.val * xy[v]
            if constexpr (kMode == Mode::OUTER) {
                Val y_k = rhs[c.k * channels_rhs];
                Val cy = c.val * y_k;
                Vect<N,Val> x_j = load<N,Val>(lhs + c.j * channels_lhs);
                if (c.i != acc.i) {
                    acc = {c.i, mul<N,Val>(x_j, broadcast<N,Val>(cy))};
                    return out;
                }
                fmadd<N,Val>(out.val, cy, x_j);
            } else {
                Vect<N,Val> x_j = load<N,Val>(lhs + c.j * channels_lhs);
                Vect<N,Val> y_k = load<N,Val>(rhs + c.k * channels_rhs);
                Vect<N,Val> xy = mul<N,Val>(x_j, y_k);
                if (c.i != acc.i) {
                    acc = {c.i, mul<N,Val>(xy, broadcast<N,Val>(c.val))};
                    return out;
                }
                fmadd<N,Val>(out.val, c.val, xy);
            }
        }
        // Reached last coef, update counter to exit loop from caller.
        col++;
        return out;
    }

    //  Sum register values within a warp.
    //
    //  Only used in Mode::INNER.
    //
    //  TODO: might be equivalent to __reduce_add_sync(FULL_WARP, x).
    //        however with width < 32, we don't get correct results
    //        by summing over the full warp.
    template<typename Val, bool broadcast=false>
    __device__ Val sum_warp(Val x, int width=32) {
        #pragma unroll
        for (unsigned int s = 1; s < width; s <<=1) {
            Val xs = __shfl_down_sync(WARP_MASK, x, s);
            x += xs;
        }
        if (broadcast)
            x = __shfl_sync(WARP_MASK, x, 0);
        return x;
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
    __device__ void otimes(
        const Coef<Idx,Val> *coef,
        const CoefRange range,
        CuArray2D<Val> lhs,
        CuArray2D<Val> rhs,
        CuArray2D<Val> out,
        const Channels k,
        Val *scratch
    ) {

        // NOTE: consumes p.num_out, p.unroll_{x,y}, p.channels_x.
        using Coef = Coef<Idx, Val>;
        using IdxVal = IdxVal<Idx, Val, N>;

        int col = range.begin;
        Coef c = coef[col];
        IdxVal zi,
               acc = {c.i, broadcast<N,Val>(Val(0))};

        int lane = threadIdx.x % 32;
        int warp = threadIdx.x / 32;
        // Number of warps needed to sum all channels in INNER mode.
        int num_warps_x = 1 + (lhs.shape[1] - 1) / (32 * N);

        // Apply threadwise channel offsets
        lhs.data += k.x;
        rhs.data += k.y;

        if constexpr (kMode == Mode::OUTER || kMode == Mode::MAP) {
                // OOB guard hoisted outside the coefficient loop.
                // With N chosen so channels_z/N >= 32, all threads are
                // active (k.z < k.total) and this branch is uniform — no divergence.
                // The guard is kept as a safety net for edge cases.
                if (k.z < k.total) {
                    Vect<N,Val> *out_lane =
                        reinterpret_cast<Vect<N,Val>*>(out.data) + threadIdx.x;
                    int stride_out = out.shape[1] / N;
                    // For each i, accumulate outputs (i, z[i]) above i.
                    // Jumped indices should be zeroed out from the outside.
                    //
                    // NOTE: col <= range.end important to not skip the case where
                    //       last output index has only one coefficient. In that
                    //       case, previous call to `accumulate_products` has already
                    //       read the coefficient. Accumulation then returns without
                    //       loading out-of-bounds coefficient.
                    while (col <= range.end) {
                        // Compute z[i] = ∑ c[ijk] * x[j] * y[k] by channel.
                        zi = accumulate_products<Idx,Val,kMode,N>(
                            acc, coef, col, range.end, lhs.data, rhs.data, lhs.shape[1], rhs.shape[1]
                        );
                        out_lane[zi.i * stride_out] = zi.val;
                    }
                }

        } else if constexpr (kMode == Mode::INNER) {
                // NOTE: partial warp sums have to be accumulated in SMEM
                //       as we can't synchronize over blockDim.y without
                //       reaching deadlock within the loop over coefficients.
                while (col <= range.end) {
                    // Sum z[i] over 32 channels at a time
                    zi = accumulate_products<Idx,Val,kMode,N>(
                        acc, coef, col, range.end, lhs.data, rhs.data, lhs.shape[1], rhs.shape[1]
                    );
                    // Horizontally sum N channels within each lane,
                    // then reduce the 32-lane scalar across the warp.
                    Val zi_scalar = sum_warp(hsum<N,Val>(zi.val), 32);
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


template<typename Idx, typename Val, Mode kMode, int N=4, bool strided=false, bool flush=false>
__global__ void kernel(
    const Coef<Idx, Val> *coef,
    CuArray2D<const Val> x,
    CuArray2D<const Val> y,
    CuArray2D<Val> z,
    int num_rows,
    int num_coef,
    dim3 unroll
) {
    using Coef = Coef<Idx,Val>;
    using Buffers = Buffers<Val>;

    // Map thread indices to (x,y,z) channels.
    // Number of parallel channel triplets is k.total
    Channels k = Channels::get<kMode,N>(
        threadIdx.x, x.shape[1], y.shape[1], z.shape[1]
    );

    int num_x = x.shape[0];
    int num_y = y.shape[0];
    int size_x = x.size();
    int size_y = y.size();
    int size_out = z.size();

    int num_strides = 1 + (max(max((int)x.shape[1], (int)y.shape[1]),
                                (int)z.shape[1]) - 1) / (blockDim.x * N);
    int num_warps = blockDim.x / 32;
    constexpr bool accumulate = (kMode == Mode::INNER) && strided;

    // Shared memory layout: [ coef | x | y | z ]
    // smem_base is 16-byte aligned (hardware guarantee ≥128 B).
    extern __shared__ __align__(16) char smem__[];
    char* smem_ = reinterpret_cast<char*>(smem__);
    Buffers smem = Buffers::init(x, y, z, unroll);

    // Optionally store coefficients in shared memory.
    if (BUFFER_COEFS_IN_SMEM) {
        Coef* smem_coef = reinterpret_cast<Coef*>(smem_);
        copy_pipe<1>(smem_coef, coef, num_coef);
        // Overwrite reference to GMEM `coef` with SMEM buffer.
        coef = smem_coef;
        // Align `smem_` to Vect<N, Val> for LDS.64 / LDS.128
        constexpr size_t align = N * sizeof(Val);
        size_t coef_bytes = (size_t)num_coef * sizeof(Coef);
        coef_bytes =  (coef_bytes + align - 1) & ~(align - 1);
        smem_ = (char*)smem_coef + coef_bytes;
        // Wait for coef copy.
        __pipeline_commit();
        wait_pipe();
    }

    // Allocate addresses from start of char* pointer.
    smem.to_shared<1>(smem_);

    // Distribute coefficients across blockDim.y, without z-index overlap
    CoefRange coef_range = find_coef_bounds<Idx, Val>(
        reinterpret_cast<int*>(smem_), coef, num_coef, z.shape[0]
    );

    // First row is blockIdx.x, then stride.
    x.data += blockIdx.x * size_x;
    y.data += blockIdx.x * size_y;
    z.data += blockIdx.x * size_out;

    //=== Grid stride loop ===
    for (unsigned int row = blockIdx.x; row < num_rows; row += gridDim.x) {

        // Write output zeros
        // Useless when coefs cover output indices, e.g. Clebsch-Gordan.
        if constexpr (flush)
            fill(z.data, Val(0), size_out);

        // Zero output SMEM buffer before channel stride loop
        if constexpr (accumulate) {
            fill(smem.out.data, Val(0), num_warps * z.shape[0]);
            __syncthreads();
        }

        // Stride-local GMEM views, advanced per channel slice.
        CuArray2D<const Val> x_s = { x.data, x.shape[0], x.shape[1] };
        CuArray2D<const Val> y_s = { y.data, y.shape[0], y.shape[1] };
        CuArray2D<Val> z_s = { z.data, z.shape[0], z.shape[1] };

        //=== Channel stride loop ===
        for (int s = 0; s < num_strides; s++) {

            // Load x and y into SMEM via pipeline primitives (LDGSTS).
            if (x.shape[1] > unroll.x)
                copy_pipe_strided(smem.lhs.data, x_s.data, num_x, unroll.x, x.shape[1]);
            else
                copy_pipe<N>(smem.lhs.data, x_s.data, smem.lhs.size());

            if (y.shape[1] > unroll.y)
                copy_pipe_strided(smem.rhs.data, y_s.data, num_y, unroll.y, y.shape[1]);
            else if constexpr (N > 1) {
                // Vectorized copy requires the source to be N*sizeof(T)-byte aligned.
                // The per-block y pointer advances by num_y*channels_y floats, so
                // alignment is preserved only when that stride is a multiple of N.
                if (smem.rhs.size() % N == 0)
                    copy_pipe<N>(smem.rhs.data, y_s.data, smem.rhs.size());
                else
                    copy_pipe<1>(smem.rhs.data, y_s.data, smem.rhs.size());
            } else
                copy_pipe<1>(smem.rhs.data, y_s.data, smem.rhs.size());

            __pipeline_commit();
            wait_pipe();

            // Reduce `z[i] = sum(c.val * x[c.j] * y[c.k] for c.i = i)`
            otimes<Idx,Val,kMode,N,accumulate>(
                coef, coef_range, smem.lhs, smem.rhs, z_s, k, smem.out.data
            );
            __syncthreads();

            // Advance to next channel slice (no-op for broadcast operands)
            if (x.shape[1] > unroll.x) x_s.data += unroll.x;
            if (y.shape[1] > unroll.y) y_s.data += unroll.y;
            if (!accumulate && z.shape[1] > unroll.z) z_s.data += unroll.z;

        }//=== Channel stride loop ===

        // Flush output SMEM buffer to GMEM after all strides
        if constexpr (accumulate) {
            for (unsigned int i = threadIdx.x; i < z.shape[0]; i += blockDim.x) {
                Val sum = 0;
                for (int w = 0; w < num_warps; w++)
                    sum += smem.out.data[w * z.shape[0] + i];
                z.data[i] = sum;
            }
            __syncthreads();
        }

        // Update GMEM pointers
        x.data += gridDim.x * size_x;
        y.data += gridDim.x * size_y;
        z.data += gridDim.x * size_out;

    }//=== Grid-stride loop ===
}


}// namespace trailing_channels

}// namespace tensor_product
}// namespace e3j


#undef WARP_MASK

#endif // _E3J_TENSOR_PRODUCT_TRAILING_H_
