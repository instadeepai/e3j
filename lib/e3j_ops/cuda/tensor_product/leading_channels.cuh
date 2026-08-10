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

#ifndef _E3J_TENSOR_PRODUCT_LEADING_H_
#define _E3J_TENSOR_PRODUCT_LEADING_H_

#include "cuda/tensor_product.cuh"
#include "cuda/tensor_product/details.cuh"
#include "cuda/utils.cuh"

#define WARP_MASK 0xffffffff

#define THREADS_PER_BLOCK 128
#define THREADS_PER_SM 2048

#define KB * 1024
#define SIZE_L1 256 KB
#define MAX_SMEM 112 KB

namespace e3j {
namespace tensor_product {

namespace leading_channels {

    using utils::copy;
    using utils::fill;

    struct Channels {
        int x;
        int y;
        int z;
    };

/********************************************************************
 *  Scatter-reduce (idx, val) with warp shuffles.
 *
 *  Propagate values from right to left in a tree-like fashion, while
 *  comparison of indices dictates whether to accumulate.
 *
 *  When the `width` of warp shuffles is a strict divisor of 32,
 *  the quotient `N = 32 / width` is the number of isomorphic
 *  rows that should be reduced in parallel. The depth of the
 *  reduction loop is `xlog2(width) <= 5`: reducing the width
 *  helps prevent the congestion of warp shuffles (MIO throttle).
 *
 *  @param idx_0 thread-wise destination index.
 *  @param val_0 current value this thread accumulates.
 *
 *  @return boolean identifying left-most threads holding final values.
 *
********************************************************************/
template<typename Idx, typename Val, int width>
__device__ bool scatter_add_warp(Idx idx_0, Val& val_0) {

    unsigned int lane = threadIdx.x % width;
    bool has_sibling = true;

    Idx idx_s = -1;
    Val val_s = 0;
    // "Forest" reduction across warps
    #pragma unroll 1
    for (unsigned int s = 1; s < width; s <<= 1) {

        // gather index and value from next binary sibling % 32
        idx_s = __shfl_down_sync(WARP_MASK, idx_0, s, width);
        val_s = __shfl_down_sync(WARP_MASK, val_0, s, width);

        // add values only when ancestors match and sibling < 32
        has_sibling = lane + s < width;
        has_sibling &= idx_s == idx_0;

        if (has_sibling) {
            val_0 += val_s;
        }
    }
    Idx idx_left = __shfl_up_sync(WARP_MASK, idx_0, 1, width);
    bool write_out = (lane == 0) || (idx_0 != idx_left);
    return write_out;
}

/******************************************************************
 *  Contract two (x, y) channels, loading from / storing to SMEM.
 *
 *  The way x and y channels are looped over and paired depends on
 *  the tensor product mode (outer | inner), and so does the global
 *  store pattern. Having the core tensor product operation on one
 *  channel separately available allows reuse across different loop
 *  patterns, although this function does little more than loading
 *  coefficients and delegating to `scatter_add_warp()`.
 *****************************************************************/
template<int width, typename Idx, typename Val>
__device__ void tensor_product_one_channel(
    const Coef<Idx,Val>* coef,
    SMEM<Val> smem,
    Channels k,
    Params p
) {
    using Coef = Coef<Idx,Val>;
    constexpr int n_groups = 32 / width;
    unsigned int lane = threadIdx.x % width;
    unsigned int warp = threadIdx.x / 32;

    Val* smem_z_k = smem.z + (k.z * p.num_out);
    // Scatter-reduction of `num_idx` coefficients x values
    //
    // Each warp processes `width` distinct coefficients,
    // and `n_groups` isomorphic channels at a time.
    //
    // NOTE: it is important to keep warps converged for shuffles and
    //       synchronization points, so we share the loop variable
    //       across threads in a warp.
    for (int begin = 0; begin < p.num_idx; begin += blockDim.x / n_groups) {

        int col = begin + lane + (warp * width);
        bool in_bounds = col < p.num_idx;


        // Initialize for out of bounds columns
        Coef cijk = {.val = 0, .i = -1, .j = -1, .k = -1};

        // Compute zi summand cijk * xj * yk
        if (in_bounds) {
            cijk = coef[col];
            // Gather input values from shared memory
            Val x_j = smem.x[cijk.j + k.x * p.num_x];
            Val y_k = smem.y[cijk.k + k.y * p.num_y];
            // Multiply into the coefficient
            cijk.val *= x_j;
            cijk.val *= y_k;
        }

        // Scatter-reduction of zi across warps
        bool write_out = scatter_add_warp<Idx, Val, width>(cijk.i, cijk.val);

        // Only first z-index occurrence writes out
        if (write_out and in_bounds) {
            // Note: handle races with l.h.s. warp. Striding through
            //       coefficient blocks should prevent collisions.
            atomicAdd(smem_z_k + cijk.i, cijk.val);
        }

    }//--- coef loop
}


/********************************************************************
 *  Sparse tensor product kernel with channels leading lm coordinates.
 *
 *  The coefficients and inputs are scatter-reduced with warp shuffles
 *  to aggregate products `cijk * xj * yk` on the output coordinate `i`:
 *
 *      z[i] = ∑ c[i,j,k] * x[j] * y[k]
 *
 *  When the `width` template parameter divides 32, the quotient is the
 *  number `n_groups` of rows processed in parallel by a single warp,
 *  reducing the depth of scatter-reduction to `log2(width)`.
 *
 ********************************************************************/
template<typename Idx, typename Val, Mode kMode, int width=32>
__global__ void kernel(
    const Coef<Idx,Val> *coef,
    const Val *x,
    const Val *y,
    Val *out,
    Params p
) {
    unsigned int lane = threadIdx.x % width;
    unsigned int warp = threadIdx.x / 32;
    int group = (threadIdx.x % 32) / width;

    using Coef = Coef<Idx, Val>;
    using SMEM = SMEM<Val>;

    // Shared memory layout: [ x | y | out ]
    extern __shared__ char smem_[];
    SMEM smem = SMEM::from_extern(smem_, p);

    //=== Grid stride loop ===

    for (int row = blockIdx.x; row < p.num_rows; row += gridDim.x) {

    // OUTER: for u { for v { store } }
    if constexpr (kMode == Mode::OUTER) {

        // Loop over LHS channels with stride k_x
        for (int c_x = 0; c_x < p.channels_x; c_x += p.unroll_x) {

            // Load `unroll_x` rows of x [and interleave] channels in SMEM
            int row_x = (row * p.channels_x) + c_x;
            copy(smem.x, x + (row_x * p.num_x), p.unroll_x * p.num_x);

        // Loop over RHS channels with stride k_y (outer product)
        for (int c_y = 0; c_y < p.channels_y; c_y += p.unroll_y) {

            // FIXME: k.x hardcoded to 0, requires unroll_x = 0 too.
            Channels k = {.x = 0, .y = group, .z = group};

            // Load `unroll_y` rows of y into shared memory
            int row_y = (row * p.channels_y) + c_y;
            copy(smem.y, y + (row_y * p.num_y), p.unroll_y * p.num_y);

            // Clear shared output buffer
            fill(smem.z, Val(0), p.unroll_y * p.num_out);

            // Input buffers are loaded, fill output buffer unroll_z rows at a time.
            __syncthreads();

            tensor_product_one_channel<width>(coef, smem, k, p);

            __syncthreads();

            // Write shared results to global output
            // TODO: support channels not a multiple of unroll
            int size_out = p.unroll_z * p.num_out;
            long long row_z = (long long)row * (p.channels_x * p.channels_y)
                        + c_x * p.channels_y + c_y;

            copy(out + row_z * p.num_out,  smem.z, size_out);
            fill(smem.z, Val(0), size_out);

        }//--- c_y
        }//--- c_x
        break;
    }

    // INNER: for v {} store
    else if constexpr (kMode == Mode::INNER) {

        // Clear shared output buffer
        fill(smem.z, Val(0), p.num_out);
        __syncthreads();

        // Loop over LHS = RHS channels with stride k_x
        for (int c_xy = 0; c_xy < p.channels_y; c_xy += 1) {

            // Load `unroll_x` rows of x [and interleave] channels in SMEM
            int row_x = (row * p.channels_x) + c_xy;
            copy(smem.x, x + (row_x * p.num_x), p.num_x);

            // Load `unroll_y` rows of y into shared memory
            int row_y = (row * p.channels_y) + c_xy;
            copy(smem.y, y + (row_y * p.num_y), p.num_y);

            // NOTE: can process only 1 channel at a time for now,
            //       since they need to be accumulated.
            Channels k = {.x = 0, .y = 0, .z = 0};

            // Input buffers are loaded, accumlate in output buffer.
            __syncthreads();

            tensor_product_one_channel<width>(coef, smem, k, p);

            __syncthreads();

        }//--- c_xy

        copy(out + (long long)row * p.num_out, smem.z, p.num_out);
        __syncthreads();
        break;
    }

    }//=== Grid stride loop ===
}

namespace {

template<typename Idx, typename Val>
struct LaunchConfig {

    int gridDim;
    int blockDim;
    size_t sizeSMEM;
    e3j::Error error;

    static LaunchConfig get_hints(Params p, int debug) {
        // SMEM : [ kx * num_x | ky * num_y | kz * num_z ]
        // L1   : all coefficients will fit if sizeCoef << 256 KB
        //
        // Constraints:
        //   - SMEM < 48 KB
        //   - SMEM * numBlocks + sizeCoef < 256 KB (for L1 persistence)
        // Hints:
        //   - aim for numBlocks * sizeLoads >= 32 KB / SM in-flight on H100
        //   - keep SMEM < 256 KB / SM and load at most 64 KB / SM
        size_t sizeLoad = sizeof(Val) * (p.unroll_x * p.num_x + p.unroll_y * p.num_y);
        size_t sizeSMEM = sizeLoad + sizeof(Val) * p.unroll_z * p.num_out;
        size_t sizeCoef = p.num_idx * (3 * sizeof(Idx) + sizeof(Val));

        // int numSMs = 132;
        int threadsPerBlock = THREADS_PER_BLOCK;
        int numBlocks = THREADS_PER_SM / threadsPerBlock;

        if (sizeSMEM > 48 KB) {
            std::string msg =
                "e3j_ops.tensor_product: SMEM overflow ("
                + std::to_string(sizeSMEM) + " B > 48 KB) "
                "unsupported with Layout::LEADING_CHANNELS.";
            return LaunchConfig { .error = e3j::Error::InvalidArgument(msg) };
        }

        if (debug > 1) {
            printf("> e3j::tensor_product::kernel<<<%d*SMs,%d>>>\n",
                    numBlocks, threadsPerBlock);
            printf("\tsizeLoad: %zu B/SM\n", numBlocks * sizeLoad);
            printf("\tsizeSMEM: %zu B\n", sizeSMEM);
            printf("\tsizeCoef: %zu B\n", sizeCoef);
            printf("\tunroll: (%d,%d,%d)\n", p.unroll_x, p.unroll_y, p.unroll_z);
            printf("\tchannels: (%d,%d)\n", p.channels_x, p.channels_y);
        }

        return LaunchConfig {
            .gridDim = p.num_rows, // numSMs * numBlocks
            .blockDim = threadsPerBlock,
            .sizeSMEM = sizeSMEM
        };
    }
};

}// namespace

template<typename Idx, typename Val, Mode kMode = Mode::OUTER>
e3j::Error launch(
    const Coef<Idx,Val> *coef,
    const Val *x,
    const Val *y,
    Val *out,
    Params p,
    cudaStream_t stream,
    int debug
 ) {

    if (p.mode == Mode::INNER and p.channels_x != p.channels_y) {
        return e3j::Error::InvalidArgument("Channels of 'INNER' tensor product operands must match.");
    }

    LaunchConfig<Idx,Val> cfg = LaunchConfig<Idx,Val>::get_hints(p, debug);
    if (cfg.error.failure()) return cfg.error;

    if (kMode == Mode::MAP) {
        return e3j::Error::Unimplemented("Mode::MAP not yet supported with Layout::LEADING_CHANNELS.");
    };

    // float16 is not supported yet with this layout: the kernel accumulates in
    // shared memory with atomicAdd(__half*), which does not exist. The
    // `if constexpr` keeps kernel<...,__half> from being instantiated at all.
    if constexpr (std::is_same_v<Val, __half>) {
        return e3j::Error::Unimplemented(
            "LEADING_CHANNELS tensor product does not support float16 yet."
        );
    } else {
        switch (p.unroll_z) {
            case 1:
                kernel<Idx, Val, kMode, 32>
                    <<<cfg.gridDim, cfg.blockDim, cfg.sizeSMEM, stream>>>
                    (coef, x, y, out, p);
                break;

            case 2:
                kernel<Idx, Val, kMode, 16>
                    <<<cfg.gridDim, cfg.blockDim, cfg.sizeSMEM, stream>>>
                    (coef, x, y, out, p);
                break;

            case 4:
                kernel<Idx, Val, kMode, 8>
                    <<<cfg.gridDim, cfg.blockDim, cfg.sizeSMEM, stream>>>
                    (coef, x, y, out, p);
                break;

            case 8:
                kernel<Idx, Val, kMode, 4>
                    <<<cfg.gridDim, cfg.blockDim, cfg.sizeSMEM, stream>>>
                    (coef, x, y, out, p);
                break;

            case 16:
                kernel<Idx, Val, kMode, 2>
                    <<<cfg.gridDim, cfg.blockDim, cfg.sizeSMEM, stream>>>
                    (coef, x, y, out, p);
                break;

            default:
                return e3j::Error::InvalidArgument("unroll parameter must be in {1,2,4,8,16}.");
        }
    }

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        return e3j::Error::FromCudaLaunch(launch_err);
    }

    if (debug > 0) {
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            return e3j::Error::FromCudaRuntime(error);
        }
    }

    return e3j::Error::Success();
}


}// namespace leading_channels

}// namespace tensor_product
}// namespace e3j

#undef MAX_SMEM
#undef KB
#undef SIZE_L1
#undef DEBUG

#undef THREADS_PER_SM
#undef THREADS_PER_BLOCK
#undef WARP_MASK

#endif // _E3J_TENSOR_PRODUCT_LEADING_H_
