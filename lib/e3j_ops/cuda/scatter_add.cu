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

/* Scatter-add kernel on last dimension */

#include "cuda/scatter_add.cuh"
#include "cuda/dispatch_macros.h"
#include <cstdint>
#include <iostream>
#include <type_traits>

// TODO: Compute occupancy depending on device
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK 32 * WARPS_PER_BLOCK

// Full warp mask
#define WARP_MASK 0xffffffff

namespace e3j {
namespace scatter_add_1 {

/* Scatter-add kernel on the last dimension.

Forest-like parallel reduction of values across a warp:

For every stride value `s=1,2,4,8,16` each thread compares
its right binary sibling at distance `s` but merges
if and only if the indices match: we end up with a
forest of "pruned binary trees".

Parameters
----------
* `idx` : 1D-vector of sorted destination indices, of length `p.num_idx`.
* `val` : 2D-array of values to aggregated, shape `(p.num_rows, p.num_idx)`.
* `out` : 2D-array of outputs, shape `(p.num_rows, p.num_out)`.
* `p` : struct describing operand dimensions and dtypes,
  * `num_rows`
  * `num_idx`
  * `num_out`
  * `idx_t` : one of `e3j::Dtype::I{32,64}`.
  * `val_t` : any 32bit element of `e3j::Dtype` for now,
     shared by `val` and `out`.

N.B.
- each warp loads an optimal 128B = 32 * sizeof(float) cache line.
- it takes log2(32) = 5 steps to reduce each warp.
- fast warp shuffling avoids shared memory read/writes and block-wide sync.

Further optimizations:
- pipeline reads to hide GMEM latency
+ write output to a shared memory buffer, to avoid global atomicAdd.
- vectorize reads to 16B = sizeof(float4) per thread => 4 rows per warp.
- iterate vertically on input (column-major thread id) to avoid collisions.
*/
template <typename Idx, typename Val>
__global__ void kernel(
    const Idx *idx,
    const Val *val,
    Val *out,
    Params p
) {
    // Each block works on one row
    int row = blockIdx.x;
    int col = threadIdx.x;
    int lane = threadIdx.x % 32;
    // ...parallel on blockIdx.x-wide strips only

    // Declare shared memory pointer <Val*>
    extern __shared__ __align__(sizeof(Val)) char _smem[];
    Val* smem = reinterpret_cast<Val*>(_smem);

    // Gather from in-bound threads only:
    // - trailing ones hold 0s for safe full-warp shuffling
    // - pipeline one read ahead to hide GMEM latency
    Val val_next = 0;
    Idx idx_next = -1 - lane;
    if (col < p.num_idx and row < p.num_rows) {
        val_next = val[p.num_idx * row + col];
        idx_next = idx[col];
    }

    // Flush shared output buffer
    for (volatile int start = 0; start < p.num_out; start += blockDim.x){
        int idx_out = start + col;
        if (idx_out < p.num_out) {
            smem[idx_out] = Val(0.);
        }
    }
    __syncthreads();

    // Aggregate values to shared buffer
    for (volatile int begin = 0; begin < p.num_idx; begin += blockDim.x){

        int i = begin + col;

        // Pipeline next read
        Val val_i = val_next;
        Idx idx_i = idx_next;
        int next = blockDim.x + i;
        if (next < p.num_idx and row < p.num_rows) {
            val_next = val[p.num_idx * row + next];
            idx_next = idx[next];
        } else {
            val_next = 0;
            idx_next = -1 - lane;   // strictly-negative, per-lane unique sentinel
        }

        // "Forest" reduction of values across warps
        for (unsigned int s=1; s<32; s*=2) {

            // gather index and value from next binary sibling % 32
            Val val_j = __shfl_sync(WARP_MASK, val_i, lane + s);
            Idx idx_j = __shfl_sync(WARP_MASK, idx_i, lane + s);

            // add values only when ancestors match and sibling < 32
            if (idx_j == idx_i and lane + s < 32) {
                val_i += val_j;
            }
        }

        // First occurence of `idx` holds aggregated value
        Idx idx_prev = __shfl_sync(WARP_MASK, idx_i, lane - 1);
        bool is_first = (idx_i != idx_prev or lane == 0);
        if (is_first and i < p.num_idx) {
            atomicAdd(smem + idx_i, val_i);
        }

        // Move on to next strip
        __syncwarp();
    }
    __syncthreads();

    // Write to output
    for (volatile int start = 0; start < p.num_out; start += blockDim.x){
        int idx_out = start + col;
        if (idx_out < p.num_out) {
            out[row * p.num_out + idx_out] = smem[idx_out];
        }
    }
}

template<typename Idx, typename Val>
e3j::Error launch(
    const Idx *idx,
    const Val *val,
    Val *out,
    Params p,
    cudaStream_t stream
) {

    // float16 is not supported yet: the reduction needs atomicAdd(__half*),
    // which does not exist (see NOTE below). The `if constexpr` keeps
    // kernel<Idx,__half> from being instantiated at all.
    if constexpr (std::is_same_v<Val, __half>) {
        return e3j::Error::Unimplemented(
            "scatter_add_1 does not support float16 values yet."
        );
    } else {
        size_t sharedMem = p.num_out * sizeof(Val);

        kernel<<<p.num_rows, THREADS_PER_BLOCK, sharedMem, stream>>>(
            idx, val, out, p
        );

        return e3j::Error::FromCudaLaunch(cudaGetLastError());
    }
}

/* Specialize template for all dtypes using X macro.
 *
 * See https://en.wikipedia.org/wiki/X_macro
 *
 * This is necessary for object files to actually contain compiled functions.
 * Ideally, the DISPATCH_DTYPES list could be moved to a main header file to
 * be reused across different .cu files.
 *
 * NOTE: Specializing `launch<...>` will force `kernel<...>` to be compiled too.
 * We could otherwise make the namespace `scatter_add_1` a templated class.
 *
 * NOTE: `float` and `double` are both instantiated (`atomicAdd(double*)` needs
 * sm_60+). `__half` is not, since `atomicAdd(__half*, __half)` is not
 * overloaded: `launch()` drops that instantiation and returns Unimplemented.
 * We could still implement our own (fast) version with `atomicCAS`, search
 * e.g. for `fastAtomicAdd` on the torch repo.
 */

#define FOR_EACH_DTYPE_PAIR(Idx, Val)       \
template e3j::Error launch<Idx, Val> (     \
    const Idx *idx,                         \
    const Val *val,                         \
    Val *out,                               \
    Params p,                               \
    cudaStream_t stream                     \
);
__FOR_EACH_DTYPE_PAIR
#undef FOR_EACH_DTYPE_PAIR

} // namespace scatter_add_1
} // namespace e3j

#undef WARP_MASK
#undef WARPS_PER_BLOCK
#undef THREADS_PER_BLOCK
