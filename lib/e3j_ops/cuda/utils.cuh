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

#ifndef _E3J_CUDA_UTILS_H_
#define _E3J_CUDA_UTILS_H_

#include <cuda.h>
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime_api.h>

namespace e3j {

// ── Vect<N, T> type system ────────────────────────────────────────────────
//
// N is the vectorization width: number of consecutive elements grouped into
// a single variable (held in N registers). N ∈ {1, 2, 4} maps to scalar /
// float2 / float4, emitting LDS.32 / LDS.64 / LDS.128 on shared memory loads.

template<int N, typename T> struct VectType;
template<typename T> struct VectType<1, T>          { using type = T;      };
template<> struct VectType<2, float>                { using type = float2; };
template<> struct VectType<2, std::int32_t>         { using type = int2;   };
template<> struct VectType<4, float>                { using type = float4; };
template<> struct VectType<4, std::int32_t>         { using type = int4;   };
template<typename T>
struct VectType<2, T> {
    struct T2 {T x; T y;};
    using type = T2;
};
template<typename T>
struct VectType<4, T> {
    struct T4 {T x; T y; T z; T w;};
    using type = T4;
};
template<int N, typename Val> using Vect = typename VectType<N, Val>::type;

template<int N, typename Val>
__device__ Vect<N, Val> broadcast(Val x) {
    if constexpr (N == 1) {
        return x;
    } else if constexpr (N == 2) {
        return {x, x};
    } else {
        return {x, x, x, x};
    }
}

template<int N, typename Val>
__device__ Vect<N, Val> mul(Vect<N, Val> a, Vect<N, Val> b) {
    if constexpr (N == 1) {
        return a * b;
    } else if constexpr (N == 2) {
        return {a.x*b.x, a.y*b.y};
    } else {
        return {a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w};
    }
}

// Fused multiply-add: acc[i] += a[i] * b[i].
// Uses __fmaf_rn to emit FFMA instead of FMUL+FADD, eliminating the
// intermediate prod[] register and halving FP instruction count on the
// hot accumulation path (N=4: saves 4 FADD per coefficient iteration).
template<int N, typename Val>
__device__ void fmadd(Vect<N, Val> &acc, Vect<N, Val> a, Vect<N, Val> b) {
    if constexpr (N == 1) {
        acc = __fmaf_rn(a, b, acc);
    } else if constexpr (N == 2) {
        acc.x = __fmaf_rn(a.x, b.x, acc.x);
        acc.y = __fmaf_rn(a.y, b.y, acc.y);
    } else {
        acc.x = __fmaf_rn(a.x, b.x, acc.x);
        acc.y = __fmaf_rn(a.y, b.y, acc.y);
        acc.z = __fmaf_rn(a.z, b.z, acc.z);
        acc.w = __fmaf_rn(a.w, b.w, acc.w);
    }
}

// Fused multiply-add: acc[i] += a * b[i]  (scalar a broadcast).
// Disabled for N=1 where Vect<1,Val> = Val would conflict with the above.
template<int N, typename Val, std::enable_if_t<(N > 1), int> = 0>
__device__ void fmadd(Vect<N, Val> &acc, Val a, Vect<N, Val> b) {
    if constexpr (N == 2) {
        acc.x = __fmaf_rn(a, b.x, acc.x);
        acc.y = __fmaf_rn(a, b.y, acc.y);
    } else {
        acc.x = __fmaf_rn(a, b.x, acc.x);
        acc.y = __fmaf_rn(a, b.y, acc.y);
        acc.z = __fmaf_rn(a, b.z, acc.z);
        acc.w = __fmaf_rn(a, b.w, acc.w);
    }
}

// p must be N*sizeof(Val)-byte aligned (guaranteed by SMEM layout padding).
template<int N, typename Val>
__device__ Vect<N, Val> load(const Val *p) {
    if constexpr (N == 1) {
        return *p;
    } else {
        return *reinterpret_cast<const Vect<N, Val>*>(p);
    }
}

template<int N, typename Val>
__device__ Val hsum(Vect<N, Val> v) {
    if constexpr (N == 1) {
        return v;
    } else if constexpr (N == 2) {
        return v.x + v.y;
    } else {
        return v.x + v.y + v.z + v.w;
    }
}

namespace utils {

// Simple synchronous copy.
//
// Copy with a 2D block by passing thread IDs and block dim explicitly.
// Boiler plate kept here to avoid typos and allow for easier changes
// in copy strategies and SMEM layouts.
template <typename T>
__device__ void copy(
    T* dst, const T* src, const int numel, const int tid, const int dim
) {
    for (int col = tid; col < numel; col += dim) {
        dst[col] = src[col];
    }
}

// Copy with a 1D block along blockDim.x
template <typename T>
__device__ void copy(T* dst, const T* src, const int numel) {
    copy(dst, src, numel, threadIdx.x, blockDim.x);
}

// Drain pipeline up to S commits and synchronize threads.
// Use with __pipeline_commit().
template <int S=0>
__device__ void wait_pipe() {
    __pipeline_wait_prior(S);
    __syncthreads();
}

// Contiguous LDGSTS copy via pipeline primitives (cp.async).
//
// Copies `numel` elements from src to dst. Each thread copies
// N-element chunks (N*sizeof(T) bytes per cp.async instruction),
// with element-wise fallback for the tail when numel % N != 0.
//
// N > 1 requires src and dst to be N*sizeof(T)-byte aligned.
// Use N=1 when alignment is not guaranteed.
//
// Caller must __pipeline_commit() + __pipeline_wait_prior(K)
// + __syncthreads() before reading dst.
template <int N=1, typename T>
__device__ void copy_pipe(T* dst, const T* src, const int numel) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int dim = blockDim.x * blockDim.y;
    // cp.async supports 4, 8, or 16 byte copies.
    constexpr size_t cp_size = sizeof(T) * N;
    if constexpr (cp_size > 16) {
        static_assert(cp_size % 16 == 0);
        constexpr int K = cp_size / 16;
        int4 *dst4 = reinterpret_cast<int4*>(dst);
        const int4 *src4 = reinterpret_cast<const int4*>(src);
        #pragma unroll 1
        for (int i = tid * K; i < numel * K; i += dim * K) {
            #pragma unroll
            for (int k = 0; k < K; k++)
                __pipeline_memcpy_async(&dst4[i + k], &src4[i + k], 16);
        }
    } else if constexpr (N > 1) {
        int aligned = (numel / N) * N;
        #pragma unroll 1
        for (int i = tid * N; i < aligned; i += dim * N) {
            __pipeline_memcpy_async(&dst[i], &src[i], sizeof(T) * N);
        }
        for (int i = aligned + tid; i < numel; i += dim) {
            __pipeline_memcpy_async(&dst[i], &src[i], sizeof(T));
        }
    } else {
        #pragma unroll 1
        for (int i = tid; i < numel; i += dim) {
            __pipeline_memcpy_async(&dst[i], &src[i], sizeof(T));
        }
    }
}


// Strided LDGSTS copies via pipeline primitives (cp.async).
//
// The cooperative_groups API (cg::memcpy_async, cg::tiled_partition)
// adds ~40 registers of overhead from non-inlined barrier and block
// handle state. The pipeline primitives emit cp.async directly as
// per-thread inline PTX, avoiding the cg register bloat.
//
// Copies `width` elements per row from strided source to packed
// destination, issuing one 16-byte LDGSTS per thread per iteration.
// Caller must __pipeline_commit() + __pipeline_wait_prior(0) + __syncthreads().
template <typename T>
__device__ void copy_pipe_strided (
    T* dst,
    const T* src,
    unsigned int num_rows,
    unsigned int width,
    unsigned int stride
) {
    constexpr int CHUNK = 16 / sizeof(T);
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int dim = blockDim.x * blockDim.y;
    int total = num_rows * width;
    #pragma unroll 1
    for (int i = tid * CHUNK; i < total; i += dim * CHUNK) {
        int r = i / width;
        int c = i % width;
        int remaining = min(CHUNK, width - c);
        if (remaining == CHUNK) {
            __pipeline_memcpy_async(
                &dst[r * width + c],
                &src[r * stride + c],
                sizeof(T) * CHUNK
            );
        } else {
            // Tail elements: scalar fallback
            #pragma unroll 1
            for (int k = 0; k < remaining; k++)
                dst[r * width + c + k] = src[r * stride + c + k];
        }
    }
}


// Strided chunk copies.
//
// Useful to skip independent channels (0-31, 32-63, ...) in case of large
// feature dimensions. Processing all channels within a single block may
// otherwise require too much shared memory.
//
// With 32 channel blocks, still takes advantage of 128 B coalesced loads.
template <typename T>
__device__ void copy_strided(
    T* dst,
    const T* src,
    const int num_rows,
    const int stride,
    const int width
) {
    for (int r = threadIdx.y; r < num_rows; r += blockDim.y) {
        copy(&dst[r * stride], &src[r * width], width);
    }
}

// Scalar fill with a 1D block along blockDim.x.
template <typename T>
__device__ void fill(T* dst, const T value, const int numel) {
    for (int col = threadIdx.x; col < numel; col += blockDim.x) {
        dst[col] = value;
    }
}

// Vectorized fill (N-wide stores, full 2D block).
template <int N, typename T>
__device__ void fill(T* dst, const T value, const int numel) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int dim = blockDim.x * blockDim.y;
    if constexpr (N > 1) {
        Vect<N, T> v = broadcast<N, T>(value);
        Vect<N, T>* out = reinterpret_cast<Vect<N, T>*>(dst);
        int aligned = numel / N;
        for (int i = tid; i < aligned; i += dim)
            out[i] = v;
        for (int i = aligned * N + tid; i < numel; i += dim)
            dst[i] = value;
    } else {
        for (int i = tid; i < numel; i += dim)
            dst[i] = value;
    }
}

// Helper to query useful device properties
struct DeviceProperties {

    int device = 0;
    int smem_max = 0;
    int smem_opt_in = 0;

    static DeviceProperties query() {
        DeviceProperties out;
        cudaGetDevice(&out.device);
        cudaDeviceGetAttribute(
            &out.smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, out.device
        );
        cudaDeviceGetAttribute(
            &out.smem_opt_in, cudaDevAttrMaxSharedMemoryPerBlock, out.device
        );
        return out;
    }
};

} // namespace utils
} // namespace e3j

#endif //_E3J_CUDA_UTILS_H_
