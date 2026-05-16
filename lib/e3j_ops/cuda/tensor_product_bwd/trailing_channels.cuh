#ifndef _E3J_TENSOR_PRODUCT_BWD_TRAILING_H_
#define _E3J_TENSOR_PRODUCT_BWD_TRAILING_H_

#include <cuda.h>

#include "cuda/tensor_product_bwd.cuh"
#include "cuda/tensor_product.cuh"
#include "cuda/tensor_product/details.cuh"
#include "cuda/tensor_product/debug.cuh"
#include "cuda/tensor_product/trailing_channels.cuh"
#include "cuda/dispatch_macros.h"
#include "cuda/array.cuh"
#include "cuda/utils.cuh"
#include "ffi/error.h"

#define BUFFER_BWD_COEFS_IN_SMEM true

namespace e3j {
namespace tensor_product {

namespace trailing_channels {

using utils::copy;
using utils::wait_pipe;
using utils::copy_pipe;
using utils::copy_pipe_strided;
using utils::fill;
using utils::DeviceProperties;

template <typename T>
struct BuffersBwd {
    CuArray2D<T> dz;
    CuArray2D<T> x;
    CuArray2D<T> y;
    T* dxy;


    static __device__ BuffersBwd init (
        CuArray2D<const T> x, CuArray2D<const T> y, CuArray2D<const T> dz, dim3 unroll
    ) {
        return BuffersBwd {
            .dz = { nullptr, dz.shape[0], unroll.z },
            .x = { nullptr, x.shape[0], unroll.x },
            .y = { nullptr, y.shape[0], unroll.y },
            .dxy = nullptr,
        };
    }

    template<int Stages=1>
    __device__ void to_shared(char* smem_) {
        constexpr int A = 16 / sizeof(T);
        dz.data = reinterpret_cast<T*>(smem_);
        x.data = dz.data + (Stages * dz.size() + A-1) / A * A;
        y.data = x.data + (Stages * x.size() + A-1) / A * A;
        dxy = y.data + (y.size() + A-1) / A * A;
    }

};


template<typename Idx, typename Val, Mode dxMode, Mode dyMode, int N=1, bool Fused=true>
__global__ void kernel_bwd(
    const Coef<Idx,Val> *coef,
    CuArray2D<const Val> x,
    CuArray2D<const Val> y,
    CuArray2D<const Val> dz,
    CuArray2D<Val> dx,
    CuArray2D<Val> dy,
    int num_rows,
    int num_coef,
    dim3 unroll
) {

    using Coef = Coef<Idx,Val>;

    // Map threads to (x,y,z) channels.
    // Number of parallel channel triplets is k.total
    Channels k_dx = Channels::get<dxMode, N>(
        threadIdx.x, dz.shape[1], y.shape[1], dx.shape[1]
    );
    Channels k_dy = Channels::get<dyMode, N>(
        threadIdx.x, dz.shape[1], x.shape[1], dy.shape[1]
    );

    unsigned int size_x = x.size(),
                 size_y = y.size(),
                 size_z = dz.size();

    int num_warps = blockDim.x / 32;
    constexpr bool inner_dx = (dxMode == Mode::INNER);
    constexpr bool inner_dy = (dyMode == Mode::INNER);

    // Shared memory layout: [ dz | x | y | scratch ]
    extern __shared__ __align__(16) char smem__[];
    char* smem_ = reinterpret_cast<char*>(smem__);

    if constexpr (BUFFER_BWD_COEFS_IN_SMEM) {
        // Copy coefficients at beginning of dynamic shared memory.
        Coef* smem_coef = reinterpret_cast<Coef*>(smem_);
        copy_pipe<1>(smem_coef, coef, 2 * num_coef);
        // Overwrite reference to GMEM `coef` with SMEM buffer.
        coef = smem_coef;
        // Advance past the coef buffer, rounding up to 16 B so smem_x/y
        // are 16-byte aligned for LDS.128 (N=4). This adds at most 8 B
        // of padding when sizeof(Coef)==8 and num_idx is odd.
        constexpr size_t align = N * sizeof(Val);
        size_t coef_bytes = (size_t)num_coef * sizeof(Coef) * 2;
        smem_ = (char*)smem_coef + ((coef_bytes + align - 1) & ~(align - 1));
        // Wait for coef copy.
        __pipeline_commit();
        wait_pipe();
    }

    using BuffersBwd = BuffersBwd<Val>;

    // Build SMEM Buffers for each pass, sharing underlying memory.
    // dx pass: lhs=dz, rhs=y, out=scratch
    // dy pass: lhs=dz, rhs=x, out=scratch
    BuffersBwd smem = BuffersBwd::init(x, y, dz, unroll);
    smem.to_shared(smem_);

    const Coef *coef_dx = coef;
    const Coef *coef_dy = coef + num_coef;

    // Distribute coefficients across blockDim.y, without z-index overlap.
    // Reuses dz buffer (not yet populated) as int[blockDim.y + 1] scratch.
    int *smem_cuts = reinterpret_cast<int*>(smem.dz.data);
    CoefRange coef_range_dx = find_coef_bounds<Idx, Val>(smem_cuts, coef_dx, num_coef, dx.shape[0]);
    CoefRange coef_range_dy = find_coef_bounds<Idx, Val>(smem_cuts, coef_dy, num_coef, dy.shape[0]);

    // Block-wise row offsets
    x.data += blockIdx.x * size_x;
    y.data += blockIdx.x * size_y;
    dz.data += blockIdx.x * size_z;
    dx.data += blockIdx.x * size_x;
    dy.data += blockIdx.x * size_y;

    //=== Grid stride loop ===
    for (int row = blockIdx.x; row < num_rows; row += gridDim.x) {

        unsigned int cx = x.shape[1],
                     cy = y.shape[1],
                     cz = dz.shape[1];

        int num_strides = 1 + (max(max(cx, cy), cz) - 1) / (blockDim.x * N);

        // Zero-out the scratch SMEM buffer if INNER and channel striding,
        if constexpr (inner_dx || inner_dy) {
            int scratch_n = inner_dx ? dx.shape[0] : dy.shape[0];
            fill(smem.dxy, Val(0), num_warps * scratch_n);
            __syncthreads();
        }

        // Stride-local GMEM views, advanced per channel slice.
        CuArray2D<const Val> dz_s = { dz.data, dz.shape[0], cz };
        CuArray2D<const Val> x_s  = { x.data,  x.shape[0],  cx  };
        CuArray2D<const Val> y_s  = { y.data,  y.shape[0],  cy  };
        CuArray2D<Val> dx_s = { dx.data, dx.shape[0], cx };
        CuArray2D<Val> dy_s = { dy.data, dy.shape[0], cy };

        //=== Channel stride loop ===
        for (int s = 0; s < num_strides; s++) {

            // Load dz and y — needed by the dx pass.
            // Broadcast operands (n_channels <= unroll) are contiguous:
            // use copy_pipe<1> to avoid 16 B alignment requirement.
            if (cz > unroll.z)
                copy_pipe_strided(smem.dz.data, dz_s.data, dz.shape[0], unroll.z, cz);
            else
                copy_pipe<N>(smem.dz.data, dz_s.data, smem.dz.size());

            if (cy > unroll.y)
                copy_pipe_strided(smem.y.data, y_s.data, y.shape[0], unroll.y, cy);
            else
                copy_pipe<1>(smem.y.data, y_s.data, smem.y.size());
            __pipeline_commit();

            // Issue x load — overlaps with the dx otimes below.
            if (cx > unroll.x)
                copy_pipe_strided(smem.x.data, x_s.data, x.shape[0], unroll.x, cx);
            else
                copy_pipe<N>(smem.x.data, x_s.data, smem.x.size());
            __pipeline_commit();

            // Drain dz + y (pipeline group 0).
            wait_pipe<1>();

            // Compute dx: reads dz + y (complete), does not touch smem.x.
            otimes<Idx,Val,dxMode, N, inner_dx>(
                coef_dx, coef_range_dx, smem.dz, smem.y, dx_s, k_dx, smem.dxy
            );

            // Drain x (pipeline group 1).
            wait_pipe();

            // Compute dy: reads dz + x (now complete).
            otimes<Idx,Val,dyMode, N, inner_dy>(
                coef_dy, coef_range_dy, smem.dz, smem.x, dy_s, k_dy, smem.dxy
            );

            __syncthreads();

            // Advance to next channel slice (no-op for broadcast operands)
            if (cz > unroll.z) dz_s.data += unroll.z;
            if (cx > unroll.x) x_s.data  += unroll.x;
            if (cy > unroll.y) y_s.data  += unroll.y;
            if (!inner_dx && cx > unroll.x) dx_s.data += unroll.x;
            if (!inner_dy && cy > unroll.y) dy_s.data += unroll.y;

        }//=== Channel stride loop ===

        // Flush INNER scratch to GMEM after all strides
        if constexpr (inner_dx) {
            for (unsigned int i = threadIdx.x; i < dx.shape[0]; i += blockDim.x) {
                Val sum = 0;
                for (int w = 0; w < num_warps; w++)
                    sum += smem.dxy[w * dx.shape[0] + i];
                dx.data[i] = sum;
            }
            __syncthreads();
        }
        if constexpr (inner_dy) {
            for (unsigned int i = threadIdx.x; i < dy.shape[0]; i += blockDim.x) {
                Val sum = 0;
                for (int w = 0; w < num_warps; w++)
                    sum += smem.dxy[w * dy.shape[0] + i];
                dy.data[i] = sum;
            }
            __syncthreads();
        }

        // Update GMEM pointers
        dz.data += gridDim.x * size_z;
        x.data  += gridDim.x * size_x;
        y.data  += gridDim.x * size_y;
        dx.data += gridDim.x * size_x;
        dy.data += gridDim.x * size_y;

    }//=== Grid stride loop ===

}

} // namespace trailing_channels
} // namespace tensor_product
} // namespace e3j

#endif // _E3J_TENSOR_PRODUCT_BWD_TRAILING_H_
