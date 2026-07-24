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

#ifndef _E3J_TENSOR_PRODUCT_BWD_LAUNCH_TRAILING_H_
#define _E3J_TENSOR_PRODUCT_BWD_LAUNCH_TRAILING_H_

#include "cuda/tensor_product/launch_trailing.cuh"
#include "cuda/tensor_product_bwd/trailing_channels.cuh"

namespace e3j {
namespace tensor_product {
namespace trailing_channels {


template<typename Idx, typename Val, Mode kxMode, Mode kyMode>
struct LaunchConfigBwd {

    dim3 gridDim;
    dim3 blockDim;
    size_t sizeSMEM;
    e3j::Error error = e3j::Error::Success();
    int N;

    static Sizes get_sizes(Params p, dim3 blockDims, int N, bool fused=true) {

        constexpr int A = 16 / sizeof(Val);

        int channels_z = p.channels_out();
        // Pad allocations to N-element boundary to match to_shared() alignment.
        int size_x_raw = p.num_x * min((int)blockDims.x * N, p.channels_x),
            size_y_raw = p.num_y * min((int)blockDims.x * N, p.channels_y),
            size_z_raw = p.num_out * min((int)blockDims.x * N, channels_z);
        int size_x = (size_x_raw + (A-1)) & ~(A-1),
            size_y = (size_y_raw + (A-1)) & ~(A-1),
            size_z = (size_z_raw + (A-1)) & ~(A-1);

        size_t sizeCoef = sizeof(Coef<Idx,Val>) * 2 * p.num_idx;
        size_t align = SMEM_BUFFER_ALIGN;  // 16 B, matches kernel pointer logic
        sizeCoef = (sizeCoef + align - 1) & ~(align - 1);
        size_t sizeLoad = fused
            ? sizeof(Val) * (size_z + size_x + size_y)
            : sizeof(Val) * (size_z + max(size_x, size_y));
        size_t sizeSMEM = sizeLoad;
        // Allocate scratch SMEM space for INNER backward passes.
        if (p.mode == Mode::OUTER and p.channels_x == 1) {
            sizeSMEM += sizeof(Val) * p.num_x * (1 + (p.channels_y - 1) / (32 * N));
        } else if (p.mode == Mode::OUTER and p.channels_y == 1) {
            sizeSMEM += sizeof(Val) * p.num_y * (1 + (p.channels_x - 1) / (32 * N));
        }
        if (BUFFER_BWD_COEFS_IN_SMEM)
            sizeSMEM += sizeCoef;

        return Sizes {
            .coef = sizeCoef,
            .load = sizeLoad,
            .smem = sizeSMEM
        };

    }

    // Select the largest N ∈ {1,2,4} such that channels_z / N >= 32.
    //
    // Assumes LHS operand `x` carries channel content, and ensures
    // N divides `p.channels_x`.
    static int get_vectorization(Params p) {
        int ch = p.channels_z();
        if (ch / 4 >= 32 && p.channels_x % 4 == 0) {
            return 4;
        } else if (ch / 2 >= 32 && p.channels_x % 2 == 0) {
            return 2;
        } else {
            return 1;
        }
    }

    static LaunchConfigBwd get_hints(Params p, int debug, int N=-1) {

        // SMEM : [ kz * num_z | kx * num_x or ky * num_y ]

        // Query device properties for SMEM bounds.
        DeviceProperties device = DeviceProperties::query();

        // Best possible vectorization given channel dimensions
        bool find_N = (N == -1);
        if (find_N)
            N = get_vectorization(p);

        ParamsBwd p_bwd = GetParamsBwd(p);
        using CfgLHS = LaunchConfig<Idx, Val, kxMode>;
        using CfgRHS = LaunchConfig<Idx, Val, kyMode>;
        CfgLHS cfg_lhs = CfgLHS::get_hints(p_bwd.lhs, 0, -1);
        CfgRHS cfg_rhs = CfgRHS::get_hints(p_bwd.rhs, 0, -1);

        dim3 grid = {
            min(p.num_rows, 8192),
            1,
            1
        };

        dim3 block = {
            min(cfg_lhs.blockDim.x, cfg_rhs.blockDim.x),
            min(cfg_lhs.blockDim.y, cfg_rhs.blockDim.y),
            1
        };

        if (find_N)
            N = min(cfg_lhs.N, cfg_rhs.N);

        Sizes sizes = get_sizes(p, block, N);

        LaunchConfigBwd cfg {
            .gridDim = grid,
            .blockDim = block,
            .sizeSMEM = sizes.smem,
            .N = N,
        };


        cfg.log(sizes, p, debug);
        cfg.validate(sizes, device, debug);
        return cfg;
    }

    void validate(Sizes sizes, DeviceProperties device, int debug) {
        switch(N) {
            case 4:
                return validate<4>(sizes, device, debug); break;
            case 2:
                return validate<2>(sizes, device, debug);
            default:
                return validate<1>(sizes, device, debug);
        }
    }

    template<int N>
    void validate(Sizes sizes, DeviceProperties device, int debug) {

        if (sizes.smem >= device.smem_max) {
            std::string msg =
                "e3j_ops.tensor_product: SMEM overflow ("
                + std::to_string(sizes.smem) + " B >= "
                + std::to_string(device.smem_max) + " B max)";
            error = e3j::Error::InvalidArgument(msg);
        } else if (sizes.smem > device.smem_opt_in) {
            if (debug > 0)
                printf("WARNING: opting in to %zu B > %d B shared memory",
                        sizes.smem, device.smem_opt_in);
            cudaFuncSetAttribute(
                kernel_bwd<Idx, Val, kxMode, kyMode, N>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                device.smem_max
            );
        }
    }

    void log (Sizes sizes, Params p, int debug) {
        if (debug > 1) {
            printf("> e3j::tensor_product::kernel_bwd<<<%dx%d,%dx%d>>> N=%d\n"
                   "\t sizeSMEM: %zu B\n"
                   "\t sizeCoef: %zu B\n"
                   "\t sizeLoad: %zu B\n"
                   "\t num_idx: %d\n"
                   "\t mode: %d\n"
                   "\t dims: (%d, %d) -> %d\n"
                   "\t channels: (%d, %d)\n",
                    gridDim.x, gridDim.y, blockDim.x, blockDim.y, N,
                    sizes.smem, sizes.coef, sizes.load,
                    p.num_idx, p.mode, p.num_x, p.num_y, p.num_out,
                    p.channels_x, p.channels_y
            );
        }
    }

};


template<typename Idx, typename Val, Mode kxMode, Mode kyMode>
e3j::Error launch_bwd(
    const Coef<Idx, Val> *coef,
    const Val *gmem_x,
    const Val *gmem_y,
    const Val *gmem_dz,
    Val *gmem_dx,
    Val *gmem_dy,
    Params p,
    cudaStream_t stream,
    int debug
) {

    // Determine backward modes on LHS and RHS (rotates channel arguments)
    using LaunchConfigBwd = LaunchConfigBwd<Idx,Val,kxMode, kyMode>;
    LaunchConfigBwd cfg = LaunchConfigBwd::get_hints(p, debug);
    if (cfg.error.failure())
        return cfg.error;

    int channels_z = p.channels_out();

    // Number of channels processed per block: (blockDim.x * N), capped
    // by the total channel count. Used as SMEM buffer width in BuffersBwd.
    dim3 unroll = {
        (unsigned int)min((int)cfg.blockDim.x * cfg.N, p.channels_x),
        (unsigned int)min((int)cfg.blockDim.x * cfg.N, p.channels_y),
        (unsigned int)min((int)cfg.blockDim.x * cfg.N, p.channels_out()),
    };

    CuArray2D<const Val> x  = { gmem_x,  p.num_x,   p.channels_x     };
    CuArray2D<const Val> y  = { gmem_y,  p.num_y,   p.channels_y     };
    CuArray2D<const Val> dz = { gmem_dz, p.num_out, p.channels_out() };
    CuArray2D<Val>       dx = { gmem_dx, p.num_x,   p.channels_x     };
    CuArray2D<Val>       dy = { gmem_dy, p.num_y,   p.channels_y     };

    #define LAUNCH_BWD(N)                                               \
    kernel_bwd<Idx,Val,kxMode,kyMode,N>                                 \
        <<<cfg.gridDim, cfg.blockDim, cfg.sizeSMEM, stream>>>           \
        (coef, x, y, dz, dx, dy, p.num_rows, p.num_idx, unroll)

    switch(cfg.N) {
        case 4:  LAUNCH_BWD(4); break;
        case 2:  LAUNCH_BWD(2); break;
        default: LAUNCH_BWD(1); break;
    };

    #undef LAUNCH_BWD

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
        return e3j::Error::FromCudaLaunch(launch_err);

    if (debug > 0) {
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            return e3j::Error::FromCudaLaunch(error);
        }
    }

    return e3j::Error::Success();
}


} // namespace trailing_channels
} // namespace tensor_product
} // namespace e3j

#endif // _E3J_TENSOR_PRODUCT_BWD_LAUNCH_TRAILING_H_
