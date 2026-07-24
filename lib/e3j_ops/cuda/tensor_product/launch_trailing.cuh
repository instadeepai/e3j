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

#ifndef _E3J_TENSOR_PRODUCT_LAUNCH_TRAILING_H_
#define _E3J_TENSOR_PRODUCT_LAUNCH_TRAILING_H_

#include "cuda/tensor_product/trailing_channels.cuh"

#define MAX_BLOCK_SIZE 256

namespace e3j {
namespace tensor_product {

namespace trailing_channels {


struct Sizes {
    size_t coef;
    size_t load;
    size_t smem;
};


template<typename Idx, typename Val, Mode kMode>
struct LaunchConfig {

    dim3 gridDim;
    dim3 blockDim;
    size_t sizeSMEM;
    e3j::Error error;
    int N;

    static Sizes get_sizes(Params p, dim3 blockDims, int N) {

        constexpr int A = 16 / sizeof(Val);

        int channels_z = p.channels_z();
        // blockDims.x is in lanes; each lane handles N channels.
        // Pad x allocation to match SMEM::from_extern<N> alignment (N floats).
        int size_x_raw = p.num_x * min((int)blockDims.x * N, p.channels_x),
            size_x = (size_x_raw + (A-1)) & ~(A-1),
            size_y_raw = p.num_y * min((int)blockDims.x * N, p.channels_y),
            size_y = (size_y_raw + (A-1)) & ~(A-1);


        size_t sizeCoef = p.num_idx * sizeof(Coef<Idx,Val>);
        size_t sizeLoad = sizeof(Val) * (size_x + size_y);
        size_t sizeSMEM = sizeLoad;
        // With Mode::INNER, partial warp-wide sums accumulated in SMEM.
        if constexpr (kMode == Mode::INNER)
            sizeSMEM += sizeof(Val) * p.num_out * (1 + (p.channels_x - 1) / (32 * N));
        if (BUFFER_FWD_COEFS_IN_SMEM) {
            // Pad coef buffer to 16 B (same as kernel pointer logic) so the
            // x/y buffers that follow stay 16-byte aligned.
            size_t align = SMEM_BUFFER_ALIGN;
            sizeSMEM += (sizeCoef + align - 1) & ~(align - 1);
        }

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

    static LaunchConfig get_hints(Params p, int debug, int N=-1) {

        // SMEM : [ Coef | kx * num_x | ky * num_y ]

        // Query device properties for SMEM bounds.
        DeviceProperties device = DeviceProperties::query();

        int smem_top = device.smem_max / 2;

        // Number of hidden channels (before eventual aggregation)
        int channels_z = p.channels_z();

        // Best possible vectorization given channel dimensions
        bool find_N = (N == -1);
        if (find_N)
            N = get_vectorization(p);

        // Bound N and blockDim.x to avoid SMEM overflow.
        // Greedily process as many channels as possible until smem_top overflow.
        unsigned int size_in = (sizeof(Val) * p.num_x + sizeof(Val) * p.num_y);
        unsigned int threadsX = channels_z / N;
        while (size_in * threadsX * N > smem_top and threadsX > 32) {
            threadsX /= 2;
        }
        if (find_N) {
            while (size_in * threadsX * N > smem_top and N > 1) {
                N /= 2;
            }
        }
        threadsX = max(32, 32 * (threadsX / 32));

        // Estimate load, shared and coef sizes with blockDim.y = 1.
        Sizes size = get_sizes(p, {threadsX, 1, 1}, N);

        // Parallelize work over non-overlapping output coefficients:
        //  - bound blockDim.y by the number of output coordinates
        //  - ensure maxBlocks * blockSize tends to 2048 threads/SM
        unsigned int maxBlocks = device.smem_max / size.smem;
        unsigned int threadsY = 1;
        while (
            maxBlocks * threadsX * threadsY < 2048
            and threadsX * threadsY < MAX_BLOCK_SIZE
            and threadsY <=  (unsigned int)p.num_out / 2
        ) {
            threadsY *= 2;
        }

        Sizes sizes = get_sizes(p, {threadsX, threadsY, 1}, N);

        int threadsPerBlock = threadsX * threadsY;
        unsigned int blocksX = max(2, 2048 / threadsPerBlock);

        LaunchConfig cfg {
            .gridDim = {132 * blocksX, 1, 1},
            .blockDim = {threadsX, threadsY, 1},
            .sizeSMEM = sizes.smem,
            .N = N,
        };

        // Validate launch config, eventually using the CUDA runtime API
        // to register higher opt-in memory bounds on the kernel instance.
        cfg.log(sizes, p, debug);
        cfg.validate(sizes, device, debug);
        return cfg;
    }

    void validate (Sizes sizes, DeviceProperties device, int debug) {
        switch(N) {
            case 4:
                return validate<4>(sizes, device, debug);
            case 2:
                return validate<2>(sizes, device, debug);
            default:
                return validate<1>(sizes, device, debug);
        }
    }

    template <int N>
    void validate (Sizes sizes, DeviceProperties device, int debug) {

        // Raise if above hardware max
        if (sizes.smem >= device.smem_max) {
            std::string msg =
                "e3j_ops.tensor_product: SMEM overflow ("
                + std::to_string(sizes.smem) + " B >= "
                + std::to_string(device.smem_max) + " B max)";
            error = e3j::Error::InvalidArgument(msg);

        // Call CUDA runtime API for higher SMEM opt-in.
        // Requires N as a template parameter. Opt-in both strided
        // and unstrided kernel instances.
        } else if (sizes.smem > device.smem_opt_in) {
            cudaFuncSetAttribute(
                trailing_channels::kernel<Idx,Val,kMode,N,false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                device.smem_max
            );
            cudaFuncSetAttribute(
                trailing_channels::kernel<Idx,Val,kMode,N,true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                device.smem_max
            );
            if (debug > 0)
                printf(
                    "WARNING: opting in to %zu B > %d B shared memory",
                    sizes.smem, device.smem_opt_in
                );
        }
    }

    void log (Sizes sizes, Params p, int debug) {
        if (debug > 1) {
            printf("> e3j::tensor_product::kernel<<<%dx%d,%dx%d>>> N=%d\n"
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


template<typename Idx, typename Val, Mode kMode>
e3j::Error launch(
    const Coef<Idx,Val> *coef,
    const Val *gmem_x,
    const Val *gmem_y,
    Val *gmem_out,
    Params p,
    cudaStream_t stream,
    int debug
) {
    LaunchConfig<Idx,Val,kMode> cfg = LaunchConfig<Idx,Val,kMode>::get_hints(p, debug);
    if (cfg.error.failure())
        return cfg.error;

    // Number of channels effectively processed (N * blockDim.x)
    // caps each SMEM buffer's channel dimension.
    dim3 unroll = {
        min((int)cfg.blockDim.x * cfg.N, p.channels_x),
        min((int)cfg.blockDim.x * cfg.N, p.channels_y),
        min((int)cfg.blockDim.x * cfg.N, p.channels_z()),
    };

    CuArray2D<const Val> x = { gmem_x, p.num_x, p.channels_x };
    CuArray2D<const Val> y = { gmem_y, p.num_y, p.channels_y };
    CuArray2D<Val> out = { gmem_out, p.num_out, p.channels_out() };

    bool is_strided = (int)(cfg.blockDim.x * cfg.N) <
        max(max(p.channels_x, p.channels_y), p.channels_z());


    // TODO: consider dropping STRIDED as a template parameter, if both
    //       pathways are equally efficient and don't inflate registers.

    // Dispatch over vectorization N and strided/unstrided branch.
    #define LAUNCH(N, STRIDED)                                          \
    trailing_channels::kernel<Idx,Val,kMode,N,STRIDED>                  \
        <<<cfg.gridDim, cfg.blockDim, cfg.sizeSMEM, stream>>>           \
        (coef, x, y, out, p.num_rows, p.num_idx, unroll)

    switch(cfg.N) {
        case 4:
            is_strided ? LAUNCH(4, true) : LAUNCH(4, false);
            break;
        case 2:
            is_strided ? LAUNCH(2, true) : LAUNCH(2, false);
            break;
        default:
            is_strided ? LAUNCH(1, true) : LAUNCH(1, false);
            break;
    };

    #undef LAUNCH

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        return e3j::Error::FromCudaLaunch(launch_err);
    }

    if (debug > 0) {
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            return e3j::Error::FromCudaLaunch(error);
        }
    }

    return e3j::Error::Success();
}


}// namespace trailing_channels

}// namespace tensor_product
}// namespace e3j

#endif // _E3J_TENSOR_PRODUCT_LAUNCH_TRAILING_H_
