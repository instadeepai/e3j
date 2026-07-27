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

#ifndef _E3J_CONVOLUTION_LAUNCH_FWD_H_
#define _E3J_CONVOLUTION_LAUNCH_FWD_H_

#include "cuda/convolution.cuh"
#include "cuda/convolution/fwd.cuh"
#include "cuda/utils.cuh"
#include "cuda/array.cuh"
#include "ffi/error.h"

// Yields better results than 1024 even if lower occupancy is obtained.
// Given register pressure, 2x1024 might often not fit anyways, however
// three blocks of 512 might coexist on the SM.
#define MAX_BLOCK_SIZE 512

namespace e3j {
namespace convolution {

using utils::DeviceProperties;

struct Sizes {
    size_t coef;
    size_t load;
    size_t smem;
};

template<typename Idx, typename Val>
struct LaunchConfig {

    dim3 gridDim;
    dim3 blockDim;
    size_t sizeSMEM;
    e3j::Error error;
    int N;

    // SMEM layout matching Buffers::to_shared<Stages=1>:
    //   [ coef | lhs | rhs | out | mix ]
    // Alignment constant A = 16 / sizeof(Val) matches to_shared.
    static Sizes get_sizes(Params p, dim3 blockDims, int N) {

        int channels_z = p.channels_x; // OUTER with channels_y = 1
        constexpr int A = 16 / sizeof(Val);

        int unroll_x = min((int)blockDims.x * N, p.channels_x);
        int unroll_z = min((int)blockDims.x * N, channels_z);

        // A-aligned buffer sizes matching Buffers::to_shared
        size_t size_lhs = ((size_t)unroll_x * p.num_x + A - 1) / A * A;
        size_t size_rhs = ((size_t)1 * p.num_y + A - 1) / A * A;
        size_t size_out = ((size_t)unroll_z * p.num_out + A - 1) / A * A;
        size_t size_mix = ((size_t)unroll_z * p.num_scalars + A - 1) / A * A;

        size_t sizeLoad = sizeof(Val)
            * (size_lhs + size_rhs + size_out + size_mix);

        size_t sizeCoef = (size_t)p.num_coef * sizeof(Coef4D<Idx,Val>);
        sizeCoef = utils::smem_align(sizeCoef);

        size_t sizeSMEM = sizeLoad + sizeCoef;

        return Sizes {
            .coef = sizeCoef,
            .load = sizeLoad,
            .smem = sizeSMEM
        };
    }

    // Select the largest N ∈ {1,2,4} such that channels_x / N >= 32.
    static int get_vectorization(Params p) {
        int ch = p.channels_x;
        if (ch / 4 >= 32 && ch % 4 == 0) {
            return 4;
        } else if (ch / 2 >= 32 && ch % 2 == 0) {
            return 2;
        } else {
            return 1;
        }
    }

    static LaunchConfig get_hints(Params p, int debug, int N=-1) {

        DeviceProperties device = DeviceProperties::query();
        int smem_top = device.smem_max / 2;
        int channels_z = p.channels_x;

        bool find_N = (N == -1);
        if (find_N)
            N = get_vectorization(p);

        // Bound blockDim.x to avoid SMEM overflow.
        // Per-channel cost: one column each of lhs, out, and mix.
        unsigned int size_per_ch = sizeof(Val) * (p.num_x + p.num_out + p.num_scalars);
        unsigned int threadsX = channels_z / N;
        while (size_per_ch * threadsX * N > smem_top && threadsX > 32) {
            threadsX /= 2;
        }
        if (find_N) {
            while (size_per_ch * threadsX * N > smem_top && N > 1) {
                N /= 2;
            }
        }
        threadsX = max(32u, 32u * (threadsX / 32));
        // Reduce N so that threadsX * N divides channels evenly.
        if (find_N) {
            while (channels_z % (threadsX * N) != 0 && N > 1)
                N /= 2;
        }

        // Estimate SMEM footprint with blockDim.y = 1.
        Sizes size = get_sizes(p, {threadsX, 1, 1}, N);

        // Parallelize work over non-overlapping output coefficients:
        //  - bound blockDim.y by the number of output coordinates
        //  - ensure maxBlocks * blockSize tends to 2048 threads/SM
        unsigned int maxBlocks = device.smem_max / size.smem;
        unsigned int threadsY = 1;
        while (
            maxBlocks * threadsX * threadsY < 2048
            and threadsX * threadsY < MAX_BLOCK_SIZE
            and threadsY <= (unsigned int)p.num_out / 2
        ) {
            threadsY *= 2;
        }

        Sizes sizes = get_sizes(p, {threadsX, threadsY, 1}, N);

        unsigned int blocksX = min(p.num_nodes, (int32_t)8192);

        LaunchConfig cfg {
            .gridDim = {blocksX, 1, 1},
            .blockDim = {threadsX, threadsY, 1},
            .sizeSMEM = sizes.smem,
            .N = N,
        };

        cfg.log(sizes, p, debug);
        cfg.validate(sizes, device, debug);
        return cfg;
    }

    void validate(Sizes sizes, DeviceProperties device, int debug) {
        switch(N) {
            case 4:  return validate<4>(sizes, device, debug);
            case 2:  return validate<2>(sizes, device, debug);
            default: return validate<1>(sizes, device, debug);
        }
    }

    template<int N>
    void validate(Sizes sizes, DeviceProperties device, int debug) {

        if (sizes.smem >= device.smem_max) {
            std::string msg =
                "e3j_ops.convolution: SMEM overflow ("
                + std::to_string(sizes.smem) + " B >= "
                + std::to_string(device.smem_max) + " B max)";
            error = e3j::Error::InvalidArgument(msg);

        } else if (sizes.smem > device.smem_opt_in) {
            cudaFuncSetAttribute(
                convolution::kernel<Idx, Val, N>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                sizes.smem + 16
            );
            if (debug > 0)
                printf("WARNING: opting in to %zu B > %d B shared memory",
                       sizes.smem, device.smem_opt_in);
        }
    }

    void log(Sizes sizes, Params p, int debug) {
        if (debug > 1) {
            printf("> e3j::convolution::kernel<<<%dx%d,%dx%d>>> N=%d\n"
                   "\t sizeSMEM: %zu B\n"
                   "\t sizeCoef: %zu B\n"
                   "\t sizeLoad: %zu B\n"
                   "\t num_coef: %d\n"
                   "\t num_nodes: %d\n"
                   "\t dims: (%d, %d) -> %d\n"
                   "\t channels_x: %d\n"
                   "\t num_scalars: %d\n",
                    gridDim.x, gridDim.y, blockDim.x, blockDim.y, N,
                    sizes.smem, sizes.coef, sizes.load,
                    p.num_coef, p.num_nodes,
                    p.num_x, p.num_y, p.num_out,
                    p.channels_x, p.num_scalars
            );
        }
    }

};


template<typename Idx, typename Val>
e3j::Error launch(
    const Coef4D<Idx, Val> *coef,
    const Val *gmem_x,
    const Val *gmem_y,
    const Val *gmem_s,
    const AdjacencyCSR adj,
    Val *gmem_out,
    Params p,
    cudaStream_t stream,
    int debug
) {

    LaunchConfig<Idx, Val> cfg = LaunchConfig<Idx, Val>::get_hints(p, debug);
    if (cfg.error.failure())
        return cfg.error;

    // Number of channels processed per stride: (blockDim.x * N), capped
    // by the total channel count. Used as SMEM buffer width in Buffers.
    int channels_z = p.channels_x; // OUTER with channels_y = 1
    dim3 unroll = {
        (unsigned int)min((int)cfg.blockDim.x * cfg.N, p.channels_x),
        1u,
        (unsigned int)min((int)cfg.blockDim.x * cfg.N, channels_z),
    };

    CuArray2D<const Val> x = { gmem_x, p.num_x,       p.channels_x };
    CuArray2D<const Val> y = { gmem_y, p.num_y,       1             };
    CuArray2D<const Val> s = { gmem_s, p.num_scalars, p.channels_x  };
    CuArray2D<Val>       m = { gmem_out, p.num_out,   p.channels_x  };

    // Dispatch over vectorization N.

    #define LAUNCH(N)                                                      \
    convolution::kernel<Idx,Val,N>                                         \
        <<<cfg.gridDim, cfg.blockDim, cfg.sizeSMEM, stream>>>              \
        (coef, x, y, s, adj, m,                                            \
         p.num_nodes, p.num_coef, unroll)

    switch(cfg.N) {
        case 4:  LAUNCH(4); break;
        case 2:  LAUNCH(2); break;
        default: LAUNCH(1);
    }

    #undef LAUNCH

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
        return e3j::Error::FromCudaLaunch(launch_err);

    if (debug > 0) {
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess)
            return e3j::Error::FromCudaLaunch(error);
    }

    return e3j::Error::Success();
}


} // namespace convolution
} // namespace e3j

#undef MAX_BLOCK_SIZE

#endif // _E3J_CONVOLUTION_LAUNCH_FWD_H_
