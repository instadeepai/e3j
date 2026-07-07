#ifndef _E3J_CONVOLUTION_LAUNCH_BWD_H_
#define _E3J_CONVOLUTION_LAUNCH_BWD_H_

#include "cuda/convolution.cuh"
#include "cuda/convolution/bwd.cuh"
#include "cuda/utils.cuh"
#include "cuda/array.cuh"
#include "ffi/error.h"

#define MAX_BLOCK_SIZE 256

namespace e3j {
namespace convolution {

using utils::DeviceProperties;

struct SizesBwd {
    size_t coef;
    size_t load;
    size_t smem;
};

template<typename Idx, typename Val>
struct LaunchConfigBwd {

    dim3 gridDim;
    dim3 blockDim;
    size_t sizeSMEM;
    e3j::Error error = e3j::Error::Success();
    int N;

    // SMEM layout matching BuffersBwd::to_shared:
    //   [ coef(3x) | dm | x | y | mix | dx | dy_scratch ]
    static SizesBwd get_sizes(Params p, dim3 blockDims, int N) {

        constexpr int A = 16 / sizeof(Val);
        int unroll_x = min((int)blockDims.x * N, p.channels_x);

        size_t size_dm  = ((size_t)unroll_x * p.num_out     + A-1) / A * A;
        size_t size_x   = ((size_t)unroll_x * p.num_x       + A-1) / A * A;
        size_t size_y   = ((size_t)p.num_y                   + A-1) / A * A;
        size_t size_mix = ((size_t)unroll_x * p.num_scalars  + A-1) / A * A;
        size_t size_dx  = ((size_t)unroll_x * p.num_x        + A-1) / A * A;
        int num_warps = max(1u, blockDims.x / 32);
        size_t size_scratch = (size_t)num_warps * p.num_y;

        size_t sizeLoad = sizeof(Val) * (
            size_dm + size_x + size_y + size_mix
            + size_dx + size_scratch
        );

        size_t sizeCoef = (size_t)(3 * p.num_coef) * sizeof(Coef4D<Idx,Val>);
        size_t align = N * sizeof(Val);
        sizeCoef = (sizeCoef + align - 1) & ~(align - 1);

        size_t sizeSMEM = sizeLoad + sizeCoef;

        return SizesBwd {
            .coef = sizeCoef,
            .load = sizeLoad,
            .smem = sizeSMEM
        };
    }

    static int get_vectorization(Params p) {
        int ch = p.channels_x;
        if (ch / 4 >= 32 && ch % 4 == 0) {
            return 4;
        } else
        if (ch / 2 >= 32 && ch % 2 == 0) {
            return 2;
        } else {
            return 1;
        }
    }

    static LaunchConfigBwd get_hints(Params p, int debug, int N=-1) {

        DeviceProperties device = DeviceProperties::query();
        int smem_top = device.smem_max / 2;

        bool find_N = (N == -1);
        if (find_N)
            N = get_vectorization(p);

        // Phase 1: bound blockDim.x by per-channel SMEM cost.
        // Channel-scaled buffers: dm + x + mix + dx.
        unsigned int size_per_ch =
            sizeof(Val) * (p.num_out + 2 * p.num_x + p.num_scalars);
        unsigned int threadsX = p.channels_x / N;
        while (size_per_ch * threadsX * N > smem_top && threadsX > 32) {
            threadsX /= 2;
        }
        if (find_N) {
            while (size_per_ch * threadsX * N > smem_top && N > 1) {
                N /= 2;
            }
        }
        threadsX = max(32u, 32u * (threadsX / 32));

        SizesBwd size = get_sizes(p, {threadsX, 1, 1}, N);

        // Phase 2: increase blockDim.y for occupancy.
        unsigned int maxBlocks = device.smem_max / size.smem;
        unsigned int threadsY = 1;
        while (
            maxBlocks * threadsX * threadsY < 2048
            and threadsX * threadsY < MAX_BLOCK_SIZE
            and threadsY <= (unsigned int)p.num_out / 2
        ) {
            threadsY *= 2;
        }

        SizesBwd sizes = get_sizes(p, {threadsX, threadsY, 1}, N);

        unsigned int blocksX = min(p.num_nodes, (int32_t)8192);

        LaunchConfigBwd cfg {
            .gridDim = {blocksX, 1, 1},
            .blockDim = {threadsX, threadsY, 1},
            .sizeSMEM = sizes.smem,
            .N = N,
        };

        cfg.log(sizes, p, debug);
        cfg.validate(sizes, device, debug);
        return cfg;
    }

    void validate(SizesBwd sizes, DeviceProperties device, int debug) {
        switch(N) {
            case 4:  return validate<4>(sizes, device, debug);
            case 2:  return validate<2>(sizes, device, debug);
            default: return validate<1>(sizes, device, debug);
        }
    }

    template<int N>
    void validate(SizesBwd sizes, DeviceProperties device, int debug) {

        if (sizes.smem >= device.smem_max) {
            std::string msg =
                "e3j_ops.convolution_bwd: SMEM overflow ("
                + std::to_string(sizes.smem) + " B >= "
                + std::to_string(device.smem_max) + " B max)";
            error = e3j::Error::InvalidArgument(msg);

        } else if (sizes.smem > device.smem_opt_in) {
            cudaFuncSetAttribute(
                convolution::kernel_bwd<Idx, Val, N>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                device.smem_max
            );
            if (debug > 0)
                printf("WARNING: opting in to %zu B > %d B shared memory",
                       sizes.smem, device.smem_opt_in);
        }
    }

    void log(SizesBwd sizes, Params p, int debug) {
        if (debug > 1) {
            printf("> e3j::convolution::kernel_bwd<<<%dx%d,%dx%d>>> N=%d\n"
                   "\t sizeSMEM: %zu B\n"
                   "\t sizeCoef: %zu B\n"
                   "\t sizeLoad: %zu B\n"
                   "\t num_coef: %d (x3)\n"
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
e3j::Error launch_bwd(
    const Coef4D<Idx, Val> *coef,
    const Val *gmem_x,
    const Val *gmem_y,
    const Val *gmem_dz,
    const Val *gmem_mix,
    const AdjacencyCSR adj,
    const int32_t *edge_perm,
    Val *gmem_dx,
    Val *gmem_dy,
    Val *gmem_dmix,
    Params p,
    cudaStream_t stream,
    int debug
) {

    LaunchConfigBwd<Idx, Val> cfg =
        LaunchConfigBwd<Idx, Val>::get_hints(p, debug);
    if (cfg.error.failure())
        return cfg.error;

    dim3 unroll = {
        (unsigned int)min((int)cfg.blockDim.x * cfg.N, p.channels_x),
        1u,
        (unsigned int)min((int)cfg.blockDim.x * cfg.N, p.channels_x),
    };

    CuArray2D<const Val> x   = { gmem_x,   p.num_x,       p.channels_x };
    CuArray2D<const Val> y   = { gmem_y,   p.num_y,       1             };
    CuArray2D<const Val> dz  = { gmem_dz,  p.num_out,     p.channels_x };
    CuArray2D<const Val> mix = { gmem_mix, p.num_scalars, p.channels_x  };

    #define LAUNCH_BWD(N)                                                    \
    convolution::kernel_bwd<Idx,Val,N>                                       \
        <<<cfg.gridDim, cfg.blockDim, cfg.sizeSMEM, stream>>>                \
        (coef, x, y, dz, mix, adj, edge_perm,                               \
         gmem_dx, gmem_dy, gmem_dmix, p.num_nodes, p.num_coef, unroll)

    switch(cfg.N) {
        case 4:  LAUNCH_BWD(4); break;
        case 2:  LAUNCH_BWD(2); break;
        default: LAUNCH_BWD(1);
    }

    #undef LAUNCH_BWD

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

#endif // _E3J_CONVOLUTION_LAUNCH_BWD_H_
