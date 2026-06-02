#ifndef _E3J_CONVOLUTION_BWD_H_
#define _E3J_CONVOLUTION_BWD_H_

#include "cuda/convolution.cuh"
#include "cuda/convolution/fwd.cuh"
#include "cuda/tensor_product/details.cuh"
#include "cuda/tensor_product/trailing_channels.cuh"
#include "cuda/utils.cuh"

#define BUFFER_CONV_BWD_COEFS_IN_SMEM true

namespace e3j {
namespace convolution {

using utils::copy;
using utils::copy_strided;
using utils::copy_pipe;
using utils::copy_pipe_strided;
using utils::wait_pipe;
using utils::fill;
using tensor_product::CoefRange;
using tensor_product::find_coef_bounds;

namespace tp = tensor_product::trailing_channels;


// Align to 16B: sizeof(T) should divide 16.
template<typename T>
__device__ unsigned int align16(unsigned int n) {
    constexpr int A = 16 / sizeof(T);
    return (n + A - 1) / A * A;
}


/*****************************************************************
 *  Buffers for backward convolution:
 *
 *      [ dm | x | y | mix | dx | dy_scratch ]
 *
 *  - dm:   cotangent receiver messages
 *  - x:    primal node features
 *  - y:    primal edge features
 *  - mix:  edge scalars
 *  - dx:   cotangent sender message accumulator
 *  - dy_scratch:  cotangent edge features scratch memory (Mode::INNER)
 *
 *****************************************************************/
template <typename Idx, typename Val>
struct BuffersBwd {
    CuArray2D<Val> dm;
    CuArray2D<Val> x;
    CuArray2D<Val> y;
    CuArray2D<Val> mix;
    CuArray2D<Val> dx;
    Val* dy_scratch;

    static __device__ BuffersBwd init (
        CuArray2D<const Val> x,
        CuArray2D<const Val> y,
        CuArray2D<const Val> dm,
        CuArray2D<const Val> mix,
        dim3 unroll
    ) {
        return BuffersBwd {
            .dm  = { nullptr, dm.shape[0],  unroll.z },
            .x   = { nullptr, x.shape[0],   unroll.x },
            .y   = { nullptr, y.shape[0],   1u       },
            .mix = { nullptr, mix.shape[0], unroll.z },
            .dx  = { nullptr, x.shape[0],   unroll.x },
            .dy_scratch = nullptr,
        };
    }

    template<int Stages=1>
    __device__ void to_shared (char *smem_) {
        dm.data  = reinterpret_cast<Val*>(smem_);
        x.data   = dm.data  + align16<Val>(dm.size());
        y.data   = x.data   + align16<Val>(x.size());
        mix.data = y.data   + align16<Val>(y.size());
        dx.data  = mix.data + align16<Val>(mix.size());
        dy_scratch = dx.data + align16<Val>(dx.size());
    }
};


/*****************************************************************
 *  Convolution backward kernel.
 *
 *  Sender-grouped loop over the transposed CSR adjacency:
 *    - x[sender] loaded once per sender group
 *    - dx accumulated in SMEM across edges (OUTER), one GMEM
 *      store per sender — no atomics
 *    - dy written per-edge to GMEM (INNER warp reduction)
 *
 *  Coefficients in 4D COO format are packed in three orderings:
 *      [coef_dmix | coef_dx | coef_dy]
 *
 *  Given forward coefficients indices (i, j, k, l) for (m, x, s, y)
 *  the orderings should always keep the broadcasted operand y last,
 *  and keep dm first by convention:
 *
 *   - coef_dmix :  {val, k, i, j, l}  for bigotimes(dm, x, y)
 *   - coef_dx :    {val, j, i, k, l}  for bigotimes(dm, s, y)
 *   - coef_dy :    {val, l, i, k, j}  for bigotimes(dm, s, x)
 *
 *****************************************************************/
template <typename Idx, typename Val, int N=1>
__global__ void kernel_bwd (
    const Coef4D<Idx, Val> *coef,
    CuArray2D<const Val> x,
    CuArray2D<const Val> y,
    CuArray2D<const Val> dm,
    CuArray2D<const Val> mix,
    const AdjacencyCSR adj,
    const int32_t *edge_perm,
    Val *gmem_dx,
    Val *gmem_dy,
    Val *gmem_dmix,
    unsigned int num_nodes,
    unsigned int num_coef,
    dim3 unroll
) {

    using Coef = Coef4D<Idx,Val>;

    int size_x   = x.size();
    int size_y   = y.size();
    int size_dm  = dm.size();
    int size_mix = mix.size();
    int num_y    = y.shape[0];
    int num_warps = blockDim.x / 32;

    // --- Shared memory prolog ----------------------------------------

    extern __shared__ __align__(16) char smem__[];
    char* smem_ = reinterpret_cast<char*>(smem__);
    BuffersBwd<Idx, Val> smem = BuffersBwd<Idx, Val>::init(
        x, y, dm, mix, unroll
    );

    if (BUFFER_CONV_BWD_COEFS_IN_SMEM) {
        Coef* smem_coef = reinterpret_cast<Coef*>(smem_);
        copy_pipe<1>(smem_coef, coef, 3 * num_coef);
        coef = smem_coef;
        constexpr size_t align = N * sizeof(Val);
        size_t coef_bytes = (size_t)(3 * num_coef) * sizeof(Coef);
        coef_bytes = (coef_bytes + align - 1) & ~(align - 1);
        smem_ = (char*)smem_coef + coef_bytes;
        __pipeline_commit();
        wait_pipe();
    }
    smem.to_shared(smem_);

    const Coef *coef_dx = coef + num_coef;
    const Coef *coef_dy = coef + 2 * num_coef;

    // Partition coefficients across blockDim.y
    int *smem_cuts = reinterpret_cast<int*>(smem.dm.data);
    CoefRange range_dmix = find_coef_bounds<Idx, Val, Coef4D>(
        smem_cuts, coef, num_coef, mix.shape[0]
    );
    CoefRange range_dx = find_coef_bounds<Idx, Val, Coef4D>(
        smem_cuts, coef_dx, num_coef, x.shape[0]
    );
    CoefRange range_dy = find_coef_bounds<Idx, Val, Coef4D>(
        smem_cuts, coef_dy, num_coef, num_y
    );

    // Dummy view: only shape[0] = num_y is used (scratch indexing).
    CuArray2D<Val> dy_view = {
        nullptr, (unsigned int)num_y, smem.dm.shape[1]
    };

    int num_strides = 1 + (max((int)x.shape[1], (int)dm.shape[1]) - 1)
                         / (blockDim.x * N);

    // --- Grid-stride loop over senders -------------------------------

    for (unsigned int sender = blockIdx.x;
         sender < num_nodes;
         sender += gridDim.x)
    {
        int first_edge = adj.receiver_ptr[sender],
            last_edge  = adj.receiver_ptr[sender + 1];

        CuArray2D<const Val> x_s   = { x.data,   x.shape[0],   x.shape[1] };
        CuArray2D<const Val> dm_s  = { dm.data,  dm.shape[0],  dm.shape[1] };
        CuArray2D<const Val> mix_s = { mix.data, mix.shape[0], mix.shape[1] };
        Val *dx_out_s = gmem_dx;
        Val *dmix_s   = gmem_dmix;

        for (int s = 0; s < num_strides; s++) {

        // Zero dx accumulator for this sender.
        fill(smem.dx.data, Val(0), smem.dx.size());

        // Load x[sender] to smem.x (constant across edges from sender).
        if (x.shape[1] > unroll.x)
            copy_pipe_strided(smem.x.data, x_s.data + sender * size_x,
                              x.shape[0], unroll.x, x.shape[1]);
        else
            copy_pipe<N>(smem.x.data, x_s.data + sender * size_x, smem.x.size());
        __pipeline_commit();
        wait_pipe();

        for (int edge = first_edge; edge < last_edge; edge++) {

            int edge_t = edge_perm ? edge_perm[edge] : edge;
            int recv = adj.sender[edge];

            // Load dm[recv], y[edge_t], mix[edge_t]
            if (dm.shape[1] > unroll.z)
                copy_pipe_strided(smem.dm.data, dm_s.data + recv * size_dm,
                                  dm.shape[0], unroll.z, dm.shape[1]);
            else
                copy_pipe<N>(smem.dm.data, dm_s.data + recv * size_dm, smem.dm.size());
            copy_pipe<1>(smem.y.data, y.data + edge_t * size_y, size_y);
            if (mix.shape[1] > unroll.z)
                copy_pipe_strided(smem.mix.data, mix_s.data + edge_t * size_mix,
                                  mix.shape[0], unroll.z, mix.shape[1]);
            else
                copy_pipe<N>(smem.mix.data, mix_s.data + edge_t * size_mix, smem.mix.size());
            __pipeline_commit();
            wait_pipe();

            // Edge scalar cotangents
            int dmix_edge_size = mix.shape[0] * dm.shape[1];
            CuArray2D<Val> dmix_view = {
                dmix_s + edge_t * dmix_edge_size, mix.shape[0], dm.shape[1]
            };

            bigotimes<Idx,Val,Mode::OUTER,N,false>(
                coef, range_dmix,
                smem.dm, smem.x, smem.y,
                dmix_view, nullptr
            );
            __syncthreads();

            // Sender feature cotangents
            bigotimes<Idx,Val,Mode::OUTER,N,true>(
                coef_dx, range_dx,
                smem.dm, smem.mix, smem.y,
                smem.dx, nullptr
            );

            // Prepare dy scratch for INNER warp reduction.
            fill(smem.dy_scratch, Val(0), num_warps * num_y);
            __syncthreads();

            // Edge feature cotangents
            bigotimes<Idx,Val,Mode::INNER,N,true>(
                coef_dy, range_dy,
                smem.dm, smem.mix, smem.x,
                dy_view, smem.dy_scratch
            );
            __syncthreads();

            // Reduce dy buffer, STG (single y-group to avoid GMEM race on +=)
            if (threadIdx.y == 0) {
                for (int i = threadIdx.x; i < num_y; i += blockDim.x) {
                    Val sum = 0;
                    for (int w = 0; w < num_warps; w++)
                        sum += smem.dy_scratch[w * num_y + i];
                    if (s == 0)
                        gmem_dy[edge_t * num_y + i] = sum;
                    else
                        gmem_dy[edge_t * num_y + i] += sum;
                }
            }
            __syncthreads();

        }//=== Edge loop ===

        // Store sender cotangent
        if (x.shape[1] > unroll.x)
            copy_strided(dx_out_s + sender * size_x, smem.dx.data,
                         x.shape[0], smem.dx.shape[1], x.shape[1], smem.dx.shape[1]);
        else
            copy(dx_out_s + sender * size_x, smem.dx.data, smem.dx.size());
        __syncthreads();

        // Advance to next channel slice.
        if (x.shape[1]  > unroll.x) {
            x_s.data  += unroll.x;
            dx_out_s += unroll.x;
        }
        if (dm.shape[1] > unroll.z) {
            dm_s.data += unroll.z;
            mix_s.data += unroll.z;
            dmix_s += unroll.z;
        }

        }//=== Channel stride loop ===

    }//=== Grid-stride loop ===

}

} // namespace convolution
} // namespace e3j


#endif // _E3J_CONVOLUTION_BWD_H_
