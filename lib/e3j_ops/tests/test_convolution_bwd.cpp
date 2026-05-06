#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>

#include "vec.h"

#include "cuda.h"
#include "cuda_runtime_api.h"

#include "cuda/convolution_bwd.cuh"

#define NUM_NODES 4
#define NUM_EDGES 6
#define NUM_CHANNELS 32
#define NUM_STRIPS 3
#define DEBUG 2

using e3j::convolution::Params;
using e3j::convolution::AdjacencyCSR;
using e3j::tensor_product::Coef;

using vec::Vec;
using int32 = std::int32_t;


template <typename Idx>
std::vector<Coef<Idx, float>> packCoefs(
    Vec<Idx> idx_flat, Vec<float> val, int nnz
) {
    std::vector<Coef<Idx, float>> out(nnz);
    for (int n = 0; n < nnz; n++) {
        out[n] = {
            .val = val[n],
            .i = idx_flat[n],
            .j = idx_flat[n + nnz],
            .k = idx_flat[n + nnz * 2],
        };
    }
    return out;
}


// Transpose coefficient indices for backward pass.
//   coef_dx: (j, i, k) sorted by j — OUTER on dx
//   coef_dy: (k, i, j) sorted by k — INNER on dy
template <typename Idx>
std::vector<Coef<Idx, float>> transposeCoefs(
    std::vector<Coef<Idx, float>> coef,
    int target  // 0 = dx (j,i,k), 1 = dy (k,i,j)
) {
    int n = (int)coef.size();
    std::vector<Coef<Idx, float>> out(n);
    for (int c = 0; c < n; c++) {
        if (target == 0) {
            out[c] = { coef[c].val, coef[c].j, coef[c].i, coef[c].k };
        } else {
            out[c] = { coef[c].val, coef[c].k, coef[c].i, coef[c].j };
        }
    }
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) {
        return a.i < b.i || (a.i == b.i && a.j < b.j);
    });
    return out;
}


// Transpose CSR: group edges by sender instead of receiver.
struct TransposedCSR {
    Vec<int32> receiver;     // per-edge receiver in transposed order
    Vec<int32> sender_ptr;   // CSR row pointers by sender
    Vec<int32> perm;         // perm[new_edge] = old_edge
};

TransposedCSR transpose_csr(
    Vec<int32> sender,
    Vec<int32> receiver_ptr,
    int num_nodes
) {
    int num_edges = sender.size();

    // Expand receiver_ptr to per-edge receiver
    Vec<int32> receiver(num_edges);
    for (int r = 0; r < num_nodes; r++)
        for (int e = receiver_ptr[r]; e < receiver_ptr[r + 1]; e++)
            receiver[e] = r;

    // Argsort edges by sender
    Vec<int32> perm = vec::arange<int32>(num_edges);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        return sender[a] < sender[b];
    });

    // Permuted receiver array
    Vec<int32> receiver_t(num_edges);
    for (int i = 0; i < num_edges; i++)
        receiver_t[i] = receiver[perm[i]];

    // Sender pointer via bincount + cumsum
    Vec<int32> sender_ptr_t(num_nodes + 1);
    for (int i = 0; i <= num_nodes; i++) sender_ptr_t[i] = 0;
    for (int e = 0; e < num_edges; e++)
        sender_ptr_t[sender[e] + 1]++;
    for (int i = 1; i <= num_nodes; i++)
        sender_ptr_t[i] += sender_ptr_t[i - 1];

    return { receiver_t, sender_ptr_t, perm };
}


// Permute per-edge data by perm.
Vec<float> permute_edges(Vec<float> data, Vec<int32> perm, int stride) {
    int num_edges = perm.size();
    Vec<float> out(num_edges * stride);
    for (int e = 0; e < num_edges; e++) {
        int oe = perm[e];
        for (int i = 0; i < stride; i++)
            out[e * stride + i] = data[oe * stride + i];
    }
    return out;
}


// CPU reference for convolution backward.
//
// Iterates over (receiver, edge, sender) in original CSR order.
// Produces dx in node order and dy in original edge order.
template<typename Idx>
void cpu_convolution_bwd(
    std::vector<Coef<Idx, float>> coef,
    Vec<float> x,
    Vec<float> y,
    Vec<float> dz,
    Vec<float> mix,
    Vec<Idx> irrep_out,
    Vec<int32> sender,
    Vec<int32> receiver_ptr,
    Params p,
    Vec<float>& dx_out,
    Vec<float>& dy_out,
    Vec<float>& dmix_out
) {
    int num_coef = (int)coef.size();

    for (int recv = 0; recv < p.num_nodes; recv++) {
        int first = receiver_ptr[recv];
        int last  = receiver_ptr[recv + 1];

        for (int edge = first; edge < last; edge++) {
            int send = sender[edge];

            for (int ch = 0; ch < p.channels_x; ch++) {

                // dmix[edge,s,ch] += sum_c(c.val * x_j * y_k * dz_i)
                for (int c = 0; c < num_coef; c++) {
                    int i = coef[c].i, j = coef[c].j, k = coef[c].k;
                    float v = coef[c].val;
                    float x_val = x[send * p.num_x * p.channels_x
                                    + j * p.channels_x + ch];
                    float y_val = y[edge * p.num_y + k];
                    float dz_val = dz[recv * p.num_out * p.channels_x
                                      + i * p.channels_x + ch];
                    int s = irrep_out[i];
                    dmix_out[edge * p.num_scalars * p.channels_x
                             + s * p.channels_x + ch] += v * x_val * y_val * dz_val;
                }

                // Cotangent receiver message: dm = mix * dz
                std::vector<float> dm(p.num_out, 0.f);
                for (int i = 0; i < p.num_out; i++) {
                    int s = irrep_out[i];
                    float mix_val = mix[edge * p.num_scalars * p.channels_x
                                        + s * p.channels_x + ch];
                    float dz_val = dz[recv * p.num_out * p.channels_x
                                      + i * p.channels_x + ch];
                    dm[i] = mix_val * dz_val;
                }

                // dx contribution
                for (int c = 0; c < num_coef; c++) {
                    int i = coef[c].i, j = coef[c].j, k = coef[c].k;
                    float v = coef[c].val;
                    float y_val = y[edge * p.num_y + k];
                    dx_out[send * p.num_x * p.channels_x
                           + j * p.channels_x + ch] += v * dm[i] * y_val;
                }

                // dy contribution (sum over channels)
                for (int c = 0; c < num_coef; c++) {
                    int i = coef[c].i, j = coef[c].j, k = coef[c].k;
                    float v = coef[c].val;
                    float x_val = x[send * p.num_x * p.channels_x
                                    + j * p.channels_x + ch];
                    dy_out[edge * p.num_y + k] += v * dm[i] * x_val;
                }
            }
        }
    }
}


template <typename Idx>
struct ConvBwdArgs {
    std::vector<Coef<Idx, float>> coef_packed;
    Vec<float> x;
    Vec<float> y_t;
    Vec<float> dz;
    Vec<float> mix_t;
    Vec<Idx> irrep_out;
    Vec<int32> receiver_t;
    Vec<int32> sender_ptr;
    Vec<int32> perm;
    Vec<float> dx_expect;
    Vec<float> dy_expect_t;   // in transposed edge order
    Vec<float> dmix_expect_t; // in transposed edge order
};


template <typename Idx>
ConvBwdArgs<Idx> prepareHostArgs(Params p, int num_strips, bool scale_r) {

    int n_per_strip = 6;
    int num_edges = NUM_EDGES;

    // Strip-wise coefficient indices (sorted by output index i)
    Vec<Idx> idx_i {0, 0, 0, 1, 1, 1};
    Vec<Idx> idx_j {0, 0, 0, 1, 1, 1};
    Vec<Idx> idx_k {0, 1, 2, 0, 1, 2};
    Vec<float> coef_val = {1., 1., 1., 1., 1., 1.};

    // Tile and offset for multiple strips
    idx_i = idx_i.tile(num_strips);
    idx_j = idx_j.tile(num_strips);
    idx_k = idx_k.tile(num_strips);

    idx_i = idx_i + vec::arange<Idx>(num_strips).repeat(n_per_strip) * Idx(2);
    idx_j = idx_j + vec::arange<Idx>(num_strips).repeat(n_per_strip) * Idx(2);
    idx_k = idx_k + vec::arange<Idx>(num_strips).repeat(n_per_strip) * Idx(3);

    Vec<Idx> idx_flat = Vec<Idx>::concat({idx_i, idx_j, idx_k});
    Vec<float> val = coef_val.tile(num_strips);
    int num_coef = p.num_coef;
    std::vector<Coef<Idx, float>> coef_fwd =
        packCoefs<Idx>(idx_flat, val, num_coef);

    // Transpose coefs for backward kernel
    std::vector<Coef<Idx, float>> coef_dx = transposeCoefs<Idx>(coef_fwd, 0);
    std::vector<Coef<Idx, float>> coef_dy = transposeCoefs<Idx>(coef_fwd, 1);

    std::vector<Coef<Idx, float>> coef_packed;
    coef_packed.reserve(3 * num_coef);
    coef_packed.insert(coef_packed.end(), coef_fwd.begin(), coef_fwd.end());
    coef_packed.insert(coef_packed.end(), coef_dx.begin(), coef_dx.end());
    coef_packed.insert(coef_packed.end(), coef_dy.begin(), coef_dy.end());

    // Node features x: [num_nodes, num_x, channels_x]
    Vec<float> x_strip = {1., 2.};
    Vec<float> x_per_node = x_strip.tile(num_strips).repeat(p.channels_x);
    Vec<float> x = x_per_node.tile(p.num_nodes);

    // Edge embeddings y: [num_edges, num_y] (channels_y = 1)
    Vec<float> y_strip = {3., 4., 5.};
    Vec<float> y = y_strip.tile(num_strips).tile(num_edges);

    // Upstream gradient dz: [num_nodes, num_out, channels_x]
    Vec<float> dz_strip = {1., 1.};
    Vec<float> dz = dz_strip.tile(num_strips).repeat(p.channels_x)
                            .tile(p.num_nodes);

    // Radial scalars mix: [num_edges, num_scalars, channels_x]
    Vec<float> mix(num_edges * p.num_scalars * p.channels_x);
    if (scale_r) {
        for (int e = 0; e < num_edges; e++)
            for (int s = 0; s < p.num_scalars; s++)
                for (int ch = 0; ch < p.channels_x; ch++)
                    mix[e * p.num_scalars * p.channels_x
                        + s * p.channels_x + ch] =
                            (float)(e + 1) * (float)(s + 1);
    } else {
        mix = Vec<float>::ones(num_edges * p.num_scalars * p.channels_x);
    }

    // irrep_out: 2 output features per strip, one scalar per strip
    Vec<Idx> irrep_out(p.num_out);
    for (int s = 0; s < num_strips; s++) {
        irrep_out[2 * s]     = Idx(s);
        irrep_out[2 * s + 1] = Idx(s);
    }

    // Graph adjacency (CSR, same as forward test)
    Vec<int32> sender_fwd       = {1, 2, 0, 3, 0, 1};
    Vec<int32> receiver_ptr_fwd = {0, 2, 4, 5, 6};

    // Transpose CSR for backward
    TransposedCSR tcsr = transpose_csr(sender_fwd, receiver_ptr_fwd, p.num_nodes);

    // Permute per-edge data to transposed order
    Vec<float> y_t   = permute_edges(y,   tcsr.perm, p.num_y);
    Vec<float> mix_t = permute_edges(mix, tcsr.perm,
                                     p.num_scalars * p.channels_x);

    // CPU reference (in original edge order)
    Vec<float> dx_expect = Vec<float>::zeros(
        p.num_nodes * p.num_x * p.channels_x
    );
    Vec<float> dy_expect = Vec<float>::zeros(num_edges * p.num_y);
    Vec<float> dmix_expect = Vec<float>::zeros(
        num_edges * p.num_scalars * p.channels_x
    );

    cpu_convolution_bwd<Idx>(
        coef_fwd, x, y, dz, mix, irrep_out,
        sender_fwd, receiver_ptr_fwd, p,
        dx_expect, dy_expect, dmix_expect
    );

    // Permute expected dy, dmix to transposed edge order
    Vec<float> dy_expect_t = permute_edges(dy_expect, tcsr.perm, p.num_y);
    Vec<float> dmix_expect_t = permute_edges(
        dmix_expect, tcsr.perm, p.num_scalars * p.channels_x
    );

    return ConvBwdArgs<Idx> {
        .coef_packed = coef_packed,
        .x = x,
        .y_t = y_t,
        .dz = dz,
        .mix_t = mix_t,
        .irrep_out = irrep_out,
        .receiver_t = tcsr.receiver,
        .sender_ptr = tcsr.sender_ptr,
        .perm = tcsr.perm,
        .dx_expect = dx_expect,
        .dy_expect_t = dy_expect_t,
        .dmix_expect_t = dmix_expect_t,
    };
}


template <typename Idx>
int test_convolution_bwd(Params p, int num_strips, bool scale_r = false) {

    #define H2D cudaMemcpyHostToDevice
    #define D2H cudaMemcpyDeviceToHost

    printf("=== Test convolution_bwd ======================================\n");
    printf("Idx size: %zu bytes\n", sizeof(Idx));
    printf("num_nodes: %d, channels_x: %d, num_strips: %d, scale_r: %d\n",
           p.num_nodes, p.channels_x, num_strips, scale_r);

    using CoefT = Coef<Idx, float>;

    ConvBwdArgs<Idx> args = prepareHostArgs<Idx>(p, num_strips, scale_r);

    printf("Moving inputs to device...\n");

    int num_edges = NUM_EDGES;
    int num_coef = p.num_coef;

    size_t size_coef   = sizeof(CoefT) * 3 * num_coef;
    size_t size_x      = sizeof(float) * p.num_nodes * p.num_x * p.channels_x;
    size_t size_y      = sizeof(float) * num_edges * p.num_y;
    size_t size_dz     = sizeof(float) * p.num_nodes * p.num_out * p.channels_x;
    size_t size_mix    = sizeof(float) * num_edges * p.num_scalars * p.channels_x;
    size_t size_irrep  = sizeof(Idx) * p.num_out;
    size_t size_dx     = sizeof(float) * p.num_nodes * p.num_x * p.channels_x;
    size_t size_dy     = sizeof(float) * num_edges * p.num_y;
    size_t size_dmix   = sizeof(float) * num_edges * p.num_scalars * p.channels_x;

    CoefT *coef_d;
    float *x_d, *y_d, *dz_d, *mix_d, *dx_d, *dy_d, *dmix_d;
    Idx *irrep_out_d;
    int32 *receiver_t_d, *sender_ptr_d;

    cudaMalloc((void**)&coef_d, size_coef);
    cudaMalloc((void**)&x_d, size_x);
    cudaMalloc((void**)&y_d, size_y);
    cudaMalloc((void**)&dz_d, size_dz);
    cudaMalloc((void**)&mix_d, size_mix);
    cudaMalloc((void**)&irrep_out_d, size_irrep);
    cudaMalloc((void**)&receiver_t_d, sizeof(int32) * num_edges);
    cudaMalloc((void**)&sender_ptr_d, sizeof(int32) * (p.num_nodes + 1));
    cudaMalloc((void**)&dx_d, size_dx);
    cudaMalloc((void**)&dy_d, size_dy);
    cudaMalloc((void**)&dmix_d, size_dmix);

    cudaMemcpy(coef_d, args.coef_packed.data(), size_coef, H2D);
    cudaMemcpy(x_d, args.x.data(), size_x, H2D);
    cudaMemcpy(y_d, args.y_t.data(), size_y, H2D);
    cudaMemcpy(dz_d, args.dz.data(), size_dz, H2D);
    cudaMemcpy(mix_d, args.mix_t.data(), size_mix, H2D);
    cudaMemcpy(irrep_out_d, args.irrep_out.data(), size_irrep, H2D);
    cudaMemcpy(receiver_t_d, args.receiver_t.data(),
               sizeof(int32) * num_edges, H2D);
    cudaMemcpy(sender_ptr_d, args.sender_ptr.data(),
               sizeof(int32) * (p.num_nodes + 1), H2D);

    // Zero outputs
    cudaMemset(dx_d, 0, size_dx);
    cudaMemset(dy_d, 0, size_dy);
    cudaMemset(dmix_d, 0, size_dmix);

    cudaDeviceSynchronize();

    printf("Calling convolution backward kernel...\n");

    // Transposed CSR: receiver_t = per-edge receivers, sender_ptr = row ptrs
    AdjacencyCSR adj_t = { receiver_t_d, sender_ptr_d };

    e3j::Error err = e3j::convolution::launch_bwd<Idx, float>(
        coef_d, x_d, y_d, dz_d, mix_d, irrep_out_d, adj_t,
        dx_d, dy_d, dmix_d,
        p, cudaStream_t(0), DEBUG
    );

    if (err.failure()) {
        printf("Launch error: %s\n", err.message().c_str());
        cudaFree(coef_d); cudaFree(x_d); cudaFree(y_d); cudaFree(dz_d);
        cudaFree(mix_d); cudaFree(irrep_out_d);
        cudaFree(receiver_t_d); cudaFree(sender_ptr_d);
        cudaFree(dx_d); cudaFree(dy_d); cudaFree(dmix_d);
        return 1;
    }

    cudaDeviceSynchronize();

    Vec<float> result_dx(p.num_nodes * p.num_x * p.channels_x);
    Vec<float> result_dy(num_edges * p.num_y);
    Vec<float> result_dmix(num_edges * p.num_scalars * p.channels_x);
    cudaMemcpy(result_dx.data(), dx_d, size_dx, D2H);
    cudaMemcpy(result_dy.data(), dy_d, size_dy, D2H);
    cudaMemcpy(result_dmix.data(), dmix_d, size_dmix, D2H);
    cudaDeviceSynchronize();

    // Compare dx
    float tol = 1e-4f;
    bool dx_match = true;
    for (int i = 0; i < (int)result_dx.size(); i++) {
        if (std::abs(result_dx[i] - args.dx_expect[i]) > tol) {
            dx_match = false;
            break;
        }
    }

    // Compare dy (in transposed edge order)
    bool dy_match = true;
    for (int i = 0; i < (int)result_dy.size(); i++) {
        if (std::abs(result_dy[i] - args.dy_expect_t[i]) > tol) {
            dy_match = false;
            break;
        }
    }

    // Compare dmix (in transposed edge order)
    bool dmix_match = true;
    for (int i = 0; i < (int)result_dmix.size(); i++) {
        if (std::abs(result_dmix[i] - args.dmix_expect_t[i]) > tol) {
            dmix_match = false;
            break;
        }
    }

    printf("dx:   %s\n", dx_match   ? "✅" : "❌");
    printf("dy:   %s\n", dy_match   ? "✅" : "❌");
    printf("dmix: %s\n", dmix_match ? "✅" : "❌");

    if (!dx_match) {
        printf("Expect dx:\n");
        vec::showMatrix(std::cout, args.dx_expect.data(),
                       p.num_nodes * p.num_x, p.channels_x);
        printf("Result dx:\n");
        vec::showMatrix(std::cout, result_dx.data(),
                       p.num_nodes * p.num_x, p.channels_x);
    }
    if (!dy_match) {
        printf("Expect dy (transposed order):\n");
        vec::showMatrix(std::cout, args.dy_expect_t.data(),
                       NUM_EDGES, p.num_y);
        printf("Result dy:\n");
        vec::showMatrix(std::cout, result_dy.data(),
                       NUM_EDGES, p.num_y);
    }
    if (!dmix_match) {
        printf("Expect dmix (transposed order):\n");
        vec::showMatrix(std::cout, args.dmix_expect_t.data(),
                       NUM_EDGES * p.num_scalars, p.channels_x);
        printf("Result dmix:\n");
        vec::showMatrix(std::cout, result_dmix.data(),
                       NUM_EDGES * p.num_scalars, p.channels_x);
    }

    cudaFree(coef_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(dz_d);
    cudaFree(mix_d);
    cudaFree(irrep_out_d);
    cudaFree(receiver_t_d);
    cudaFree(sender_ptr_d);
    cudaFree(dx_d);
    cudaFree(dy_d);
    cudaFree(dmix_d);

    return (dx_match && dy_match && dmix_match) ? 0 : 1;
}


int main() {

    int num_strips = NUM_STRIPS;
    int num_coef   = 6 * num_strips;
    int num_x      = 2 * num_strips;
    int num_y      = 3 * num_strips;
    int num_out    = 2 * num_strips;
    int num_scalars = num_strips;

    Params p1 = {
        /*num_nodes*/   NUM_NODES,
        /*num_coef*/    num_coef,
        /*num_x*/       num_x,
        /*num_y*/       num_y,
        /*num_out*/     num_out,
        /*num_scalars*/ num_scalars,
        /*channels_x*/  NUM_CHANNELS,
    };

    int fail = 0;

    // Test 1: r = 1 (identity mixing), N=1, int32
    fail |= test_convolution_bwd<int32>(p1, num_strips, false);

    // Test 2: r = (edge+1)*(scalar+1) (scaled mixing), N=1, int32
    fail |= test_convolution_bwd<int32>(p1, num_strips, true);

    // Test 3: 64 channels (N=2)
    Params p2 = {
        /*num_nodes*/   NUM_NODES,
        /*num_coef*/    num_coef,
        /*num_x*/       num_x,
        /*num_y*/       num_y,
        /*num_out*/     num_out,
        /*num_scalars*/ num_scalars,
        /*channels_x*/  64,
    };
    fail |= test_convolution_bwd<int32>(p2, num_strips, false);

    // Test 4: 128 channels (N=4)
    Params p3 = {
        /*num_nodes*/   NUM_NODES,
        /*num_coef*/    num_coef,
        /*num_x*/       num_x,
        /*num_y*/       num_y,
        /*num_out*/     num_out,
        /*num_scalars*/ num_scalars,
        /*channels_x*/  128,
    };
    fail |= test_convolution_bwd<int32>(p3, num_strips, true);

    // Test 5: 256 channels (strided: channels_x > blockDim.x * N)
    Params p4 = {
        /*num_nodes*/   NUM_NODES,
        /*num_coef*/    num_coef,
        /*num_x*/       num_x,
        /*num_y*/       num_y,
        /*num_out*/     num_out,
        /*num_scalars*/ num_scalars,
        /*channels_x*/  256,
    };
    fail |= test_convolution_bwd<int32>(p4, num_strips, true);

    // Test 6: 256 channels with large irreps (forces blockDim.x=32, num_strides=2)
    // size_per_ch = 4*(16+32+8) = 224, 224*64*4 = 57344 > smem_top → threadsX=32
    {
        int strips_lg = 8;
        Params p5 = {
            /*num_nodes*/   NUM_NODES,
            /*num_coef*/    6 * strips_lg,
            /*num_x*/       2 * strips_lg,
            /*num_y*/       3 * strips_lg,
            /*num_out*/     2 * strips_lg,
            /*num_scalars*/ strips_lg,
            /*channels_x*/  256,
        };
        fail |= test_convolution_bwd<int32>(p5, strips_lg, true);
    }

    // Test 7: narrow Idx (uint8)
    fail |= test_convolution_bwd<std::uint8_t>(p1, num_strips, false);
    fail |= test_convolution_bwd<std::uint8_t>(p1, num_strips, true);

    return fail;
}
