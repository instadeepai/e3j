#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <vector>

#include "vec.h"

#include "cuda.h"
#include "cuda_runtime_api.h"

#include "cuda/convolution.cuh"
#include "cuda/convolution/utils.cuh"

#define NUM_NODES 4
#define NUM_EDGES 6
#define NUM_CHANNELS 32
#define NUM_STRIPS 3
#define DEBUG 2

using e3j::convolution::Params;
using e3j::convolution::AdjacencyCSR;
using e3j::convolution::Coef4D;
using e3j::convolution::otimes_mix_coefficients;
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


// CPU reference: for each receiver, sum reweighted tensor products over edges.
//
//   out[recv, i, ch] += r[edge, irrep_out[i]]
//                     * sum_c(c.val * x[sender, c.j, ch] * y[edge, c.k])
//
// Layout: trailing channels (feature-major).
template<typename Idx>
Vec<float> cpu_convolution(
    std::vector<Coef<Idx, float>> coef,
    Vec<float> x,
    Vec<float> y,
    Vec<float> r,
    Vec<Idx> irrep_out,
    Vec<int32> sender,
    Vec<int32> receiver_ptr,
    Params p
) {
    int num_coef = (int)coef.size();
    Vec<float> out = Vec<float>::zeros(
        p.num_nodes * p.num_out * p.channels_x
    );

    for (int recv = 0; recv < p.num_nodes; recv++) {
        int first = receiver_ptr[recv];
        int last  = receiver_ptr[recv + 1];

        for (int edge = first; edge < last; edge++) {
            int send = sender[edge];

            for (int ch = 0; ch < p.channels_x; ch++) {
                // TP reduction per output feature (OUTER, channels_y = 1)
                std::vector<float> tp(p.num_out, 0.f);
                for (int c = 0; c < num_coef; c++) {
                    int i = coef[c].i;
                    int j = coef[c].j;
                    int k = coef[c].k;
                    float v = coef[c].val;
                    float x_val = x[send * p.num_x * p.channels_x
                                    + j * p.channels_x + ch];
                    float y_val = y[edge * p.num_y + k];
                    tp[i] += v * x_val * y_val;
                }
                // Scalar mixing + scatter-add
                for (int i = 0; i < p.num_out; i++) {
                    int s = irrep_out[i];
                    float r_val = r[edge * p.num_scalars * p.channels_x
                                    + s * p.channels_x + ch];
                    out[recv * p.num_out * p.channels_x
                        + i * p.channels_x + ch] += r_val * tp[i];
                }
            }
        }
    }
    return out;
}


template <typename Idx>
struct ConvArgs {
    std::vector<Coef4D<Idx, float>> coef;
    Vec<float> x;
    Vec<float> y;
    Vec<float> r;
    Vec<int32> sender;
    Vec<int32> receiver_ptr;
    Vec<float> out;
    Vec<float> expect;
};


template <typename Idx>
ConvArgs<Idx> prepareHostArgs(Params p, int num_strips, bool scale_r) {

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
    std::vector<Coef<Idx, float>> coef3 =
        packCoefs<Idx>(idx_flat, val, p.num_coef);

    // Node features x: [num_nodes, num_x, channels_x] (trailing channels)
    Vec<float> x_strip = {1., 2.};
    Vec<float> x_per_node = x_strip.tile(num_strips).repeat(p.channels_x);
    Vec<float> x = x_per_node.tile(p.num_nodes);

    // Edge embeddings y: [num_edges, num_y] (channels_y = 1)
    Vec<float> y_strip = {3., 4., 5.};
    Vec<float> y = y_strip.tile(num_strips).tile(num_edges);

    // Radial scalars r: [num_edges, num_scalars, channels_x] (trailing channels)
    Vec<float> r(num_edges * p.num_scalars * p.channels_x);
    if (scale_r) {
        // r[edge, s, ch] = (edge + 1) * (s + 1), constant across channels
        for (int e = 0; e < num_edges; e++)
            for (int s = 0; s < p.num_scalars; s++)
                for (int ch = 0; ch < p.channels_x; ch++)
                    r[e * p.num_scalars * p.channels_x
                      + s * p.channels_x + ch] = (float)(e + 1) * (float)(s + 1);
    } else {
        r = Vec<float>::ones(num_edges * p.num_scalars * p.channels_x);
    }

    // irrep_out: 2 output features per strip, one scalar per strip
    Vec<Idx> irrep_out(p.num_out);
    for (int s = 0; s < num_strips; s++) {
        irrep_out[2 * s]     = Idx(s);
        irrep_out[2 * s + 1] = Idx(s);
    }

    // Graph adjacency (CSR)
    //   Node 0 <- {1, 2}   (edges 0, 1)
    //   Node 1 <- {0, 3}   (edges 2, 3)
    //   Node 2 <- {0}      (edge 4)
    //   Node 3 <- {1}      (edge 5)
    Vec<int32> sender       = {1, 2, 0, 3, 0, 1};
    Vec<int32> receiver_ptr = {0, 2, 4, 5, 6};

    Vec<float> out = Vec<float>::zeros(
        p.num_nodes * p.num_out * p.channels_x
    );

    Vec<float> expect = cpu_convolution<Idx>(
        coef3, x, y, r, irrep_out,
        sender, receiver_ptr, p
    );

    std::vector<Coef4D<Idx, float>> coef =
        otimes_mix_coefficients<Idx, float>(coef3, irrep_out.data());

    return ConvArgs<Idx> {
        .coef = coef,
        .x = x,
        .y = y,
        .r = r,
        .sender = sender,
        .receiver_ptr = receiver_ptr,
        .out = out,
        .expect = expect,
    };
}


template <typename Idx>
int test_convolution(Params p, int num_strips, bool scale_r = false) {

    #define H2D cudaMemcpyHostToDevice
    #define D2H cudaMemcpyDeviceToHost

    printf("=== Test convolution ==========================================\n");
    printf("Idx size: %zu bytes\n", sizeof(Idx));
    printf("num_nodes: %d, channels_x: %d, num_strips: %d, scale_r: %d\n",
           p.num_nodes, p.channels_x, num_strips, scale_r);

    using CoefT = Coef4D<Idx, float>;

    ConvArgs<Idx> args = prepareHostArgs<Idx>(p, num_strips, scale_r);

    printf("Moving inputs to device...\n");

    int num_edges = NUM_EDGES;

    size_t size_coef   = sizeof(CoefT) * p.num_coef;
    size_t size_x      = sizeof(float) * p.num_nodes * p.num_x * p.channels_x;
    size_t size_y      = sizeof(float) * num_edges * p.num_y;
    size_t size_r      = sizeof(float) * num_edges * p.num_scalars * p.channels_x;
    size_t size_out    = sizeof(float) * p.num_nodes * p.num_out * p.channels_x;

    CoefT *coef_d;
    float *x_d, *y_d, *r_d, *out_d;
    int32 *sender_d, *receiver_ptr_d;

    cudaMalloc((void**)&coef_d, size_coef);
    cudaMalloc((void**)&x_d, size_x);
    cudaMalloc((void**)&y_d, size_y);
    cudaMalloc((void**)&r_d, size_r);
    cudaMalloc((void**)&sender_d, sizeof(int32) * num_edges);
    cudaMalloc((void**)&receiver_ptr_d, sizeof(int32) * (p.num_nodes + 1));
    cudaMalloc((void**)&out_d, size_out);

    cudaMemcpy(coef_d, args.coef.data(), size_coef, H2D);
    cudaMemcpy(x_d, args.x.data(), size_x, H2D);
    cudaMemcpy(y_d, args.y.data(), size_y, H2D);
    cudaMemcpy(r_d, args.r.data(), size_r, H2D);
    cudaMemcpy(sender_d, args.sender.data(), sizeof(int32) * num_edges, H2D);
    cudaMemcpy(receiver_ptr_d, args.receiver_ptr.data(),
               sizeof(int32) * (p.num_nodes + 1), H2D);
    cudaMemcpy(out_d, args.out.data(), size_out, H2D);

    cudaDeviceSynchronize();

    printf("Calling convolution kernel...\n");

    AdjacencyCSR adj = { sender_d, receiver_ptr_d };

    e3j::Error err = e3j::convolution::launch<Idx, float>(
        coef_d, x_d, y_d, r_d, adj, out_d,
        p, cudaStream_t(0), DEBUG
    );

    if (err.failure()) {
        printf("Launch error: %s\n", err.message().c_str());
        cudaFree(coef_d); cudaFree(x_d); cudaFree(y_d); cudaFree(r_d);
        cudaFree(sender_d); cudaFree(receiver_ptr_d); cudaFree(out_d);
        return 1;
    }

    cudaDeviceSynchronize();

    Vec<float> result(args.out.size());
    cudaMemcpy(result.data(), out_d, size_out, D2H);
    cudaDeviceSynchronize();

    bool match = true;
    for (int i = 0; i < (int)result.size(); i++)
        match &= (result[i] == args.expect[i]);

    printf(match ? "✅\n" : "❌\n");

    if (!match) {
        printf("Expect:\n");
        vec::showMatrix(std::cout, args.expect.data(),
                       p.num_nodes * p.num_out, p.channels_x);
        printf("Result:\n");
        vec::showMatrix(std::cout, result.data(),
                       p.num_nodes * p.num_out, p.channels_x);
    }

    cudaFree(coef_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(r_d);
    cudaFree(sender_d);
    cudaFree(receiver_ptr_d);
    cudaFree(out_d);

    return match ? 0 : 1;
}


int main() {

    int num_strips = NUM_STRIPS;
    int num_coef   = 6 * num_strips;
    int num_x      = 2 * num_strips;
    int num_y      = 3 * num_strips;
    int num_out    = 2 * num_strips;
    int num_scalars = num_strips;

    Params p1 = {
        .num_nodes   = NUM_NODES,
        .num_coef    = num_coef,
        .num_x       = num_x,
        .num_y       = num_y,
        .num_out     = num_out,
        .num_scalars = num_scalars,
        .channels_x  = NUM_CHANNELS,
    };

    // Test 1: r = 1, identity mixing — validates TP + aggregation
    test_convolution<int32>(p1, num_strips, false);

    // Test 2: r = (edge+1)*(scalar+1) — validates scalar mixing
    test_convolution<int32>(p1, num_strips, true);

    // Test 3: 64 channels (vectorization N=2)
    Params p2 = {
        .num_nodes   = NUM_NODES,
        .num_coef    = num_coef,
        .num_x       = num_x,
        .num_y       = num_y,
        .num_out     = num_out,
        .num_scalars = num_scalars,
        .channels_x  = 64,
    };
    test_convolution<int32>(p2, num_strips, false);

    // Test 4: 128 channels (vectorization N=4)
    Params p3 = {
        .num_nodes   = NUM_NODES,
        .num_coef    = num_coef,
        .num_x       = num_x,
        .num_y       = num_y,
        .num_out     = num_out,
        .num_scalars = num_scalars,
        .channels_x  = 128,
    };
    test_convolution<int32>(p3, num_strips, true);

    // Test 5: narrow Idx
    test_convolution<std::uint8_t>(p1, num_strips, false);
    test_convolution<std::uint8_t>(p1, num_strips, true);

    // Test 7: 96 channels — triggers channel striding (N=2, threadsX=32,
    //   blockDim.x * N = 64 < 96 = channels_z, num_strides = 2)
    Params p4 = {
        .num_nodes   = NUM_NODES,
        .num_coef    = num_coef,
        .num_x       = num_x,
        .num_y       = num_y,
        .num_out     = num_out,
        .num_scalars = num_scalars,
        .channels_x  = 96,
    };
    test_convolution<int32>(p4, num_strips, false);
    test_convolution<int32>(p4, num_strips, true);

    return 0;
}
