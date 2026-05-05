#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <algorithm>

#include <map>

#include "vec.h"

#include "cuda.h"
#include "cuda_runtime_api.h"

#include "cuda/tensor_product_bwd.cuh"

#define NUM_ROWS 12
#define NUM_CHANNELS 32 // FIXME: TRAILING_CHANNELS requires 32-multiple
#define NUM_STRIPS 3

#define UNROLL_Y 1
#define UNROLL_Z 1

#define LOG false
#define DEBUG 2

#define ASSERT_EQUAL(LHS, RHS, MSG)             \
    if (LHS != RHS) {                           \
        std::cout << "ASSERT_EQUAL:" << MSG;    \
        std::cout << "\n";                      \
        std::cout << LHS << "!=" << RHS;        \
        std::cout << "\n";                      \
        exit(1);                                \
    }


using e3j::tensor_product::Params;
using e3j::tensor_product::Mode;
using e3j::tensor_product::Layout;
using e3j::tensor_product::Coef;

using vec::Vec;

template <typename Idx>
struct Args {
    std::vector<Coef<Idx, float>> coef;
    Vec<float> x;
    Vec<float> y;
    Vec<float> dz;
    Vec<float> dx;
    Vec<float> dy;
};

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

template <typename Idx>
Args<Idx> prepareHostArgs (Params p, int num_strips=NUM_STRIPS) {

    int n_idx = 6;
    // Strip-wise data:
    // (z, x, y) order for forward pass (unused)
    Vec<Idx> idx_zxy {
        0, 0, 0,
        1, 0, 1,
        2, 0, 2,
        3, 1, 0,
        4, 1, 1,
        5, 1, 2
    };
    // (x, z, y) also lexsorted
    Vec<Idx> idx_xzy {
        0, 0, 0,
        0, 1, 1,
        0, 2, 2,
        1, 3, 0,
        1, 4, 1,
        1, 5, 2
    };
    // (y, z, x) lexsort
    Vec<Idx> idx_yzx {
        0, 0, 0,
        0, 3, 1,
        1, 1, 0,
        1, 4, 1,
        2, 2, 0,
        2, 5, 1,
    };

    // Transpose and repeat indices with offsets
    Vec<float> coef_xzy = {1., 2., 3., 4., 5., 6.};
    Vec<float> coef_yzx = {1., 4., 2., 5., 3., 6.};

    printf("indices before repeats and offsets:\n");
    vec::showMatrix(std::cout, idx_xzy.data(), 3, n_idx);
    vec::showMatrix(std::cout, idx_yzx.data(), 3, n_idx);

    idx_xzy = transpose_trailing_axes(idx_xzy, n_idx, 3);
    idx_yzx = transpose_trailing_axes(idx_yzx, n_idx, 3);

    idx_xzy = idx_xzy.tile_trailing(num_strips, n_idx);
    idx_yzx = idx_yzx.tile_trailing(num_strips, n_idx);

    idx_xzy = idx_xzy + Vec<Idx>::concat({
        vec::arange<Idx>(num_strips).repeat(n_idx) * Idx(2),
        vec::arange<Idx>(num_strips).repeat(n_idx) * Idx(6),
        vec::arange<Idx>(num_strips).repeat(n_idx) * Idx(3)
    });
    idx_yzx = idx_yzx + Vec<Idx>::concat({
        vec::arange<Idx>(num_strips).repeat(n_idx) * Idx(3),
        vec::arange<Idx>(num_strips).repeat(n_idx) * Idx(6),
        vec::arange<Idx>(num_strips).repeat(n_idx) * Idx(2)
    });

    // Pack each coefficient set into Coef<Idx,float> structs
    int nnz = n_idx * num_strips;
    Vec<float> val_xzy = coef_xzy.tile(num_strips);
    Vec<float> val_yzx = coef_yzx.tile(num_strips);

    std::vector<Coef<Idx, float>> packed_xzy = packCoefs<Idx>(idx_xzy, val_xzy, nnz);
    std::vector<Coef<Idx, float>> packed_yzx = packCoefs<Idx>(idx_yzx, val_yzx, nnz);

    // Concatenate dx and dy coefficient sets
    std::vector<Coef<Idx, float>> coef_packed;
    coef_packed.reserve(2 * nnz);
    coef_packed.insert(coef_packed.end(), packed_xzy.begin(), packed_xzy.end());
    coef_packed.insert(coef_packed.end(), packed_yzx.begin(), packed_yzx.end());

    // Input operands
    Vec<float> x_ = {1., 1.};
    Vec<float> y_ = {1., 1., 1.};
    Vec<float> dz_ = {1., 1., 1., 1., 1., 1.};

    // Output buffers
    Vec<float> dx_ = {0., 0.};
    Vec<float> dy_ = {0., 0., 0.};

    // Repeat index patterns across strips
    int cx = p.channels_x,
        cy = p.channels_y,
        cz = p.channels_out();

    // Multiply y and result channel axis by vec::arange(NUM_CHANNELS)
    Vec<float> scale_y = (vec::arange<float>(cy) + 1)
        .repeat(num_strips * y_.size())
        .tile(p.num_rows);

    // Repeat other operands
    Args<Idx> args = {
        .coef = coef_packed,
        .x = x_.tile(p.num_rows * cx * num_strips),
        .y = y_.tile(p.num_rows * cy * num_strips) * scale_y,
        .dz = dz_.tile(p.num_rows * cz * num_strips),
        .dx = dx_.tile(p.num_rows * cx * num_strips),
        .dy = dy_.tile(p.num_rows * cy * num_strips),
    };

    // Transpose depending on layout
    if (p.layout == Layout::TRAILING_CHANNELS) {
        printf("Layout::TRAILING_CHANNELS\n");
        args = Args<Idx> {
            .coef = args.coef,
            .x = transpose_trailing_axes(args.x, p.channels_x, p.num_x),
            .y = transpose_trailing_axes(args.y, p.channels_y, p.num_y),
            .dz = transpose_trailing_axes(args.dz, cz, p.num_out),
            .dx = args.dx,
            .dy = args.dy
        };
    }
    ASSERT_EQUAL(p.num_x * p.num_rows * p.channels_x, (int)args.x.size(), "x")
    ASSERT_EQUAL(p.num_x * p.num_rows * p.channels_x, (int)args.dx.size(), "dx")
    ASSERT_EQUAL(p.num_y * p.num_rows * p.channels_y, (int)args.y.size(), "y")
    ASSERT_EQUAL(p.num_y * p.num_rows * p.channels_y, (int)args.dy.size(), "dy")
    ASSERT_EQUAL(p.num_out * p.num_rows * p.channels_out(), (int)args.dz.size(), "dz")
    ASSERT_EQUAL(2 * p.num_idx, (int)args.coef.size(), "coef")
    return args;
}


void showOutput (float* data, int channels, int features, Params p) {

    if (p.layout == Layout::LEADING_CHANNELS)
        vec::showMatrix(std::cout, data, channels, features);

    else if (p.layout == Layout::TRAILING_CHANNELS)
        vec::showMatrix(std::cout, data, channels, features);
}


void checkOutput (float* out, float* res, int channels_z, Params p) {

    int num_rows = p.num_rows,
        num_out = p.num_out;

	bool success = true;

    for (int k = 0; k < num_rows * num_out * channels_z; k++) {
		int col_k = k % num_out;
		success &= (out[k] == res[k]);
		if (LOG) {
            printf((col_k < num_out - 1 ? "%.1f " : "%.1f\n     "), out[k]);
        }
	}

    if (DEBUG >= 2 or not success)  {
        // printf("expect:\n"); showOutput(res, channels_z, p);
        // printf("result:\n"); showOutput(out, channels_z, p);
    }

    printf(success ? "✅\n" : "❌\n");
}



template <typename Idx>
int test_tensor_product_bwd (Params p, int num_strips=NUM_STRIPS) {

    #define H2D cudaMemcpyHostToDevice
    #define D2H cudaMemcpyDeviceToHost

    using namespace e3j::tensor_product;

    using e3j::tensor_product::trailing_channels::launch_bwd;

    printf("=== Test tensor_product_bwd ===================================\n");

    std::map<Mode, char const*> modes = {
        {Mode::INNER, "INNER"},
        {Mode::OUTER, "OUTER"},
        {Mode::MAP, "MAP"},
    };
    std::cout << "Mode::" << modes[p.mode] << "\n";
    printf("Idx size: %zu bytes\n", sizeof(Idx));

    using CoefT = Coef<Idx, float>;

    Args<Idx> args = prepareHostArgs<Idx>(p, num_strips);

	printf("Moving inputs to device...\n");

    CoefT *coef_d;
    float *x_d, *y_d, *dz_d, *dx_d, *dy_d;

    int cz = p.channels_out();

    size_t size_coef = sizeof(CoefT) * 2 * p.num_idx,
           size_x = sizeof(float) * p.num_rows * p.num_x * p.channels_x,
           size_y = sizeof(float) * p.num_rows * p.num_y * p.channels_y,
           size_z = sizeof(float) * p.num_rows * p.num_out * cz;

	cudaMalloc((void**)&coef_d, size_coef);
	cudaMalloc((void**)&x_d, size_x);
	cudaMalloc((void**)&y_d, size_y);
	cudaMalloc((void**)&dz_d, size_z);
	cudaMalloc((void**)&dx_d, size_x);
	cudaMalloc((void**)&dy_d, size_y);

	cudaMemcpy(coef_d, args.coef.data(), size_coef, H2D);
	cudaMemcpy(x_d, args.x.data(), size_x, H2D);
	cudaMemcpy(y_d, args.y.data(), size_y, H2D);
	cudaMemcpy(dz_d, args.dz.data(), size_z, H2D);
	cudaMemcpy(dx_d, args.dx.data(), size_x, H2D);
	cudaMemcpy(dy_d, args.dy.data(), size_y, H2D);

    cudaDeviceSynchronize();

    printf("Calling backward tensor product kernel...\n");

    launch_bwd<Idx, float>(
        coef_d,
        x_d, y_d, dz_d,
        dx_d, dy_d,
        p, cudaStream_t(0), DEBUG
    );

    cudaDeviceSynchronize();

    Vec<float> result_dx (args.dx.size());
    Vec<float> result_dy (args.dy.size());
    cudaMemcpy(result_dx.data(), dx_d, size_x, D2H);
    cudaMemcpy(result_dy.data(), dy_d, size_y, D2H);

    cudaDeviceSynchronize();

    // Infer backward parameters on dx and dy
    ParamsBwd p_bwd = GetParamsBwd(p);
    Params p_dx = p_bwd.lhs,
           p_dy = p_bwd.rhs;

    printf("Calling the two equivalent tensor product kernels...\n");

    // Flush outputs
	cudaMemcpy(dx_d, args.dx.data(), size_x, H2D);
	cudaMemcpy(dy_d, args.dy.data(), size_y, H2D);

    // Split packed coefficients: first p.num_idx for dx, next for dy
    CoefT *coef_dx = coef_d,
          *coef_dy = coef_d + p.num_idx;

    // Call two TP kernels.
    cudaDeviceSynchronize();

    launch<Idx, float>(
        coef_dx,
        dz_d, y_d, dx_d,
        p_dx, cudaStream_t(0), DEBUG
    );
    launch<Idx, float>(
        coef_dy,
        dz_d, x_d, dy_d,
        p_dy, cudaStream_t(0), DEBUG
    );

    // Collect expected results
    cudaDeviceSynchronize();

    Vec<float> expect_dx (args.dx.size());
    Vec<float> expect_dy (args.dy.size());

    cudaMemcpy(expect_dx.data(), dx_d, size_x, D2H);
    cudaMemcpy(expect_dy.data(), dy_d, size_y, D2H);
    cudaDeviceSynchronize();

    bool dx_match = true, dy_match = true;
    for (int i = 0; i < (int)expect_dx.size(); i++)
        dx_match &= (result_dx[i] == expect_dx[i]);
    for (int i = 0; i < (int)expect_dy.size(); i++)
        dy_match &= (result_dy[i] == expect_dy[i]);

    printf("dx: %s\n", dx_match ? "✅" : "❌");
    printf("dy: %s\n", dy_match ? "✅" : "❌");

    if (!dx_match) {
        printf("Expect dx:\n");
        showOutput(expect_dx.data(), 3 * p.num_x, p.channels_x, p);
        printf("Result dx:\n");
        showOutput(result_dx.data(), 3 * p.num_x, p.channels_x, p_dx);
    }
    if (!dy_match) {
        printf("Expect dy:\n");
        showOutput(expect_dy.data(), 3 * p.num_y, p.channels_y, p);
        printf("Result dy:\n");
        showOutput(result_dy.data(), 3 * p.num_y, p.channels_y, p_dy);
    }

    cudaFree(coef_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(dz_d);
    cudaFree(dx_d);
    cudaFree(dy_d);
    return 0;
}


Params GetParams(Mode mode, int channels, int num_strips=NUM_STRIPS) {
    int num_idx = 6 * num_strips;
    int num_x = 2 * num_strips;
    int num_y = 3 * num_strips;
    int num_out = 6 * num_strips;
    int channels_x = (mode == Mode::OUTER) ? 1 : channels;
    int channels_y = channels;
    return Params {
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/channels_x,
        /*channels_y*/channels_y,
        /*mode*/mode,
        /*unroll_x*/1,
        /*unroll_y*/1,
        /*unroll_z*/1,
        /*layout*/Layout::TRAILING_CHANNELS
    };
}

// OUTER with channels on x, broadcast on y — the message passing layout.
// Backward modes: {OUTER, INNER} (vs {INNER, OUTER} from GetParams).
Params GetParamsOuterX(int channels, int num_strips=NUM_STRIPS) {
    int num_idx = 6 * num_strips;
    int num_x = 2 * num_strips;
    int num_y = 3 * num_strips;
    int num_out = 6 * num_strips;
    return Params {
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/channels,
        /*channels_y*/1,
        /*mode*/Mode::OUTER,
        /*unroll_x*/1,
        /*unroll_y*/1,
        /*unroll_z*/1,
        /*layout*/Layout::TRAILING_CHANNELS
    };
}

int main() {

    // --- int32_t, 32 channels (N=1) ---
    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::OUTER, 32));
    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::INNER, 32));
    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::MAP,   32));

    // --- int32_t, 64 channels (N=2) ---
    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::OUTER, 64));
    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::MAP,   64));

    // --- int32_t, 512 channels (N=4, multi-stride) ---
    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::OUTER, 512));
    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::INNER, 512));
    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::MAP,   512));

    // --- uint8_t, 32 channels (N=1) ---
    test_tensor_product_bwd<std::uint8_t>(GetParams(Mode::OUTER, 32));
    test_tensor_product_bwd<std::uint8_t>(GetParams(Mode::INNER, 32));
    test_tensor_product_bwd<std::uint8_t>(GetParams(Mode::MAP,   32));

    // --- uint8_t, 64 channels (N=2) ---
    test_tensor_product_bwd<std::uint8_t>(GetParams(Mode::OUTER, 64));
    test_tensor_product_bwd<std::uint8_t>(GetParams(Mode::MAP,   64));

    // --- uint8_t, 512 channels (N=4, multi-stride) ---
    test_tensor_product_bwd<std::uint8_t>(GetParams(Mode::OUTER, 512));
    test_tensor_product_bwd<std::uint8_t>(GetParams(Mode::MAP,   512));

    // --- OUTER with channels on x (message passing layout) ---
    // Backward modes: {OUTER, INNER} — not covered by GetParams.
    test_tensor_product_bwd<std::int32_t>(GetParamsOuterX(32));
    test_tensor_product_bwd<std::int32_t>(GetParamsOuterX(64));
    test_tensor_product_bwd<std::int32_t>(GetParamsOuterX(512));
    test_tensor_product_bwd<std::uint8_t>(GetParamsOuterX(32));
    test_tensor_product_bwd<std::uint8_t>(GetParamsOuterX(64));
    test_tensor_product_bwd<std::uint8_t>(GetParamsOuterX(512));

    // --- Large num_strips: triggers SMEM opt-in and channel striding ---
    int large = 40;

    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::OUTER, 32, large), large);
    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::OUTER, 64, large), large);
    test_tensor_product_bwd<std::int32_t>(GetParams(Mode::MAP,   64, large), large);
    test_tensor_product_bwd<std::uint8_t>(GetParams(Mode::OUTER, 64, large), large);

    test_tensor_product_bwd<std::int32_t>(GetParamsOuterX(32, large), large);
    test_tensor_product_bwd<std::int32_t>(GetParamsOuterX(64, large), large);
    test_tensor_product_bwd<std::uint8_t>(GetParamsOuterX(64, large), large);
}
