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

#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <algorithm>

#include <map>

#include "cuda.h"
#include "cuda_runtime_api.h"

#include "cuda/tensor_product.cuh"

#include "vec.h"

#define NUM_ROWS 4
#define NUM_CHANNELS 32 // FIXME: TRAILING_CHANNELS requires 32-multiple
#define NUM_STRIPS 12

#define UNROLL_Y 1
#define UNROLL_Z 1

#define LOG false
#define DEBUG 1

typedef std::int32_t int32;


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
    Vec<float> out;
    Vec<float> res;
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

    // Strip-wise data
    Vec<Idx> idx_0 {0, 0, 0, 1, 1, 1};
    Vec<Idx> idx_1 {0, 0, 0, 1, 1, 1};
    Vec<Idx> idx_2 {0, 1, 2, 0, 1, 2};

    Vec<float> coef_ = {1., 1., 1., 1., 1., 1.};
    Vec<float> x_ = {1., 2.};
    Vec<float> y_ = {3., 4., 5.};

    Vec<float> out_ = {0., 0.};
    Vec<float> res_ = {12., 24.};

    // Repeat + translate indices
    idx_0 = idx_0.tile(num_strips);
    idx_1 = idx_1.tile(num_strips);
    idx_2 = idx_2.tile(num_strips);
    idx_0 = idx_0 + vec::arange<Idx>(num_strips).repeat(6) * Idx(2);
    idx_1 = idx_1 + vec::arange<Idx>(num_strips).repeat(6) * Idx(2);
    idx_2 = idx_2 + vec::arange<Idx>(num_strips).repeat(6) * Idx(3);

    int cx = p.channels_x,
        cy = p.channels_y,
        cz = p.channels_out();

    // Multiply y and result channel axis by vec::arange(NUM_CHANNELS)
    Vec<float> scale_y = (vec::arange<float>(cy) + 1)
        .repeat(num_strips * y_.size())
        .tile(p.num_rows);

    Vec<float> scale_res {0};
    switch(p.mode) {
        case Mode::OUTER:
            scale_res = (vec::arange<float>(cy) + 1)
                .repeat(num_strips * res_.size())
                .tile(cx * p.num_rows);
            break;
        case Mode::INNER:
            scale_res = Vec<float>({(cy + 1.0f) * (cy) / 2, })
                .repeat(num_strips * res_.size())
                .tile(p.num_rows);
            break;
        case Mode::MAP:
            scale_res = (vec::arange<float>(cx) + 1)
                .repeat(num_strips * res_.size())
                .tile(p.num_rows);
            break;
    }

    // Pack coefficients as passed from JAX
    Vec<Idx> idx_flat = Vec<Idx>::concat({idx_0, idx_1, idx_2});
    Vec<float> val = coef_.tile(num_strips);
    using CoefT = Coef<Idx, float>;
    std::vector<CoefT> coef_packed = packCoefs(idx_flat, val, p.num_idx);

    // Repeat other operands
    Args<Idx> args = {
        .coef = coef_packed,
        .x = x_.tile(p.num_rows * cx * num_strips),
        .y = y_.tile(p.num_rows * cy * num_strips) * scale_y,
        .out = out_.tile(p.num_rows * cz * num_strips),
        .res = res_.tile(p.num_rows * cz * num_strips) * scale_res
    };

    // Transpose depending on layout
    if (p.layout == Layout::TRAILING_CHANNELS) {
        printf("Layout::TRAILING_CHANNELS\n");
        args = Args<Idx> {
            .coef = args.coef,
            .x = transpose_trailing_axes(args.x, p.channels_x, p.num_x),
            .y = transpose_trailing_axes(args.y, p.channels_y, p.num_y),
            .out = transpose_trailing_axes(args.out, cz, p.num_out),
            .res = transpose_trailing_axes(args.res, cz, p.num_out)
        };
    }

    return args;
}


void showOutput (float* data, int channels_z, Params p) {
    if (p.layout == Layout::LEADING_CHANNELS)
        vec::showMatrix(std::cout, data, channels_z, p.num_out);
    else if (p.layout == Layout::TRAILING_CHANNELS and p.mode == Mode::INNER)
        vec::showMatrix(std::cout, data, p.num_rows, p.num_out);
    else if (p.layout == Layout::TRAILING_CHANNELS)
        vec::showMatrix(std::cout, data, p.num_out, channels_z);
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
        printf("expect:\n"); showOutput(res, channels_z, p);
        printf("result:\n"); showOutput(out, channels_z, p);
    }

    printf(success ? "✅\n" : "❌\n");
}


template <typename Idx>
int test_tensor_product (Params p, int num_strips=NUM_STRIPS) {

    printf("=== Test tensor_product =======================================\n");

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
    float *x_d, *y_d, *out_d;

    int cz = p.channels_out();

    size_t size_coef = sizeof(CoefT) * p.num_idx,
           size_x = sizeof(float) * p.num_rows * p.num_x * p.channels_x,
           size_y = sizeof(float) * p.num_rows * p.num_y * p.channels_y,
           size_out = sizeof(float) * p.num_rows * p.num_out * cz;

	cudaMalloc((void**)&coef_d, size_coef);
	cudaMalloc((void**)&x_d, size_x);
	cudaMalloc((void**)&y_d, size_y);
	cudaMalloc((void**)&out_d, size_out);

	cudaMemcpy(coef_d, args.coef.data(), size_coef, cudaMemcpyHostToDevice);
	cudaMemcpy(x_d, args.x.data(), size_x, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, args.y.data(), size_y, cudaMemcpyHostToDevice);
	cudaMemcpy(out_d, args.out.data(), size_out, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

	// Call tensor product kernel
	e3j::tensor_product::launch<Idx, float>(
		coef_d, x_d, y_d, out_d, p, cudaStream_t(0), DEBUG
	);

	cudaDeviceSynchronize();

	printf("Collecting output from device...\n");
	cudaMemcpy(
        args.out.data(), out_d, size_out, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	// Check correctness : 1 2 3 4 1 2 3 4 ...
	printf("Checking output correctness...\n");
	checkOutput(args.out.data(), args.res.data(), cz, p);

	// Cleanup
    cudaFree(coef_d);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(out_d);

	return 0;
}


int main() {

    // Input widths
    int num_idx = 6 * NUM_STRIPS;
    int num_x = 2 * NUM_STRIPS;
    int num_y = 3 * NUM_STRIPS;
    int num_out = 2 * NUM_STRIPS;

    Params p1 = {
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/1,
        /*channels_y*/NUM_CHANNELS,
        /*mode*/Mode::OUTER,
        /*unroll_x*/1,
        /*unroll_y*/UNROLL_Y,
        /*unroll_z*/UNROLL_Z
    };

    test_tensor_product<int32>(p1);

    Params p2 = {
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/NUM_CHANNELS,
        /*channels_y*/NUM_CHANNELS,
        /*mode*/Mode::INNER,
        /*unroll_x*/1,
        /*unroll_y*/1,
        /*unroll_z*/1
    };

    test_tensor_product<int32>(p2);

    Params p3 = {
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/1,
        /*channels_y*/NUM_CHANNELS,
        /*mode*/Mode::OUTER,
        /*unroll_x*/1,
        /*unroll_y*/UNROLL_Y,
        /*unroll_z*/UNROLL_Z,
        /*layout*/Layout::TRAILING_CHANNELS
    };

    test_tensor_product<int32>(p3);

    Params p4 = Params({
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/NUM_CHANNELS,
        /*channels_y*/NUM_CHANNELS,
        /*mode*/Mode::INNER,
        /*unroll_x*/1,
        /*unroll_y*/1,
        /*unroll_z*/1,
        /*layout*/Layout::TRAILING_CHANNELS
    });

    test_tensor_product<int32>(p4);

    int num_strips = 60;
    Params p5 = Params({
        /*num_rows*/NUM_ROWS,
        /*num_idx*/6 * num_strips,
        /*num_x*/2 * num_strips,
        /*num_y*/3 * num_strips,
        /*num_out*/2 * num_strips,
        /*channels_x*/64,
        /*channels_y*/64,
        /*mode*/Mode::MAP,
        /*unroll_x*/1,
        /*unroll_y*/1,
        /*unroll_z*/1,
        /*layout*/Layout::TRAILING_CHANNELS
    });

    test_tensor_product<int32>(p5, num_strips);

    // 512 channels (trailing only) — triggers channel striding
    Params p6 = Params({
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/1,
        /*channels_y*/512,
        /*mode*/Mode::OUTER,
        /*unroll_x*/1,
        /*unroll_y*/1,
        /*unroll_z*/1,
        /*layout*/Layout::TRAILING_CHANNELS
    });

    test_tensor_product<int32>(p6);

    Params p7 = Params({
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/512,
        /*channels_y*/512,
        /*mode*/Mode::INNER,
        /*unroll_x*/1,
        /*unroll_y*/1,
        /*unroll_z*/1,
        /*layout*/Layout::TRAILING_CHANNELS
    });

    test_tensor_product<int32>(p7);

    Params p8 = Params({
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/512,
        /*channels_y*/512,
        /*mode*/Mode::MAP,
        /*unroll_x*/1,
        /*unroll_y*/1,
        /*unroll_z*/1,
        /*layout*/Layout::TRAILING_CHANNELS
    });

    test_tensor_product<int32>(p8);

    // OUTER with broadcast x — N=2 and N=4 (no striding)
    Params p9 = Params({
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/1,
        /*channels_y*/64,
        /*mode*/Mode::OUTER,
        /*unroll_x*/1,
        /*unroll_y*/1,
        /*unroll_z*/1,
        /*layout*/Layout::TRAILING_CHANNELS
    });

    test_tensor_product<int32>(p9);

    Params p10 = Params({
        /*num_rows*/NUM_ROWS,
        /*num_idx*/num_idx,
        /*num_x*/num_x,
        /*num_y*/num_y,
        /*num_out*/num_out,
        /*channels_x*/1,
        /*channels_y*/128,
        /*mode*/Mode::OUTER,
        /*unroll_x*/1,
        /*unroll_y*/1,
        /*unroll_z*/1,
        /*layout*/Layout::TRAILING_CHANNELS
    });

    test_tensor_product<int32>(p10);

    // Narrow index dtype tests (uint16 disabled: sizeof mismatch with numpy)
    test_tensor_product<std::uint8_t>(p1);
    test_tensor_product<std::uint8_t>(p2);
    test_tensor_product<std::uint8_t>(p3);
    test_tensor_product<std::uint8_t>(p4);
    test_tensor_product<std::uint8_t>(p5, num_strips);
    test_tensor_product<std::uint8_t>(p6);
    test_tensor_product<std::uint8_t>(p7);
    test_tensor_product<std::uint8_t>(p8);
    test_tensor_product<std::uint8_t>(p9);
    test_tensor_product<std::uint8_t>(p10);

}
