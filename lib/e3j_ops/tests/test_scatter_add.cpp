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
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime_api.h"

#include "cuda/scatter_add.cuh"


const int NUM_IDX = 72;
const int NUM_OUT = 30;
const int BATCH_SIZE = 5;

bool LOG = true;

typedef std::int32_t int32;

void prepareHostArgs (int32* idx, float* val, float* out) {
	printf("idx: ");
	for (int i=0; i < NUM_IDX; i++) {
		// idx = arange(N/10).repeat(idx % 4)
		// 0 1 1 2 2 2 3 3 3 3 | 4 5 5 ...
		int32 idx_i = 4 * (i / 10);
		int rest_i = i % 10;
		if (rest_i >= 1) {idx_i += 1;}
		if (rest_i >= 3) {idx_i += 1;}
		if (rest_i >= 6) {idx_i += 1;}
		idx[i] = idx_i;
		printf("%d ", idx_i);
	}
	printf("\n");
	for (int j = 0; j < BATCH_SIZE * NUM_IDX; j++) {
		val[j] = 1.;
	}
	for (int k = 0; k < BATCH_SIZE * NUM_OUT; k++) {
		out[k] = 0.;
	}
}

void checkOutput (float* out) {
	bool success = true;
	if (LOG) printf("out: ");

	for (int k = 0; k < BATCH_SIZE * NUM_OUT; k++) {
		int col_k = k % NUM_OUT;
		success &= (
			out[k] == 1 + (col_k % 4)
			or (col_k == 13 & out[k] == 1)
			or (col_k >= 14)
		);
		if (LOG) printf((col_k < NUM_OUT - 1 ? "%.0f " : "%.0f\n     "), out[k]);
	}
	printf(success ? "✅\n" : "❌\n");
}


int main() {

    printf("=== Test scatter_add_1 ========================================\n");
	printf("Preparing arguments on host...\n");

	int32* idx = new int32[NUM_IDX];
	float* val = new float[BATCH_SIZE * NUM_IDX];
	float* out = new float[BATCH_SIZE * NUM_OUT];
	prepareHostArgs(idx, val, out);

	printf("Moving inputs to device...\n");

	int32 *idx_d;
	float *val_d, *out_d;

	cudaMalloc((void**)&idx_d, sizeof(int32) * NUM_IDX);
	cudaMalloc((void**)&val_d, sizeof(float) * BATCH_SIZE * NUM_IDX);
	cudaMalloc((void**)&out_d, sizeof(float) * BATCH_SIZE * NUM_OUT);

	cudaMemcpy(idx_d, idx, sizeof(float) * NUM_IDX, cudaMemcpyHostToDevice);
	cudaMemcpy(val_d, val, sizeof(float) * BATCH_SIZE * NUM_IDX, cudaMemcpyHostToDevice);
	cudaMemcpy(out_d, out, sizeof(float) * BATCH_SIZE * NUM_OUT, cudaMemcpyHostToDevice);


	// Call scatter-add kernel
	e3j::scatter_add_1::Params params = {
		BATCH_SIZE, NUM_IDX, NUM_OUT
	};
	e3j::scatter_add_1::launch<int32, float>(
		idx_d, val_d, out_d, params, cudaStream_t(0)
	);

	// Cleanup
	cudaDeviceSynchronize();
	cudaFree(idx_d);
	cudaFree(val_d);

	printf("Collecting output from device...\n");
	cudaMemcpy(out, out_d, sizeof(float) * BATCH_SIZE * NUM_OUT, cudaMemcpyDeviceToHost);

	printf("Checking output correctness...\n");
	// Check correctness : 1 2 3 4 1 2 3 4 ...
	checkOutput(out);

	cudaFree(out);
	free(idx);
	free(val);
	free(out);

	return 0;
}
