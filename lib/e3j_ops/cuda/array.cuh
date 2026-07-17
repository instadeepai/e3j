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

#ifndef _E3J_ARRAY_H_
#define _E3J_ARRAY_H_

#include <cstdint>
#include "cuda/utils.cuh"

namespace e3j {

/****************************************************************
 *  CUDA arrays aware of their trailing 2D shapes.
 *
 *  Device functions can often be expressed in terms of `CuArray2D`
 *  with concise signatures, since it contains the necessary problem
 *  sizes to work on a single batch.
 *
 *  Since the leading batch axis is usually shared, it is passed as
 *  a separate single register.
 ****************************************************************/
template <typename T>
struct CuArray2D {
    T* data;
    unsigned int shape [2];

    __host__ __device__ CuArray2D (T* data, unsigned int n0, unsigned int n1) :
        data(data),
        shape{n0, n1}
    {}

    __device__ unsigned int size() {
        return (unsigned int)(shape[0] * shape[1]);
    }

    __device__ operator T* () {
        return data;
    }

    __device__ operator const T* () const {
        return data;
    }

    __device__ T& operator [] (unsigned int pos) {
        return data[pos];
    }

    // Load a 2D array to shared memory (LDGSTS, pipeline primitives API).
    // Supports striding through source channels when too large to fit in SMEM.
    template<int N=1>
    __device__ void load(CuArray2D<const T> src) {
        if (shape[1] == src.shape[1])
            utils::copy_pipe<N>(
                data, src.data, src.size()
            );
        else
            utils::copy_pipe_strided(
                data, src.data, src.shape[0], shape[1], src.shape[1]
            );
    }

    // Load a 2D slice from a GMEM CuArray2D and random row index.
    //
    // NOTE: The `row * size()` address must be computed in 64 bit to avoid overflows
    //       and segmentation faults on large graphs and feature sizes.
    template<int N=1>
    __device__ void load(CuArray2D<const T> &src, size_t row) {
        src.data += row * src.size();
        load<N>(src);
        src.data -= row * src.size();
    }

    // Store a 2D array to global memory (inline copy with striding support).
    __device__ void store(CuArray2D<T> dst) {
        if (shape[1] == dst.shape[1])
            utils::copy(
                dst.data, data, size()
            );
        else
            utils::copy_strided(
                dst.data, data, shape[0], dst.shape[1], shape[1]
            );
    }

    // Store a 2D slice to a GMEM CuArray2D at a random row index.
    //
    // NOTE: The `row * size()` address must be computed in 64 bit to avoid overflows
    //       and segmentation faults on large graphs and feature sizes.
    __device__ void store(CuArray2D<T> &dst, size_t row) {
        dst.data += row * dst.size();
        store(dst);
        dst.data -= row * dst.size();
    }

};


} // namespace e3j

#endif //_E3J_ARRAY_H_
