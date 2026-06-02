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

};


} // namespace e3j

#endif //_E3J_ARRAY_H_
