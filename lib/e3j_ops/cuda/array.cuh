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
};


} // namespace e3j

#endif //_E3J_ARRAY_H_
