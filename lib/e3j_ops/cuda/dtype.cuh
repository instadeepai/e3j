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

#ifndef E3J_CUDA_DTYPE_H_
#define E3J_CUDA_DTYPE_H_

namespace e3j {

// TODO: Deprecate and remove this enum — dtype dispatch is now
//       handled by __DISPATCH_DTYPE_PAIR in dispatch_macros.h.
enum Dtype {U8, U16, S32, S64, F32, F64, C32};

} // namespace e3j

#endif
