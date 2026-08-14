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

#ifndef _E3J_DISPATCH_MACROS_H_
#define _E3J_DISPATCH_MACROS_H_

#include <cstdint>
#include <iostream>
#include <cuda_fp16.h>  // __half (usable from host and device TUs)

// X-Macros helping declare and dispatch over template parameters.
//
// To execute a macro over the collection of supported <Idx, Val>
// dtype pairs supported (e.g. to declare template instances for
// the compiler),
//
// - define the macro as `FOR_EACH_DTYPE_PAIR(IDX, VAL)`,
// - call `__FOR_EACH_DTYPE_PAIR` from this header to loop over pairs,
// - undefine `FOR_EACH_DTYPE_PAIR` afterwards.
//
// To dispatch a runtime value into possible compile-time constants as
// a macro parameter (e.g. to call a dedicated template instance from
// runtime arguments),
//
//  - define the macro `ÐISPATCH_MODE(MODE)`,
//  - call `__DISPATCH_MODE(mode)` on the runtime `mode` variable,
//  - undefine `FOR_EACH_MODE`.
//
/*  Example:

        template<Mode mode>
        void launch(const float* x, float* y){
            kernel<Mode><<<1024, 128>>>(x, y);
        }

        void launch(const float* x, float* y, Mode mode) {

            #define DISPATCH_MODE(MODE) \
            launch<MODE>(x, y);

            __DISPATCH_MODE(mode)
            #undef DISPATCH_MODE
        }
*/


// TODO: Buffering function pointers could be more intuitive for
//       the dispatching.


// Supported (Idx, Val) dtype pairs, as a full cross product: the two dtypes are
// chosen independently.
//
// - Idx: int32 (JAX default int dtype) or uint8 (2x memory saving per coef).
// - Val: float, double or __half. The int32 value path is not supported.
#define __FOR_EACH_DTYPE_PAIR                 \
    FOR_EACH_DTYPE_PAIR(std::int32_t, float)  \
    FOR_EACH_DTYPE_PAIR(std::uint8_t, float)  \
    FOR_EACH_DTYPE_PAIR(std::int32_t, double) \
    FOR_EACH_DTYPE_PAIR(std::uint8_t, double) \
    FOR_EACH_DTYPE_PAIR(std::int32_t, __half) \
    FOR_EACH_DTYPE_PAIR(std::uint8_t, __half)


// Runtime dispatch over supported (Idx, Val) dtype pairs.
//
// Idx in {S32, U8} times Val in {F32, F64, F16}, see __FOR_EACH_DTYPE_PAIR.
//
// Usage:
//  - define the macro `DISPATCH_DTYPE_PAIR(IDX, VAL)`,
//  - define the macro `DISPATCH_DTYPE_PAIR_ERROR(IDX_T, VAL_T)`,
//  - call `__DISPATCH_DTYPE_PAIR(idx_dtype, val_dtype)`,
//  - undefine both macros afterwards.
//
// Requires `xla::DataType` to be in scope (XLA FFI).
#define __DISPATCH_DTYPE_PAIR(IDX_T, VAL_T)                                 \
    if (IDX_T == xla::DataType::U8 and VAL_T == xla::DataType::F32) {      \
        DISPATCH_DTYPE_PAIR(std::uint8_t, float)                            \
    }                                                                       \
    else if (IDX_T == xla::DataType::S32                                    \
             and VAL_T == xla::DataType::F32) {                             \
        DISPATCH_DTYPE_PAIR(std::int32_t, float)                            \
    }                                                                       \
    else if (IDX_T == xla::DataType::S32                                    \
             and VAL_T == xla::DataType::F64) {                             \
        DISPATCH_DTYPE_PAIR(std::int32_t, double)                           \
    }                                                                       \
    else if (IDX_T == xla::DataType::U8                                     \
             and VAL_T == xla::DataType::F64) {                             \
        DISPATCH_DTYPE_PAIR(std::uint8_t, double)                           \
    }                                                                       \
    else if (IDX_T == xla::DataType::S32                                    \
             and VAL_T == xla::DataType::F16) {                             \
        DISPATCH_DTYPE_PAIR(std::int32_t, __half)                           \
    }                                                                       \
    else if (IDX_T == xla::DataType::U8                                     \
             and VAL_T == xla::DataType::F16) {                             \
        DISPATCH_DTYPE_PAIR(std::uint8_t, __half)                           \
    }                                                                       \
    else {                                                                  \
        DISPATCH_DTYPE_PAIR_ERROR(IDX_T, VAL_T)                             \
    }


#define __DISPATCH_MODE(MODE)           \
switch(MODE) {                          \
    case Mode::OUTER:                   \
        DISPATCH_MODE(Mode::OUTER)      \
        break;                          \
    case Mode::INNER:                   \
        DISPATCH_MODE(Mode::INNER)      \
        break;                          \
    case Mode::MAP:                     \
        DISPATCH_MODE(Mode::MAP)        \
        break;                          \
    default:                            \
        DISPATCH_MODE_ERROR(MODE)       \
}

// TODO: prevent using macro if error case not defined?
//
//  #ifndef DISPATCH_MODE_PAIR_ERROR
//  #error "Define DISPATCH_MODE_PAIR_ERROR"
//  #endif

#define __DISPATCH_MODE_PAIR(MODE_LHS, MODE_RHS)                    \
if (MODE_LHS == Mode::OUTER and MODE_RHS == Mode::OUTER) {          \
    DISPATCH_MODE_PAIR(Mode::OUTER, Mode::OUTER)                    \
}                                                                   \
else if (MODE_LHS == Mode::INNER and MODE_RHS == Mode::OUTER) {     \
    DISPATCH_MODE_PAIR(Mode::INNER, Mode::OUTER)                    \
}                                                                   \
else if (MODE_LHS == Mode::OUTER and MODE_RHS == Mode::INNER) {     \
    DISPATCH_MODE_PAIR(Mode::OUTER, Mode::INNER)                    \
}                                                                   \
else if (MODE_LHS == Mode::MAP and MODE_RHS == Mode::MAP) {         \
    DISPATCH_MODE_PAIR(Mode::MAP, Mode::MAP)                        \
}                                                                   \
else {                                                              \
    DISPATCH_MODE_PAIR_ERROR(MODE_LHS, MODE_RHS)                    \
}


#endif// _E3J_DISPATCH_MACROS_H_
