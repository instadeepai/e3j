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

// XLA custom call boilerplate for `e3j_ops`.
//
// The Python bindings are declared with pybind11 in `e3j_ops.cpp`.
//
// Just wrap `e3j::<op_name>::launch` by first unpacking XLA's
// opaque descriptor to `e3j::<op_name>::Params`, and also
// accessing the `**buffers` explicitly.

#ifndef E3J_FFI_E3J_OPS_H_
#define E3J_FFI_E3J_OPS_H_

#include <iostream>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

// Include local copies of openxla/xla/xla/ffi/api/api headers
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "ffi/error.h"
#include "cuda/dtype.cuh"
#include "cuda/dispatch_macros.h"
#include "cuda/fill.cuh"
#include "cuda/scatter_add.cuh"
#include "cuda/tensor_product.cuh"
#include "cuda/tensor_product_bwd.cuh"
#include "cuda/convolution.cuh"

/* Boilerplate DTYPE macros are now in dispatch_macros.h:
 *
 * - `__FOR_EACH_DTYPE_PAIR`: compile-time loop for template instantiation.
 * - `__DISPATCH_DTYPE_PAIR`: runtime dispatch over (Idx, Val) dtype pairs.
 *
 * Each handler defines a `DISPATCH_DTYPE_PAIR(Idx, Val)` macro with its
 * body, then calls `__DISPATCH_DTYPE_PAIR`.
 *
 * Supported index dtypes:
 * - S32 (int32): JAX's default integer dtype, used as fallback.
 * - U8/U16 (uint8/uint16): narrow index types selected internally
 *   when feature dimensions fit. Unsigned because they have a larger
 *   positive range than their signed counterparts.
 */

namespace e3j_ops {

namespace xla = xla::ffi;

//===----------------------------------------------------------===//
//  scatter_add_1
//===----------------------------------------------------------===//
xla::Error ScatterAdd1Handler(
    cudaStream_t stream,
    int32_t num_out,
    xla::AnyBuffer idx,
    xla::AnyBuffer val,
    xla::Result<xla::AnyBuffer> out
) {
    int32_t num_idx = idx.dimensions().back();
    int32_t num_rows = val.element_count() / num_idx;

    #define DISPATCH_DTYPE_PAIR_ERROR(IDX_T, VAL_T)             \
        return e3j::Error::InvalidArgument(                  \
            "unsupported (IDX, VAL) dtype pair").to_xla();

    #define DISPATCH_DTYPE_PAIR(Idx, Val)                    \
        e3j::scatter_add_1::Params params = {               \
            num_rows, num_idx, num_out,                     \
        };                                                  \
        return e3j::scatter_add_1::launch<Idx, Val>(        \
            idx.typed_data<Idx>(),                          \
            val.typed_data<Val>(),                          \
            out->typed_data<Val>(),                         \
            params, stream                                  \
        ).to_xla();

    __DISPATCH_DTYPE_PAIR(idx.element_type(), val.element_type())
    #undef DISPATCH_DTYPE_PAIR
    #undef DISPATCH_DTYPE_PAIR_ERROR

    return e3j::Error::Success().to_xla();
}

XLA_FFI_DEFINE_HANDLER(
    xla_scatter_add_1,
    ScatterAdd1Handler,
    xla::Ffi::Bind()
        .Ctx<xla::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("num_out")
        .Arg<xla::AnyBuffer>()   // idx
        .Arg<xla::AnyBuffer>()   // val
        .Ret<xla::AnyBuffer>()
);


//===----------------------------------------------------------===//
//  tensor_product
//===----------------------------------------------------------===//
xla::Error TensorProductHandler(
    cudaStream_t stream,
    xla::AnyBuffer coef,
    xla::AnyBuffer x,
    xla::AnyBuffer y,
    xla::Result<xla::AnyBuffer> out,
    int32_t num_out,
    int32_t _mode,
    int32_t _layout,
    int debug = 0
) {

    // Parse mode and layout from int: see `e3j.utils.options`
    // for the python Enums that must match those of
    // `tensor_product.cuh`.
    using e3j::tensor_product::Mode;
    using e3j::tensor_product::Layout;
    Mode mode = Mode(_mode);
    Layout layout = Layout(_layout);

    // Packed coef buffer has shape (num_idx, numel) where
    // numel = sizeof(Coef<idx_t,val_t>) / sizeof(idx_t).
    int32_t num_idx = coef.dimensions().front();

    xla::AnyBuffer::Dimensions dims_x = x.dimensions();
    xla::AnyBuffer::Dimensions dims_y = y.dimensions();
    int32_t num_x, num_y, channels_x, channels_y;
    switch (layout) {
        case Layout::LEADING_CHANNELS:
            num_x = dims_x.back();
            num_y = dims_y.back();
            channels_x = dims_x.size() > 2 ? dims_x[1] : 1;
            channels_y = dims_y.size() > 2 ? dims_y[1] : 1;
            break;
        case Layout::TRAILING_CHANNELS:
            num_x = dims_x[1];
            num_y = dims_y[1];
            channels_x = dims_x.size() > 2 ? dims_x.back() : 1;
            channels_y = dims_y.size() > 2 ? dims_y.back() : 1;
            break;
        default:
            return e3j::Error::InvalidArgument("Invalid tensor product layout.").to_xla();
    }

    // FIXME: support non-32 multiples with TRAILING_CHANNELS
    if (layout == Layout::TRAILING_CHANNELS) {
        // Assert channels are 32-multiple or 1 in Mode::OUTER
        if (
            mode == Mode::OUTER
            and (channels_x % 32 != 0 or channels_x == 1)
            and (channels_y % 32 != 0 or channels_y == 1)
        ) {
            std::string msg =
                "TRAILING_CHANNELS requires 32 multiple as LHS channels "
                "broadcast with 1 RHS channel with mode OUTER. Got ("
                + std::to_string(channels_x) + ", " + std::to_string(channels_y) + ").";
            return e3j::Error::Unimplemented(msg).to_xla();
        }
        // Assert channels are 32-multiples and match in Mode::MAP | Mode::INNER
        else if (
            mode != Mode::OUTER
            and channels_x % 32 != 0 and channels_y != channels_x
        ) {
            std::string msg =
                "TRAILING_CHANNELS requires 32 multiple as LHS channels "
                "and as many RHS channels with modes MAP | INNER. Got ("
                + std::to_string(channels_x) + ", " + std::to_string(channels_y) + ").";
            return e3j::Error::Unimplemented(msg).to_xla();
        }
    }

    int32_t num_rows = x.element_count() / (num_x * channels_x);

    #define DISPATCH_DTYPE_PAIR_ERROR(Idx, Val)                 \
        return e3j::Error::InvalidArgument(                     \
            "unsupported (IDX, VAL) dtype pair").to_xla();

    #define DISPATCH_DTYPE_PAIR(Idx, Val)                       \
        using Coef = e3j::tensor_product::Coef<Idx, Val>;       \
        const Coef *coef_ptr = reinterpret_cast<const Coef*>(   \
            coef.typed_data<Idx>()                              \
        );                                                      \
                                                                \
        e3j::tensor_product::Params params = {                  \
            .num_rows = num_rows,                               \
            .num_idx = num_idx,                                 \
            .num_x = num_x,                                     \
            .num_y = num_y,                                     \
            .num_out = num_out,                                 \
            .channels_x = channels_x,                           \
            .channels_y = channels_y,                           \
            .mode = mode,                                       \
            .layout = layout                                    \
        };                                                      \
        return e3j::tensor_product::launch<Idx, Val>(           \
            coef_ptr,                                           \
            x.typed_data<Val>(),                                \
            y.typed_data<Val>(),                                \
            out->typed_data<Val>(),                             \
            params, stream, debug                               \
        ).to_xla();

    // The packed buffer is typed as idx_t but holds Coef<idx_t, val_t> structs.
    // reinterpret_cast recovers the struct pointer.
    // NOTE: val_t is inferred from x.element_type(), assuming that
    //       x.dtype matches the val_dtype used when packing coefficients.
    __DISPATCH_DTYPE_PAIR(coef.element_type(), x.element_type())

    #undef DISPATCH_DTYPE_PAIR
    #undef DISPATCH_DTYPE_PAIR_ERROR

    return e3j::Error::Success().to_xla();
}

XLA_FFI_DEFINE_HANDLER(
    xla_tensor_product,
    TensorProductHandler,
    xla::Ffi::Bind()
        .Ctx<xla::PlatformStream<cudaStream_t>>()
        .Arg<xla::AnyBuffer>()   // coef (packed Coef<Idx,Val> as idx_t vector)
        .Arg<xla::AnyBuffer>()   // x
        .Arg<xla::AnyBuffer>()   // y
        .Ret<xla::AnyBuffer>()
        .Attr<int32_t>("num_out")
        .Attr<int32_t>("mode")
        .Attr<int32_t>("layout")
        .Attr<int32_t>("debug")
);


//===----------------------------------------------------------===//
//  tensor_product_bwd
//===----------------------------------------------------------===//
xla::Error TensorProductBwdHandler(
    cudaStream_t stream,
    xla::AnyBuffer coef,
    xla::AnyBuffer x,
    xla::AnyBuffer y,
    xla::AnyBuffer dz,
    xla::Result<xla::AnyBuffer> dx,
    xla::Result<xla::AnyBuffer> dy,
    int32_t _mode_fwd,
    int32_t _layout,
    int debug = 0
) {

    // Parse mode and layout from int: see `e3j.utils.options`
    // for the python Enums that must match those of
    // `tensor_product.cuh`.
    using e3j::tensor_product::Mode;
    using e3j::tensor_product::Layout;
    Mode mode_fwd = Mode(_mode_fwd);
    Layout layout = Layout(_layout);

    // Coefficients for dx + dy should be concatenated from Python.
    // We rely on `element_count()` instead of Dimensions
    // for robustness, in case stack/concat of coef changes.
    int32_t numel = coef.dimensions().back();
    int32_t num_idx = coef.element_count() / (2 * numel);

    xla::AnyBuffer::Dimensions dims_x = x.dimensions();
    xla::AnyBuffer::Dimensions dims_y = y.dimensions();
    xla::AnyBuffer::Dimensions dims_z = dz.dimensions();

    int32_t num_x, num_y, num_z, channels_x, channels_y;
    switch (layout) {
        case Layout::LEADING_CHANNELS:
            num_x = dims_x.back();
            num_y = dims_y.back();
            num_z = dims_z.back();
            channels_x = dims_x.size() > 2 ? dims_x[1] : 1;
            channels_y = dims_y.size() > 2 ? dims_y[1] : 1;
            break;
        case Layout::TRAILING_CHANNELS:
            num_x = dims_x[1];
            num_y = dims_y[1];
            num_z = dims_z[1];
            channels_x = dims_x.size() > 2 ? dims_x.back() : 1;
            channels_y = dims_y.size() > 2 ? dims_y.back() : 1;
            break;
        default: exit(1);
    }

    int32_t num_rows = x.element_count() / (num_x * channels_x);

    e3j::tensor_product::Params params_fwd = {
        .num_rows = num_rows,
        .num_idx = num_idx,
        .num_x = num_x,
        .num_y = num_y,
        .num_out = num_z,
        .channels_x = channels_x,
        .channels_y = channels_y,
        .mode = mode_fwd,
        .layout = layout
    };

    using e3j::tensor_product::trailing_channels::launch_bwd;

    #define DISPATCH_DTYPE_PAIR_ERROR(IDX_T, VAL_T)          \
        return e3j::Error::InvalidArgument(                  \
            "unsupported (IDX, VAL) dtype pair").to_xla();

    #define DISPATCH_DTYPE_PAIR(Idx, Val)                       \
        using Coef = e3j::tensor_product::Coef<Idx, Val>;       \
        const Coef *coef_ptr = reinterpret_cast<const Coef*>(   \
            coef.typed_data<Idx>()                              \
        );                                                      \
                                                                \
        return launch_bwd<Idx, Val>(                            \
            coef_ptr,                                           \
            x.typed_data<Val>(),                                \
            y.typed_data<Val>(),                                \
            dz.typed_data<Val>(),                               \
            dx->typed_data<Val>(),                              \
            dy->typed_data<Val>(),                              \
            params_fwd, stream, debug                           \
        ).to_xla();

    __DISPATCH_DTYPE_PAIR(coef.element_type(), x.element_type())
    #undef DISPATCH_DTYPE_PAIR
    #undef DISPATCH_DTYPE_PAIR_ERROR

    return xla::Error::Success();
};


XLA_FFI_DEFINE_HANDLER(
    xla_tensor_product_bwd,
    TensorProductBwdHandler,
    xla::Ffi::Bind()
        .Ctx<xla::PlatformStream<cudaStream_t>>()
        .Arg<xla::AnyBuffer>()   // coef
        .Arg<xla::AnyBuffer>()   // x
        .Arg<xla::AnyBuffer>()   // y
        .Arg<xla::AnyBuffer>()   // z
        .Ret<xla::AnyBuffer>()   // dx
        .Ret<xla::AnyBuffer>()   // dy
        .Attr<int32_t>("mode")
        .Attr<int32_t>("layout")
        .Attr<int32_t>("debug")
);


//===----------------------------------------------------------===//
//  convolution
//===----------------------------------------------------------===//
xla::Error ConvolutionHandler(
    cudaStream_t stream,
    xla::AnyBuffer coef,
    xla::AnyBuffer x,
    xla::AnyBuffer y,
    xla::AnyBuffer r,
    xla::AnyBuffer irrep_out,
    xla::AnyBuffer sender,
    xla::AnyBuffer receiver_ptr,
    xla::Result<xla::AnyBuffer> out,
    int32_t num_nodes,
    int debug = 0
) {

    xla::AnyBuffer::Dimensions dims_x = x.dimensions();
    xla::AnyBuffer::Dimensions dims_y = y.dimensions();

    int32_t num_coef = coef.dimensions().front();
    int32_t num_x = dims_x[1];
    int32_t channels_x = dims_x.size() > 2 ? dims_x.back() : 1;
    int32_t num_y = dims_y[1];
    int32_t num_out = out->dimensions()[1];
    int32_t num_scalars = r.dimensions()[1];

    // Assert LHS channels are 32-multiple
    bool supported = (channels_x % 32 == 0);
    if (not supported) {
        std::string msg =
            "Convolution requires 32 multiple as LHS channels for now. Got "
            + std::to_string(channels_x) + ".\n";
        return e3j::Error::Unimplemented(msg).to_xla();
    }

    #define DISPATCH_DTYPE_PAIR_ERROR(IDX_T, VAL_T)             \
        return e3j::Error::InvalidArgument(                     \
            "unsupported (IDX, VAL) dtype pair").to_xla();

    #define DISPATCH_DTYPE_PAIR(Idx, Val)                       \
        using Coef = e3j::tensor_product::Coef<Idx, Val>;       \
        const Coef *coef_ptr = reinterpret_cast<const Coef*>(   \
            coef.typed_data<Idx>()                              \
        );                                                      \
                                                                \
        e3j::convolution::Params params = {                     \
            .num_nodes = num_nodes,                             \
            .num_coef = num_coef,                               \
            .num_x = num_x,                                     \
            .num_y = num_y,                                     \
            .num_out = num_out,                                 \
            .num_scalars = num_scalars,                         \
            .channels_x = channels_x,                           \
        };                                                      \
                                                                \
        e3j::convolution::AdjacencyCSR adj = {                  \
            .sender = const_cast<int32_t*>(                     \
                sender.typed_data<int32_t>()                    \
            ),                                                  \
            .receiver_ptr = const_cast<int32_t*>(               \
                receiver_ptr.typed_data<int32_t>()              \
            ),                                                  \
        };                                                      \
                                                                \
        return e3j::convolution::launch<Idx, Val>(              \
            coef_ptr,                                           \
            x.typed_data<Val>(),                                \
            y.typed_data<Val>(),                                \
            r.typed_data<Val>(),                                \
            irrep_out.typed_data<Idx>(),                        \
            adj,                                                \
            out->typed_data<Val>(),                             \
            params, stream, debug                               \
        ).to_xla();

    __DISPATCH_DTYPE_PAIR(coef.element_type(), x.element_type())
    #undef DISPATCH_DTYPE_PAIR
    #undef DISPATCH_DTYPE_PAIR_ERROR

    return e3j::Error::Success().to_xla();
}

XLA_FFI_DEFINE_HANDLER(
    xla_convolution,
    ConvolutionHandler,
    xla::Ffi::Bind()
        .Ctx<xla::PlatformStream<cudaStream_t>>()
        .Arg<xla::AnyBuffer>()   // coef (packed Coef<Idx,Val> as idx_t vector)
        .Arg<xla::AnyBuffer>()   // x (node features)
        .Arg<xla::AnyBuffer>()   // y (edge embeddings)
        .Arg<xla::AnyBuffer>()   // r (radial scalars)
        .Arg<xla::AnyBuffer>()   // irrep_out (output irrep index map)
        .Arg<xla::AnyBuffer>()   // sender (CSR sender indices)
        .Arg<xla::AnyBuffer>()   // receiver_ptr (CSR receiver pointers)
        .Ret<xla::AnyBuffer>()   // out (output node features)
        .Attr<int32_t>("num_nodes")
        .Attr<int32_t>("debug")
);


} // namespace e3j_ops

#endif // E3J_FFI_E3J_OPS_H_
