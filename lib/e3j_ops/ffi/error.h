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

#pragma once

#include <string>
#include <cuda_runtime_api.h>
#include "xla/ffi/api/ffi.h"

namespace e3j
{
// Internal error type meant for `xla::ffi::Error` conversion with `error.to_xla()`.
    class Error
    {
    public:
        enum class Code
        {
            kOk,
            kInternal,
            kInvalidArgument,
            kCudaLaunch,
            kCudaRuntime,
            kUnimplemented,
        };

        Error() = default;

        bool success() const { return code_ == Code::kOk; }
        bool failure() const { return !success(); }

        const std::string &message() const { return message_; }
        Code code() const { return code_; }

        static Error Success() { return Error(); }

        static Error Internal(std::string message)
        {
            return Error(Code::kInternal, std::move(message));
        }

        static Error InvalidArgument(std::string message)
        {
            return Error(Code::kInvalidArgument, std::move(message));
        }

        static Error FromCudaLaunch(cudaError_t cuda_err)
        {
            if (cuda_err == cudaSuccess)
            {
                return Success();
            }
            return Error(Code::kCudaLaunch, cudaGetErrorString(cuda_err));
        }

        static Error FromCudaRuntime(cudaError_t cuda_err)
        {
            if (cuda_err == cudaSuccess)
            {
                return Success();
            }
            return Error(Code::kCudaRuntime, cudaGetErrorString(cuda_err));
        }
        static Error Unimplemented(std::string message)
        {
            return Error(Code::kUnimplemented, std::move(message));
        }

        xla::ffi::Error to_xla() const
        {
            switch (code_)
            {
            case Code::kOk:
                return xla::ffi::Error::Success();
            case Code::kInternal:
                return xla::ffi::Error::Internal("E3J_ERROR[Internal]: " + message_);
            case Code::kInvalidArgument:
                return xla::ffi::Error::InvalidArgument("E3J_ERROR[InvalidArgument]: " + message_);
            case Code::kCudaLaunch:
                return xla::ffi::Error::Internal("E3J_ERROR[CUDA Launch]: " + message_);
            case Code::kCudaRuntime:
                return xla::ffi::Error::Internal("E3J_ERROR[CUDA Runtime]: " + message_);
            case Code::kUnimplemented:
                return xla::ffi::Error::Internal("E3J_ERROR[Unimplemented]: " + message_);
            }
        }

    private:
        Error(Code code, std::string message)
            : code_(code), message_(std::move(message)) {}

        Code code_ = Code::kOk;
        std::string message_;
    };

} // namespace e3j
