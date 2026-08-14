# ⚙️ e3j-ops

This package contains the CUDA/C++ source for [e3j].
The Python bindings to XLA handlers are bundled into the `e3j_ops` shared object, which
the main [e3j] package wraps in custom JAX primitives via the [ffi_call][xla/ffi] API.

> **Note:** The `e3j_ops` ABI should not be considered stable for now, but
> considered private within the `e3j` Python API.


### Building from source

The CMake build recipes are defined in [CMakeLists.txt](CMakeLists.txt).
The `Makefile` of [e3j] also defines fine-graind recipes to build and test individual CUDA/C++ objects.


Project structure
-----------------

The source is organized as follows:

- [cuda](cuda) : CUDA kernel implementations
- [ffi](ffi) : XLA-FFI handlers declarations and pybind11 module definition
- [xla](xla) : vendored [XLA FFI headers](https://github.com/openxla/xla/tree/main/xla/ffi/api)
- [tests](tests) : C++ kernel tests

### CUDA/C++ source

Each operation is defined in its own namespace following a common interface
(see e.g. [tensor_product.cuh](cuda/tensor_product.cuh) for detailed signatures):

- a namespace `e3j::op_name` within the enclosing `e3j` namespace,
- a struct `e3j::op_name::Params` for passing hyperparameters,
- a `__global__` CUDA function `e3j::op_name::kernel()`,
- a `__host__` launcher `e3j::op_name::launch()`

Although object-oriented patterns are hard to mix with `__device__` code,
this name-based  ABI is subject to change.

```cpp
namespace e3j {
namespace op_name {

    struct Params;

    template <typename Idx, typename Val>
    __global__ void kernel (Params p, ...);

    template <typename Idx, typename Val>
    e3j::Error launch (..., Params p, cudaStream_t stream);

} // namespace op_name
} // namespace e3j
```

Most kernels are templated across a range of index and/or value data types. Some [e3j] primitives automatically dispatch to narrow index dtypes (`uint8` / `uint16`) when feature spaces dimensions  are small enough. Value dtypes are `float32`, `float64` and `float16`, each usable with any index dtype. Kernels compute and accumulate in the value dtype. `float16` is not supported yet on the `atomicAdd` based paths (`scatter_add_1` and the `LEADING_CHANNELS` tensor product), which return `Unimplemented`.

### XLA-FFI handlers

The FFI layer uses the [XLA FFI API][xla/ffi] to bind kernel launchers
as XLA custom calls. Each handler is registered with `XLA_FFI_DEFINE_HANDLER`,
receiving typed buffers and attributes directly (no opaque descriptor packing):

```cpp
xla::Error OpNameHandler(
    cudaStream_t stream,
    int32_t num_out,
    xla::AnyBuffer x,
    xla::Result<xla::AnyBuffer> out
) {
    // ... dtype dispatch ...
    return e3j::op_name::launch<Idx, Val>(
        x.typed_data<Idx>(),
        out->typed_data<Val>(),
        params, stream
    ).to_xla();
}

XLA_FFI_DEFINE_HANDLER(
    xla_op_name,
    OpNameHandler,
    xla::Ffi::Bind()
        .Ctx<xla::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("num_out")
        .Arg<xla::AnyBuffer>()
        .Ret<xla::AnyBuffer>()
);
```

Handlers are exposed to Python as PyCapsules via `pyEncapsulateFunction`
in [e3j_ops.h](ffi/e3j_ops.h), and registered in the
pybind11 module defined in [e3j_ops.cpp](ffi/e3j_ops.cpp).

[xla/ffi]: https://jax.readthedocs.io/en/latest/ffi.html

[e3j]: https://github.com/instadeepai/e3j
[e3j-pypi]: https://pypi.org/projects/e3j
[e3j-ops-pypi]: https://pypi.org/projects/e3j_ops
[e3j_ops]: https://github.com/instadeepai/e3j/tree/main/lib/e3j_ops


Contributing
------------
Although it is too early for [e3j_ops] to accept significant external contributions, bug reports or questions are very welcome via [GitHub][e3j] issues and  discussions.

Citing
------
If you use [e3j] within your work, we kindly ask you to cite the following preprint:

```
@article{Peltre26-e3j,
    title   = {{E3J}: an Efficient and Open-Source Euclidean Equivariance Backend},
    author  = {Peltre, Olivier and Picard, Armand and Pichard, Adrien and Giacomoni, Luca and Braganca, Miguel and Heyraud, Valentin and Brunken, Christoph and Tilly, Jules},
    journal = {preprint},
    year    = {2026},
    url     = {(preprint)}
  }
}
```
