CUDA/C++ source
===============

- [lib](lib) : standalone library for custom CUDA kernels
- [ffi](ffi) : C++ binding code, using `pybind11` and the
[legacy XLA custom call API][xla/custom].

[xla/custom]: https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html
[xla/ffi]: https://jax.readthedocs.io/en/latest/ffi.html

:point_right: Note that the following is subject to change as we migrate to the [new XLA ffi api][xla/ffi].

lib
---
To keep the cognitive load of foreign bindings a minimum, we suggest to that every custom kernel's code exposes a common interface,
see e.g. [scatter_add.cuh](cuda) for detailed signatures.
The interface consists of:
- a namespace `e3j::op_name` within the enclosing `e3j` namespace,
- a `struct` definition `e3j::op_name::Params`,
- a `__global__` CUDA function `e3j::op_name::kernel`,
- a host wrapper `e3j::op_name::launch`

The `Params` will serve as target type for XLA's opaque parameter `UnpackDescriptor`, see below.

Update (June 25)
----------------

Since we rely on dynamic linking to operate with XLA, the *external* headers from
[xla/ffi/api](https://github.com/openxla/xla/tree/main/xla/ffi/api) have been copied:
- `api.h`
- `c_api.h`
- `ffi.h`

Note that the XLA FFI API is not 100% stable yet:

> WARNING: XLA FFI in under construction and currently does not provide any backward compatibility
  guarantees. Once we reach a point when we are reasonably confident that we got all APIs right,
  we will define `XLA_FFI_API_MAJOR` and `XLA_FFI_API_MINOR` API versions and will start providing
  API and ABI backward compatibility.

ffi
---

NOTE: Within `e3j`, the goal is to use the two helpers defined in [kernel_helpers.h](ffi/kernel_helpers.h) generically, e.g.
```cpp
- PackDescriptor<Params>	(Params p) -> std::string(opaque, opaque_len)
- UnPackDescriptor<Params>	(char *opaque, size_t opaque_len) -> Params p
```
so as to pass kernel hyperparameters to the XLA custom call. To make the
FFI and XLA binding boilerplate generic, the current to strategy is to
let any custom kernel be defined in its own namespace within `e3j`, with
its own `kernel` and `launch` functions:

```cpp
namespace e3j { namespace op_name {

	struct Params;

	template <typename T>
	__global__ void kernel<T> (T *a, ..., Params p);

	template <typename T>
	void launch (T *a, ..., Params p, cudaStream_t stream);

	...

}}
```

This way, it is straightforward for the FFI-directed `e3j_ops` namespace
to define the XLA custom call (without cognitive overload) as:

```cpp
namespace e3j_ops {

	void op_name (
		cudaStream_t,
		void **buffers,
		char *opaque,
		size_t opaque_len
	){

		e3j::op_name::Params params =
			UnpackDescriptor(opaque, opaque_len);

		e3j::op_name::launch(
			buffers[0],
			...,
			params,
			stream
		);
	}
}
```
