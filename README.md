# e3j

Euclid-equivariant operations and harmonic polynomials for JAX.

This library is intended as a faster and full-featured substitute for the [e3nn] and [e3x] Euclidean equivariance backends, replacing slow components
in Machine Learned Interatomic Potentials (MLIPs) with carefully optimized
and open-source CUDA kernels.

The equivariance backend of our MLIP library is `e3j` as of [mlip] 0.2.0.

> **Note:** `e3j` is currently in pre-release,
> with version 0.1.0 planned for early June 2026.
> Additional CUDA kernels and dedicated Pallas kernels for TPU
> will be rolled out progressively.

[e3nn]: https://github.com/e3nn/e3nn-jax
[e3x]: https://github.com/google-research/e3x
[mlip]: https://github.com/instadeepai/mlip

## Installation

### Requirements

The `e3j` package consists of a thin JAX-based Python API which can run on CPU, GPU and TPU, and currently supports Python versions from 3.11 to 3.14 included.

For efficiency on GPU, our CUDA binaries need to be pulled via the `"e3j[ops]"` extra:

```sh
# requirements.txt
jax[cuda13_local] ~= 0.8.0
e3j[ops] == 0.1.0b0
```
See [JAX installation](https://docs.jax.dev/en/latest/installation.html) instructions for more information.

### Building from source

Our dependencies are managed with uv. After cloning the repository, you can
build from source by running run one of:

```sh
# Existing CUDA 13 install with `e3j_ops` kernels:
uv sync --group cuda13_local --extra ops
# Install CUDA 13 via pip and the `exp` group for benchmarks:
uv sync --group cuda13 --extra ops
```

The Python build internally relies on CMake, scikit-build and pybind11. You can also look at the [Makefile](Makefile) for alternate recipes to build kernels,
C++ tests and the Python bindings.
The `e3j_ops` Python package only contains our CUDA binaries and bindings
to their associated XLA handlers, and is not meant to be used as standalone.

The JAX primitives wrapping our custom XLA handlers are defined in the `e3j.ops` subpackage of `e3j`, provided the `e3j_ops` binaries can be found in the environment.

### Project structure:
- [src/](src/e3j) : python source
    + [e3j/core](src/e3j/core) :
    framework-agnostic implementations of equivariant operations
    + [e3j/linen](src/e3j/linen) : [flax.linen.Module] wrappers
- [lib/](lib/e3j_ops) : C++/CUDA source for the `e3j_ops` subpackage
    + [cuda](lib/e3j_ops/cuda) : custom kernel implementations
    + [ffi](lib/e3j_ops/ffi) : XLA and Python binding boilerplate

[flax.linen.Module]: https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html
