# e3j

Euclid-equivariant operations and harmonic polynomials for JAX.

This library is intended as a faster and reliable substitute for the [e3nn] and [e3x] Euclid-equivariant
backends.
Our longer term goal is to replace slow components in equivariant MLIPs, as shipped with the [mlip-jax] repo.

For now, e3j is meant to operate alongside `e3nn`,
on which it depends for a few non-critical operations (e.g. Irreps manipulations and Clebsch-Gordan coefficients).
This dependency may be dropped in the future.

[e3nn]: https://github.com/e3nn/e3nn-jax
[e3x]: https://github.com/google-research/e3x
[mlip-jax]: https://github.com/instadeepai/mlip-jax

### Project structure:
- [src/](src/e3j) : python source
    + [e3j/core](src/e3j/core) :
    framework-agnostic implementations of equivariant operations
    + [e3j/linen](src/e3j/linen) : [flax.linen.Module] wrappers
- [lib/](lib/e3j_ops) : C++/CUDA source for the `e3j_ops` subpackage
    + [cuda](lib/e3j_ops/cuda) : custom kernel implementations
    + [ffi](lib/e3j_ops/ffi) : XLA and Python binding boilerplate

[flax.linen.Module]: https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html

## Installation

### As a dependency

In the MLIP dependencies, managed with `uv`, we have the following set up:

```toml
[dependency-groups]
e3j = ["e3j"]
e3j_ops = ["e3j_ops"]

[tool.uv.sources]
e3j = {git="https://github.com/instadeepai/e3j", branch="main"}
#e3j = {git = "https://github.com/instadeepai/e3j", rev="fea221c4191204c87a960153738da38cc1923a6b"}
```

### For development

Dependencies are managed with uv, after cloning the repository you can run one of:

```sh
# CPU install
uv sync --group cpu
# Existing CUDA 12 install with `e3j_ops` kernels:
uv sync --group cuda_local --extra ops
# Install CUDA 12 via pip and the `exp` group for benchmarks:
uv sync --group cuda --group exp --extra ops
```

You should now be able to run the tests with:

```sh
uv run pytest
```

__Note:__ if you make a change to the `lib/` code and would like to rebuild CUDA for the `uv` environment, you can run:
```sh
make uv # forces `uv cache clean` of the e3j_ops shared object
make pytest # pytest -m "e3j_ops" tests/test_ops
```

### Building `e3j_ops`

The `e3j.ops` subpackage optionally provides JAX bindings to our
sparse kernels via the XLA FFI.

There are currently two ways to build or test the `e3j_ops` shared object:

1. Installing `e3j` with the `ops` extra, which internally relies on `cmake`
and `scikit-build`.

   ```sh
   # Editable dev install
   uv sync --group cuda|cuda_local --extra ops
   # Dependency install
   pip install e3j[ops] # (once uploaded to PyPI, add git url/token for now)
   ```
2. Building any of the recipes from the [Makefile](Makefile) e.g.

   ```sh
   # Build the Pybind11 shared object
   make e3j_ops && export PYTHONPATH=$PWD/bin:$PYTHONPATH
   # Run a specific test
   make test_tensor_product
   # Build tests and run all e3j_ops dependent tests (C++ and Python)
   make test
   ```

Both (1) and (2) require that you have `nvcc`, `g++` and eventually `pybind11`
installed. You should then be able to import `e3j_ops` from Python, exposing
the raw XLA custom calls. Kernels should however be called from Python
using the `e3j.ops` namespace API.


## Documentation

Build the docs locally with:
```sh
cd docs && uv run make html
#=> file:///<path-to-e3j>/docs/_build/html/index.html`.
```

## Running benchmarks

See the [scripts](scripts) directory for benchmarks of [e3x], [e3nn] and e3j.

```sh
# run a collection of benchmarks
uv run python scripts/benchmark_main.py
# run individual benchmarks
uv run python -m scripts.benchmarks.benchmark_tensor_product
uv run python -m scripts.benchmarks.benchmark_harmonics
...
```

The main benchmarking script can be configured by the
[scripts/config.yaml](scripts/config.yaml) file.
For each of the `E3Benchmark` subclasses listed within,
- the `skip` field control whether the benchmark case will be executed,
- the `grid` field will be expanded into a cartesian product of hyper-parameters,
- Hyper-parameters are converted by `BENCHMARK_PARSERS[name]` into a tuple of constructor arguments (e.g. `l_max` determines source and target representations, etc.)

TODO: cleanup benchmarks, to automate the definition of backward cases and get rid of the hacky `BENCHMARK_PARSERS`, that should be implemented in the class.
