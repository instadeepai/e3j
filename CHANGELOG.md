# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

### Added

- Pallas Mosaic TPU tensor product forward and backward kernels, selectable via
  the new `TensorProduct.FUSED_MOSAIC_TPU` option (TRAILING_CHANNELS layout, OUTER
  and MAP modes). The kernels statically unroll the Clebsch-Gordan coefficients
  and use pipeling to reach peak performance.

## [0.1.0b3] — 2026-06-10

### Fixed

- Fused tensor product backward (`tensor_product_bwd()`) left gradient rows for
  uncoupled input coordinates uninitialized (garbage / NaN) instead of zero. The
  kernel only writes coordinates present in the Clebsch-Gordan coefficients, so
  coordinates absent from them — common when the forward output is truncated,
  e.g. symmetric contraction in MAP mode — were never written, diverging from the
  SPARSE/DENSE autodiff gradients. Those rows are now zeroed explicitly after the
  kernel.

### Changed

- Tensor product forward and backward CUDA kernels now share a single
  `CuArray2D.load()` / `.store()` abstraction for global↔shared memory copies,
  replacing the per-operand branching between contiguous and strided variants.
  Loads use LDGSTS (asynchronous copy via pipeline primitives) and stride through
  source channels automatically when they exceed the shared-memory budget.

## [0.1.0b2] — 2026-05-28

### Added

- CUDA kernel for tensor product backward (trailing channels). This kernel loads output
  cotangents only once to compute both input cotangents via two tensor product operations.
- JAX primitive `e3j.ops.tensor_product_bwd()` with AD rules, vmap and sharding support.
  The double backward rule is expressed in terms of one `tensor_product_bwd()` call and
  two `tensor_product()` calls, providing infinite differentiability.

### Changed

- Dropped leftover `unroll` parameters that could be passed from the Python side. They
  are now private to the CUDA side and inferred alongside kernel launch configuration.

## [0.1.0b1] — 2026-05-21

### Added

- Python 3.14 `e3j_ops` build on PyPI.

### Changed

- Scalar mixing now exported as `e3j.core.ScalarMixing`.
- READMEs of `e3j` and `e3j_ops` on PyPI.

## [0.1.0b0] — 2026-05-21

### Fixed

- Tensor product broadcasting behaviour when only one operand has an additional batch axis.
  The bug occured for instance when computing Hessians with `jax.jacrev`, batched cotangents
  then require broadcast.

### Changed

- The default layout is now `TRAILING_CHANNELS`.
- The default config selects CUDA kernels for tensor product, unless `ModuleNotFoundError` is caught
  when trying to import `e3j_ops` binaries.

## [0.1.0a12] — 2026-05-12

### Fixed

- Tensor product `vmap` behaviour when one operand doesn't have an explicit channel axis.

### Changed

- Construction of Clebsch-Gordan coefficients no longer uses `sparse.BCOO.fromdense()`
  and relies on numpy for most of the sparse concatenation logic to speedup
  the initialization of tensor product modules.

## [0.1.0a11] — 2026-05-12

### Changed

- `TRAILING_CHANNELS` OUTER mode now vectorizes over channels of LHS `x` instead
  of RHS `y`, to match calls where `x` stands for node features and `y` stands
  for harmonic embeddings of edge vectors (better mnemonics).
- O3Space now supports `tuple[MulIrrep]` as input, since `flax.linen` modules
  implicitly casts `e3nn.Irreps` into `tuple[MulIrrep]`. It is not necessary
  to serialize e3nn irreps to strings as flax attributes anymore.
- O3Space and SO3space are now hashable.


## [0.1.0a10] — 2026-04-29

### Fixed

- Relaxed error handling in `e3j_ops` that was too strict and would fail in
  second backward pass. For now only checks channels are 1 or 32-multiple with
  `TRAILING_CHANNELS` layout.

## [0.1.0a9] — 2026-04-28

### Fixed

- Inconsistent alignment with `uint16` indices and `float` values (dropped `uint16`).
- Better error handling in `e3j_ops` with `TRAILING_CHANNELS` layout.

## [0.1.0a8] — 2026-04-25

### Added

- Support for leading / trailing channels layouts on CPU, with `tensor_product="SPARSE"`.
  Note the `tensor_product="FUSED"` option is still not available on CPU until a dedicated
  CPU-only build of `e3j_ops` is provided.

## [0.1.0a7] — 2026-04-25

### Fixed

- Explicit 32, 64 or 128 bit alignment of coefficients on CUDA with `TRAILING_CHANNELS`
  layout (avoids 92 bits).

### Changed

- Tensor product kernel with `TRAILING_CHANNELS` layout can vectorize over 2 or 4 channels
  at a time, reducing the cost of coefficient loads by the same factor.

## [0.1.0a6] — 2026-04-24

### Added

- Trailing and leading channel layout support in `Linear` and `LinearIndexwise`,
  controlled by the `layout` parameter (defaults to `config().layout`).
- Support for padding/skipping zero blocks absent from source/target
  in `LinearIndexwise`, useful e.g. for skip-connections.
- `LinearInitialization` and `LinearIndexwiseInitialization` enum options.
  The `FAN_IN_FCTP` option for LinearIndexwise matches the effective normalization
  of `e3nn.FullyConnectedTensorProduct` with one-hot encoded species.
- Channels parameter on `Linear` for mapping over an explicit I/O channel dimension.
- `ScalarMixing` module for rescaling equivariant features by per-irrep scalars.
- `Coef` dataclass for 3D sparse COO coefficients as a structured heterogeneous-dtype.
  The `coef.pack_jax()` passes a 32-bit aligned opaque `idx_t*` array through the
  FFI boundary which is decoded as `val, (i, j, k)` on the CUDA side, so as to
  avoid allocating memory ourselves for packed coefficients.
- `vmap` support for `tensor_product` via `jax.custom_batching.custom_vmap`,
  batching over x and y by merging the vmap axis with the leading row dimension.
  Batching over coefficients is not supported and raises an `AssertionError`.
- Multi-device sharding for vmapped `tensor_product` via
  `jax.experimental.custom_partitioning`. Coefficients are replicated across
  devices; x, y, and the output follow the caller's sharding.

### Changed

- `Linear` and `LinearIndexwise` accept `kernel_init` and `rescale_gradients`
  instead of `weights_normalization` and `grad_normalization`.
- Initialization scaling is applied downstream in the computation graph
  (when `rescale_gradients=True`) rather than baked into stddev alone.

## [0.1.0a5] — 2026-04-23

### Fixed

- Slow compilation and OOM with `e3j.config.aggregation="SCATTER"`
  (JAX-based, CPU compatible implementation) caused by
  `indices_are_sorted=True` in `jax.lax.scatter_add`, which triggered
  XLA constant-folding of large gather intermediates in the backward pass.
- Integer overflow in `target_matrix` when narrow `uint8` coefficient
  indices are used with `e3j.config.aggregation="DENSE"` or `"SPARSE"`.
  The `np.arange(nnz)` row indices now always use `int32`.
- `O3Array` subclass registration as dataclass to allow returning from
  JIT-compiled functions.

## [0.1.0a4] — 2026-04-21

### Fixed

- Stricter error handling in `O3Irrep` and `SO3Irrep` initialization.
- Return subtype of `xla::Ffi::Error` from CUDA kernels and XLA handlers.
- Overflow in index computation by using `size_t` instead of `int`.
- Device synchronization before freeing coefficient memory with `TRAILING_CHANNELS`.
- Attribute name `space.l_max` in `RBF` and `Linear`

### Changed

- Replace `switch` with `if constexpr` for compile-time mode dispatch,
  allowing nvcc elimination of unused branches in device code.
- Read from buffered coefficients in SMEM in the CUDA tensor product kernel.
- Query device shared memory limits at runtime instead of hardcoded H100 values.
- Use narrow `uint8` or `uint16` dtypes for coefficient indices when feature sizes allow.

## [0.1.0a3] — 2026-04-03

Initial tag for MLIP v2.
