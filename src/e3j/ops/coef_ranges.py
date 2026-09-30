# Copyright (c) 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side partition of sparse coefficients across the `blockDim.y` warps of
the fused CUDA kernels.

Output rows (runs of coefficients sharing an output index) are assigned whole
to warps: longest rows first, each to the least-loaded warp. Each pass is then
laid out warp by warp, and `coef_ranges` holds the per-warp end offsets.
Passes without a barrier between them (convolution backward) share one
per-warp load, passes separated by a barrier (tensor product backward) are
balanced independently.
"""

import functools

import e3j_ops
import numpy

# CUDA limit on threads per block.
MAX_BLOCK_SIZE = 1024
# Largest `blockDim.y` considered by `choose_block_y`.
MAX_BLOCK_Y = 16


@functools.lru_cache(maxsize=1)
def device_properties():
    """Cached `e3j_ops.get_device_properties()`."""
    return e3j_ops.get_device_properties()


@functools.lru_cache(maxsize=None)
def kernel_registers(kernel, idx_dtype, val_dtype, N):
    """Registers per thread of the compiled `kernel<Idx, Val, N>` (cached)."""
    idx_name = numpy.dtype(idx_dtype).name
    val_name = numpy.dtype(val_dtype).name
    return e3j_ops.get_kernel_registers(kernel, idx_name, val_name, N)


def launch_threads_x(channels):
    """Initial `(threadsX, N)` of the CUDA launchers, before any reduction to
    fit shared memory."""
    if channels // 4 >= 32 and channels % 4 == 0:
        N = 4
    elif channels // 2 >= 32 and channels % 2 == 0:
        N = 2
    else:
        N = 1
    return max(32, (channels // N // 32) * 32), N


def estimate_smem_bytes(load_elems, coef_elems, coef_bytes, val_bytes=4, extra_bytes=0):
    """Shared memory per block at `blockDim.y = 1`: 16-byte aligned operand
    buffers (`load_elems`, element counts), coefficients, and `extra_bytes`."""
    align = lambda n: -(-n // 16) * 16  # noqa: E731
    size_load = sum(align(e * val_bytes) for e in load_elems) + extra_bytes
    size_coef = align(coef_elems * coef_bytes)
    return size_load + size_coef


def _pass_rows(out_idx, num_coef):
    """`(begin, weight)` of each output row of one pass, in order."""
    begins = [0] + [c for c in range(1, num_coef) if out_idx[c] != out_idx[c - 1]]
    ends = begins[1:] + [num_coef]
    return [(begin, end - begin) for begin, end in zip(begins, ends)]


def _assign_rows(passes, num_coef, num_warps, barrier_between_passes=False):
    """Assign whole output rows to warps, longest first, each to the
    least-loaded warp.

    Loads are summed across passes, or kept per pass under
    `barrier_between_passes`. Returns per-pass permutations (warp by warp,
    within-row order kept), `(num_passes, num_warps)` end offsets, and
    per-warp loads.
    """
    rows = [_pass_rows(out_idx, num_coef) for out_idx in passes]
    loads = numpy.zeros((len(rows), num_warps), dtype=numpy.int64)
    assign = [[[] for _ in range(num_warps)] for _ in rows]

    if barrier_between_passes:
        groups = [[p] for p in range(len(rows))]
    else:
        groups = [list(range(len(rows)))]
    for group in groups:
        items = [
            (weight, p, r) for p in group for r, (_begin, weight) in enumerate(rows[p])
        ]
        items.sort(key=lambda item: (-item[0], item[1], item[2]))
        for weight, p, r in items:
            w = int(numpy.argmin(loads[group].sum(axis=0)))
            assign[p][w].append(r)
            loads[p, w] += weight

    # Rows in ascending order within each warp.
    perms = []
    endpoints = numpy.empty((len(rows), num_warps), dtype=numpy.int32)
    for p, pass_rows in enumerate(rows):
        perm = []
        cursor = 0
        for w in range(num_warps):
            for r in sorted(assign[p][w]):
                begin, weight = pass_rows[r]
                perm.extend(range(begin, begin + weight))
                cursor += weight
            endpoints[p, w] = cursor
        perms.append(numpy.asarray(perm, dtype=numpy.int64))
    return tuple(perms), endpoints, loads


def balance_coef_ranges(passes, num_coef, num_warps, barrier_between_passes=False):
    """Per-pass permutations and flat `coef_ranges`, read by the kernel as
    `coef_ranges[p * blockDim.y + threadIdx.y]`."""
    perms, endpoints, _ = _assign_rows(
        passes, num_coef, num_warps, barrier_between_passes
    )
    return perms, endpoints.reshape(-1)


def _critical_path(passes, num_coef, num_warps, barrier_between_passes=False):
    """Coefficient count of the busiest warp over summed passes, or the sum of
    per-pass busiest warps under a barrier."""
    _, _, loads = _assign_rows(passes, num_coef, num_warps, barrier_between_passes)
    if barrier_between_passes:
        return int(loads.max(axis=1).sum())
    return int(loads.sum(axis=0).max())


def choose_block_y(
    passes,
    num_coef,
    channels,
    *,
    max_block_size=MAX_BLOCK_SIZE,
    smem_bytes=None,
    registers_per_thread=None,
    barrier_between_passes=False,
):
    """`blockDim.y` minimizing the critical path divided by the number of
    thread blocks resident per SM.

    Residency is bounded by threads, shared memory (`smem_bytes`) and registers
    (`registers_per_thread`), both taken at `blockDim.y = 1`. `channels` and
    `max_block_size` match the CUDA launcher.
    """
    threads_x, _N = launch_threads_x(channels)
    cap = max(1, min(MAX_BLOCK_Y, max_block_size // threads_x))

    device = device_properties()
    max_blocks_smem = (
        max(1, device["smem_max"] // smem_bytes)
        if smem_bytes
        else device["max_blocks_per_sm"]
    )

    def cost(y):
        critical_path = _critical_path(passes, num_coef, y, barrier_between_passes)
        # Register usage per block grows with `y`.
        max_blocks_registers = (
            max(1, device["regs_per_sm"] // (threads_x * y * registers_per_thread))
            if registers_per_thread
            else device["max_blocks_per_sm"]
        )
        resident_blocks = max(
            1,
            min(
                device["max_blocks_per_sm"],
                max_blocks_smem,
                max_blocks_registers,
                device["max_threads_per_sm"] // (threads_x * y),
            ),
        )
        return critical_path / resident_blocks

    return min(range(1, cap + 1), key=cost)
