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

"""Mosaic TPU helpers shared across Pallas kernels."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Block sizes (in batch elements) a Mosaic TPU pipeline may tile over. Kept as
# powers of two so blocks evenly partition typical batch sizes.
DEFAULT_BATCH_BLOCK_SIZE_CANDIDATES = [1, 2, 4, 8, 16, 32]


def round_up(n: int, multiple: int) -> int:
    """Smallest multiple of `multiple` that is >= `n`."""
    return pl.cdiv(n, multiple) * multiple


def select_batch_block_size(
    batch_size: int,
    vmem_bytes_per_batch_element: int,
    *,
    batch_block_size_candidates: list[int] | None = None,
    override: int | None = None,
    num_kernel: int | None = None,
    vmem_capacity_bytes: int | None = None,
    allow_padding: bool = False,
) -> int:
    """Pick how many batch elements to process per Mosaic TPU pipeline block.

    Chooses the largest candidate block size that fits in VMEM and evenly
    divides `batch_size`, so the pipeline tiles the batch axis without a
    ragged remainder. The resulting number of blocks must also be divisible by
    `num_kernel` so the megacore split partitions blocks evenly across cores.

    Args:
        batch_size: Total number of batch elements the kernel processes.
        vmem_bytes_per_batch_element: VMEM footprint of a single batch element;
            used to bound the block size against `vmem_capacity_bytes`.
        batch_block_size_candidates: Allowed block sizes to choose from. Defaults
            to `DEFAULT_BATCH_BLOCK_SIZE_CANDIDATES`.
        override: If given, use this block size directly and skip the VMEM-based
            selection (divisibility assertions still apply).
        num_kernel: Number of cores the work is split across. Defaults to the
            TPU's core count.
        vmem_capacity_bytes: VMEM budget per core. Defaults to the TPU's VMEM
            capacity.
        allow_padding: If set, the caller pads `batch_size` up to a multiple of
            the block, so the largest VMEM-fitting candidate is chosen without
            requiring divisibility and the divisibility assertions are skipped.

    Returns:
        The selected batch block size (in batch elements).
    """
    if num_kernel is None or vmem_capacity_bytes is None:
        tpu_info = pltpu.get_tpu_info()
        num_kernel = num_kernel or tpu_info.num_cores
        vmem_capacity_bytes = vmem_capacity_bytes or tpu_info.vmem_capacity_bytes

    batch_block_size_candidates = (
        batch_block_size_candidates or DEFAULT_BATCH_BLOCK_SIZE_CANDIDATES
    )
    if override is not None:
        batch_block_size = override
    else:
        max_bbs = vmem_capacity_bytes // vmem_bytes_per_batch_element
        fits = [bbs for bbs in batch_block_size_candidates if bbs <= max_bbs]
        if allow_padding:
            batch_block_size = max(fits) if fits else min(batch_block_size_candidates)
        else:
            candidates = [
                bbs
                for bbs in fits
                if batch_size % bbs == 0 and (batch_size // bbs) % num_kernel == 0
            ]
            if not candidates:
                raise ValueError(
                    f"No batch block size in {batch_block_size_candidates} both fits "
                    f"VMEM (<= {max_bbs} batch elements) and divides batch_size="
                    f"{batch_size} into a block count that is a multiple of "
                    f"num_kernel={num_kernel}. Lower the per-element VMEM footprint "
                    f"or choose a batch size divisible by num_kernel."
                )
            batch_block_size = max(candidates)

    if not allow_padding:
        assert batch_size % batch_block_size == 0, (
            f"batch_size {batch_size} must be divisible by "
            f"batch_block_size={batch_block_size}"
        )
        num_blocks = batch_size // batch_block_size
        assert num_blocks % num_kernel == 0, (
            f"num_blocks (batch_size // batch_block_size = {num_blocks}) must be "
            f"divisible by num_kernel={num_kernel} so the megacore split "
            f"partitions blocks evenly across cores"
        )
    return batch_block_size


def pad_for_tpu(x: jax.Array) -> jax.Array:
    """Pad the last two axes to TPU tile boundaries.

    The last axis is padded up to a multiple of 128 (lane), the second-to-last
    up to a multiple of 8 (sublane). Leading axes are left untouched.
    """
    base = tuple((0, 0) for _ in range(len(x.shape[:-2])))
    return jnp.pad(x, (*base, (0, -(x.shape[-2] % -8)), (0, -(x.shape[-1] % -128))))
