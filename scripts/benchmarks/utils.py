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

import functools
import gc
import itertools
import timeit
from pathlib import Path
from time import perf_counter as time
from typing import Any, Callable, Iterable, Optional

import git
import jax
import numpy as np
from e3nn_jax import IrrepsArray

from .backends import torch


def sizeof(x: torch.Tensor | np.ndarray | IrrepsArray) -> int:
    """Return size of array in bytes, assuming 4B dtype."""
    if isinstance(x, IrrepsArray):
        return sizeof(x.array)
    elif hasattr(x, "numel"):
        return 4 * x.numel()
    elif hasattr(x, "size"):
        return 4 * x.size
    else:
        raise ValueError(f"Unsupported array type for sizeof: {type(x)}")


def await_for(y: torch.Tensor | np.ndarray | IrrepsArray, is_torch: bool = False):
    """Await for asynchronous array computation."""
    if is_torch:
        torch.cuda.synchronize()
    else:
        jax.block_until_ready(y)


def timed_loop(
    f: Callable[[Any], Any],
    n_it: int,
    write: bool = False,
    warmup: int = 2,
    is_torch: bool = False,
    keep_best: float = 0.2,
    debug: int = 0,
) -> Callable[[Iterable[Any]], float]:
    """Measure average runtime and bandwidth over an iterable of inputs.

    The returned function consumes batched (leading axis size `n_samp`)
    and periodically loops over them for `n_it` steps.

    It returns a dictionary with keys `"runtime µs"` and "throughput GB/s".
    """
    # retrieve function or nn.Module name
    name = f.__name__ if hasattr(f, "__name__") else f.__class__.__name__

    def timed_f(*inputs):
        """Compute average runtimes and throughputs over batched inputs."""
        # warmup for compilation + size calculation
        n_samp = min(len(xd) for xd in inputs)
        xs = [x[0] for x in inputs]
        size = sum(sizeof(x) for x in xs)
        y = f(*(x[0] for x in inputs))
        size += sum(sizeof(yi) for yi in y) if isinstance(y, tuple) else sizeof(y)
        # print shapes if debug > 0
        if debug:
            sgn = " -> ".join(str(x.shape) for x in (*xs, y))
            print(f.__name__, ":", sgn)
        for i in range(1, warmup):
            f(*(x[i % n_samp] for x in inputs))
        # average over nit
        runtimes = []
        wait_for = functools.partial(await_for, is_torch=is_torch)
        for i in range(n_it):
            inputs_i = tuple(x[i % n_samp] for x in inputs)
            wait_for(inputs_i)
            dt = timeit.timeit(
                "y = f(*inputs_i); wait_for(y)",
                globals=locals(),
                number=1,
            )
            runtimes.append(dt)
        # keep best samples to mitigate garbage collection
        n_best = max(int(keep_best * n_it), 1)
        # collect throughputs
        dt = np.array(runtimes)
        dt.sort()
        dt_avg = dt[:n_best].mean()
        dt_avg_us = dt_avg * 1_000_000
        throughput_gb_s = 0.001 * size / dt_avg_us
        if write:
            print(
                f"-> '{name}' ran in {dt_avg_us:.1f} μs\t"
                f"({throughput_gb_s:.2f} GB/s, {n_it} iterations)"
            )
        return {"runtime µs": dt_avg_us, "throughput GB/s": throughput_gb_s}

    timed_f.__name__ = name
    return timed_f


def dict_grid(dcts: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    """Cartesian product of iterable fields.

    Given a flat dictionary of iterables, loop over
    their values using `itertools.product` to construct
    an iterable of dictionaries.

    To set a constant field value, provide a singleton iterable.
    """
    value_grid = itertools.product(*dcts.values())
    return (dict(zip(dcts.keys(), values)) for values in value_grid)


def dict_zip(dcts: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    """Joint (zipped) iterator over fields.

    Given a flat dictionary of iterables, zip their fields to
    return a iterable of dictionaries.
    The length of the iterator is equal to the maximal field length,
    and any shorter iterable will be looped over.

    This is a cheaper alternative to `conf_grid`.

    Example
    -------
    >>> conf_zip({"a": [0, 1], "b": [1, 2, 3, 4]})
    ... # yield {"a" : 0, "b": 1}
    ... # yield {"a" : 1, "b": 2}
    ... # yield {"a" : 0, "b": 3}
    ... # yield {"a" : 1, "b": 4}
    """
    max_size = max(len(dcts_k) for dcts_k in dcts.values())

    def repeat(items: list, size: int) -> list:
        quotient, rest = divmod(size, len(items))
        return items * quotient + items[:rest]

    repeat_dcts = {k: repeat(dcts_k, max_size) for k, dcts_k in dcts.items()}

    return (
        {k: dcts_k[i] for k, dcts_k in repeat_dcts.items()} for i in range(max_size)
    )
