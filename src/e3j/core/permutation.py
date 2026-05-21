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

import jax
import jax.numpy as np

from e3j.spaces import O3Space


class Permutation:
    """Base class for permutations on a given axis.

    There are various possible implementations (sparse/dense matmul, indexing...),
    but this class is mostly intended to act (once and for all) on tensor product
    operators themselves, by reordering their coefficients.

    It is therefore mostly useful for the E3-specific classmethod it provides.

    Classmethods
    ------------
    `Permutation.sort(rep_in)` :
        sort irreducible blocks by angular momentum and parity.
    """

    def __init__(self, sigma: np.ndarray, axis: int = -1):
        self.sigma = sigma
        self.axis = axis

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return `x[...,σ[i]]` for `i <= x.shape[axis]`."""
        return np.take(x, self.sigma, axis=self.axis)

    @functools.cached_property
    def sigma_1(self) -> np.ndarray:
        """Inverse permutation."""
        return np.argsort(self.sigma)

    @classmethod
    def sort(cls, source: str | O3Space, axis: int = -1) -> str:
        """Sort irreducible blocks by angular momentum and parity.

        Note
        ----
        The `(l, p)` pairs indexing O3-representations are sorted by:

        * increasing momentum `l`,
            * matching parities `p = (-1)**l` first,
            * pseudo-tensor parities `p = -(-1)**l` last.
        """
        rep = O3Space(source)
        rep_sorted, inv = rep.sort(return_inverse=True)
        slices = rep.slices()
        slices_p = [(slices[pi].start, slices[pi].stop) for pi in inv]
        sigma = np.concat(
            [i0 + np.arange(i1 - i0) for i0, i1 in slices_p],
        )
        perm = cls(sigma, axis=axis)
        perm.source = rep
        perm.target = rep_sorted.regroup()
        return perm
