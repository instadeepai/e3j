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

import math

import jax.numpy as np

from e3j.core.polynomials import Polynomial
from e3j.spaces import O3Space
from e3j.utils.safe import safe_norm
from e3j.utils.spherical_harmonics import Y


def _normalization_factor(l: int, normalization: str) -> float:  # noqa: E741
    """Scalar rescaling of the degree-l harmonics block relative to "integral".

    e3j's `Y(l, ...)` is normalized so that `integral(S2) Y_lm^2 = 1` (the
    "integral" convention). e3nn's other conventions rescale each l-block
    by a single l-dependent constant: "component" has `||Y^l||^2 = 2l + 1`,
    "norm" has `||Y^l|| = 1`.
    """
    if normalization == "integral":
        return 1.0
    if normalization == "component":
        return math.sqrt(4 * math.pi)
    if normalization == "norm":
        return math.sqrt(4 * math.pi / (2 * l + 1))
    raise NotImplementedError(
        f"We don't support {normalization} normalization for now. "
        'We only support "integral", "component" and "norm"'
    )


class Harmonics:
    r"""Evaluation module for harmonic polynomials.

    This class provides additional control and options over polynomial
    evaluation, e.g. by safely projecting inputs on the S2 sphere of
    by rescaling polynomials by L2-norm, dimension (2l + 1), etc.

    Parameters
    ----------
    out : `int | str`
        output representation, integers being interpreted as `l in range(0, out + 1)`,
    normalize : `bool`
        whether to project inputs on the S2 sphere,
    normalization : `str`
        momentum-dependent normalization factor, one of "component"
        (default, matching `e3nn_jax`'s default, :math:`\|Y^l\|^2 = 2l + 1`),
        "integral" (:math:`\int_{S^2} Y_{lm}^2 = 1`) or "norm"
        (:math:`\|Y^l\| = 1`),
    real : `bool`
        use real spherical harmonics :math:`Y_{lm}` by default, switch off for
        complex eigenvalues :math:`Y_l^m = |lm\rangle` of :math:`J_z, J^2`.
    """

    def __init__(
        self,
        target: int | str,
        normalize: bool = False,
        normalization: str = "component",
        real: bool = True,
    ):
        """Initialise polynomials Y(l) for l in out (or l <= l_max)."""
        self.target = (
            O3Space(target)
            if isinstance(target, str)
            else O3Space([(1, (l, (-1) ** l)) for l in range(target + 1)])
        )
        Ys = [
            _normalization_factor(irrep_out.l, normalization)
            * Y(irrep_out.l, None, real=real)
            for mul, irrep_out in self.target
        ]
        self.polynomial = Polynomial.concat(Ys)
        # hack to avoid matvec prior to exponentiation
        self.polynomial.basis = "harmonics"
        self.normalize = normalize
        self.normalization = normalization
        self.real = real

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Evaluate polynomials on a 3D point cloud."""
        if self.normalize:
            norm = safe_norm(r, axis=-1)
            r_norm = r / norm[:, None]
            Yr = self.polynomial(r_norm)
        else:
            Yr = self.polynomial(r)
        return Yr if not self.real else Yr.real
