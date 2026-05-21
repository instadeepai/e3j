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

import e3nn_jax as e3nn
import jax.numpy as np

from e3j.core.polynomials import Polynomial
from e3j.spaces import O3Space
from e3j.utils.safe import safe_norm
from e3j.utils.spherical_harmonics import Y


class Harmonics:
    """Evaluation module for harmonic polynomials.

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
        momentum-dependent normalization factor,
    real : `bool`
        use real spherical harmonics :math:`Y_{lm}` by default, switch off for
        complex eigenvalues :math:`Y_l^m = |lm⟩` of :math:`J_z, J^2`.
    """

    def __init__(
        self,
        target: int | str,
        normalize: bool = False,
        normalization: str = "norm",
        real: bool = True,
    ):
        """Initialise polynomials Y(l) for l in out (or l <= l_max)."""
        self.target = (
            O3Space(target)
            if isinstance(target, str)
            else O3Space([(1, (l, (-1) ** l)) for l in range(target + 1)])
        )
        Ys = [Y(irrep_out.l, None, real=real) for mul, irrep_out in self.target]
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
