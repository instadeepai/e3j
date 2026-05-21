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

from e3j.core.polynomials import Monomial, Polynomial
from e3j.utils.factorials import bifact_even, bifact_odd

HBAR = 1


@functools.cache
def _harmonic_generators():
    """Return monomials (X + jY), (X - jY) and Z.

    These degree-1 harmonic polynomials generate higher degree
    harmonics conveniently.
    """
    x, y, z = np.eye(3)
    basis = np.stack((x + 1j * y, x - 1j * y, z), axis=0).T
    # --- Degree-1 harmonics
    a, b, z = (
        Monomial(exp[None, :], coords=basis) for exp in np.eye(3, dtype=np.int32)
    )
    return a, b, z


@functools.cache
def Y(l: int, m: int | None = None, real: bool = False) -> Polynomial:  # noqa: N802
    """
    Complex harmonic polynomial Ylm(r).
    """
    # complex Y(l, m) from Y(l, l) + recursion
    if not real and m is not None:
        if m == l:
            x_plus_jy = _harmonic_generators()[0]
            inv_norm = 1 / norm_s2(l)
            return inv_norm * Polynomial(x_plus_jy**l)
        scale = HBAR / c_(l, m + 1)
        Ylm = scale * J_(Y(l, m + 1))
        return Ylm.coalesce()

    # real polynomial from Y(l, m) and Y(l, -m)
    if real and m is not None:
        Ylm, Yl_m = Y(l, m), Y(l, -m)
        if m == 0:
            return Y(l, 0)
        if m > 0:
            return np.sqrt(0.5) * (Ylm + (-1) ** m * Yl_m)
        else:
            return -1j * np.sqrt(0.5) * (Ylm - (-1) ** m * Yl_m)

    # concatenate Y(l, -l) | ... | Y(l, l)
    if m is None:
        Yl = Y(l, -l, real)
        for m in range(-l + 1, l + 1):
            Yl = Yl | Y(l, m, real)
        return Yl.coalesce()


def J_(P: Monomial | Polynomial) -> Polynomial:  # noqa: N802
    """
    Decreasing ladder operator `J_ = Jx - 1j * Jy`.

    The action of `J_` on generators is given by:

        J_(x + jy) = - 2 z
        J_(x - jy) = 0
        J_(z) = (x - jy)

    The action of `J_` on higher-degree monomials follows by
    the Leibniz rule.
    """
    m = P.monomials if isinstance(P, Polynomial) else P
    p, q, r = m.exp.T
    # action of J- on monomial exponents
    dp = np.stack((p - 1, q, r + 1), axis=-1)
    dr = np.stack((p, q + 1, r - 1), axis=-1)
    # filter out negative coefficients
    mask_dp, mask_dr = (exp > 0 for exp in (p, r))
    nnz_dp, nnz_dr = (np.nonzero(mask)[0] for mask in (mask_dp, mask_dr))
    # coefficients of J- on each path
    ones = np.ones((1, m.exp.shape[0]))
    coef_dp = (-2 * p * ones * P.coef)[:, nnz_dp]
    coef_dr = (r * ones * P.coef)[:, nnz_dr]
    return Polynomial(Monomial(dp[nnz_dp], m.coords), coef_dp) + Polynomial(
        Monomial(dr[nnz_dr], m.coords), coef_dr
    )


def c_(l: int, m: int) -> jax.Array:
    """
    Decreasing ladder coefficient.

    The ladder coefficient c_(l, m) is given by:

        c_(l, m) = √((l+m) * (l-m+1))

    It measures the norm scaling of the non-unitary ladder operator J_ by:

        J_⋅|lm⟩ = c_(l, m) |l(m-1)⟩

    Note
    ----
    When computing normalized gener ators `|lm⟩` by repeated application of
    the ladder operator `J_`, the succession of square-roots / divisions will
    yield unsatisfying numerical errors in single precision.

    Set `JAX_ENABLE_X64=True` to force double precision.
    """
    if jax.config.x64_enabled:
        print("float64")
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    l = np.array(l, dtype=dtype)
    m = np.array(m, dtype=dtype)
    return np.sqrt((l + m) * (l - m + 1))


def norm_s2(l: int) -> jax.Array:
    """S2-norm of top-spin polynomial `(x + jy)**l`.

    Integrating by parts to get a simple recurrence formula, yields:

        ∫∫(x²+y²)ˡsinθ.dθ.dϕ = 4π (2l)!! / (2l+1)!!

    Where n!! denotes the so-called bifactorial of n (product of decreasing
    even/odd integers below n).

    This function returns the square-root of the r.h.s. above, i.e.
    the L2-norm of `(x + jy)**l` on the unit sphere.
    """
    norm2 = 4 * np.pi * bifact_even(2 * l) / bifact_odd(2 * l + 1)
    return np.sqrt(norm2)
