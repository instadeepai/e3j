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
    """Return monomials (X - jY), Z and (X + jY).

    These degree-1 harmonic polynomials generate higher degree
    harmonics conveniently.

    The harmonic recursion's physical axes ``(X, Y, Z)`` are mapped to the
    input components ``(z, x, y)``. This reproduces the e3nn ``(y, z, x)``
    convention, where the degree-1 harmonics are ordered ``(y, z, x)`` with
    ``m = -1, 0, +1``, so that ``Harmonics`` matches ``e3nn.spherical_harmonics``
    value-for-value (equivalently, ``Harmonics(r) == e3nn(r)`` with no coordinate
    swap on the input ``r``).
    """
    x, y, z = np.eye(3)
    X, Y, Z = z, x, y
    basis = np.stack((X - 1j * Y, Z, X + 1j * Y), axis=0).T
    # --- Degree-1 harmonics
    x_minus_jy, z, x_plus_jy = (
        Monomial(exp[None, :], coords=basis) for exp in np.eye(3, dtype=np.int32)
    )
    return x_minus_jy, z, x_plus_jy


@functools.cache
def Y(l: int, m: int | None = None, real: bool = False) -> Polynomial:  # noqa: N802
    """
    Complex harmonic polynomial Ylm(r).
    """
    # complex Y(l, m) from Y(l, l) + recursion
    if not real and m is not None:
        if m == l:
            x_plus_jy = _harmonic_generators()[2]
            inv_norm = 1 / norm_s2(l)
            return inv_norm * Polynomial(x_plus_jy**l)
        scale = HBAR / c_(l, m + 1)
        Ylm = scale * J_(Y(l, m + 1))
        return Ylm.coalesce()

    # real polynomial from Y(l, m) and Y(l, -m), in the standard
    # (Condon-Shortley / e3nn) sign convention.
    if real and m is not None:
        sign = (-1) ** (l + m) if m >= 0 else (-1) ** (l + 1)
        if m == 0:
            return sign * Y(l, 0)
        Ylm, Yl_m = Y(l, m), Y(l, -m)
        if m > 0:
            return sign * (np.sqrt(0.5) * (Ylm + (-1) ** m * Yl_m))
        else:
            return sign * (-1j * np.sqrt(0.5) * (Ylm - (-1) ** m * Yl_m))

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
    # action of J- on monomial exponents, with generators ordered
    # (p, q, r) <-> (x - jy, z, x + jy)
    # J_(x + jy) = -2 z : lower (x + jy), raise z
    d_xy = np.stack((p, q + 1, r - 1), axis=-1)
    # J_(z) = (x - jy) : lower z, raise (x - jy)
    d_z = np.stack((p + 1, q - 1, r), axis=-1)
    # filter out negative coefficients
    mask_xy, mask_z = (exp > 0 for exp in (r, q))
    nnz_xy, nnz_z = (np.nonzero(mask)[0] for mask in (mask_xy, mask_z))
    # coefficients of J- on each path
    ones = np.ones((1, m.exp.shape[0]))
    coef_xy = (-2 * r * ones * P.coef)[:, nnz_xy]
    coef_z = (q * ones * P.coef)[:, nnz_z]
    return Polynomial(Monomial(d_xy[nnz_xy], m.coords), coef_xy) + Polynomial(
        Monomial(d_z[nnz_z], m.coords), coef_z
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
