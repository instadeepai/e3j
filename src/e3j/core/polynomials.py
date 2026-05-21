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

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as np
import jax.random
from jax.experimental import sparse

from e3j.utils.custom_jvp import CustomJVP


def block_diagonal(a: jax.Array, b: jax.Array) -> jax.Array:
    """
    Block diagonal matrix stacking blocks a and b.
    """
    # offdiag blocks
    za = np.zeros((a.shape[0], b.shape[1]))
    zb = np.zeros((b.shape[0], a.shape[1]))
    # fill rows with zeros
    row_a = np.concat((a, za), axis=-1)
    row_b = np.concat((zb, b), axis=-1)
    # concat rows
    return np.concat((row_a, row_b), axis=0)


def block_diag(*blocks: jax.Array) -> jax.Array:
    """
    Block diagonal matrix arbitrary number of blocks
    """
    rows = []
    for i, block in enumerate(blocks):
        ncols_l = sum(b.shape[1] for b in blocks[:i])
        ncols_r = sum(b.shape[1] for b in blocks[i + 1 :])
        nrows = block.shape[0]
        zl, zr = (
            np.zeros((nrows, ncols), dtype=block.dtype) for ncols in (ncols_l, ncols_r)
        )
        row_i = np.concat((zl, block, zr), axis=1)
        rows.append(row_i)
    return np.concat(rows, axis=0)


class Monomial:
    """
    Batched monomials w.r.t. an optional coordinate basis.

    Given an array `m = m[:n,:d]` of exponents, `M = Monomial(m)` will
    efficiently evaluate n monomials on a d-dimensional point cloud `x`::

        M(x)[:, i] = x[:, 0] ** m[i,0] * ... * x[:, d-1] ** m[i, d-1]

    Optionally, a (possibly complex) coordinate matrix can be used in place
    of the canonical degree-1 generators.

    Note
    ----
    The `Monomial` class is intended to store the collection of monomials
    that need to be evaluated by downstream `Polynomial` instances, i.e.
    before aggregation with a coefficient matrix.

    Keeping these classes distinct (although monomials are one-term polynomials)
    is useful for having distinct product/power implementations.
    """

    def __init__(self, exp: jax.Array, coords: jax.Array | str | None = None):
        """
        Create monomials from exponents array and optional coordinate matrix.

        Parameters
        ----------
            exp (`jax.Array`):
                exponents array of shape `[n, d]`
            coords (`jax.Array | None`):
                an optional `[d, d]` matrix of coordinate functions.
        """
        self.exp = exp
        self.coords = coords

    @functools.partial(jax.profiler.annotate_function, name="monomial")
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Evaluate monomials on a point cloud.

        Parameters
        ----------
            x (`jax.Array`):
                a point cloud of shape `[b, d]`
        Returns
        -------
            mx (`jax.Array`):
                evaluated monomials of shape `[b, n]`
        """
        # evaluate coordinates = generating order-1 monomials
        x = self.eval_coords(x)
        # exponentiate coordinates
        return self.exponentiate_and_multiply(x)

    def eval_coords(self, r: jax.Array) -> jax.Array:
        if self.coords == "harmonic":
            x, y, z = r[..., 0], r[..., 1], r[..., 2]
            return np.stack([x + 1j * y, x - 1j * y, z], axis=-1)
        return r if r is None else r @ self.coords

    @jax.profiler.annotate_function
    def exponentiate_and_multiply(self, x: jax.Array) -> jax.Array:
        # expand monomial dimension
        x = np.expand_dims(x, -2)
        # exponentiate coordinates
        xm = x**self.exp
        # fold monomials coordinate-wise
        return np.prod(xm, axis=-1)

    @classmethod
    def concat(cls, ms: tuple[Monomial]) -> Monomial:
        if not len(ms):
            raise ValueError("Cannot concatenate empty monomial sequence.")
        if any(m.coords is not ms[0].coords for m in ms):
            raise NotImplementedError(
                "Cannot concatenate monomials with different coordinates."
            )
        exp = np.concat(tuple(m.exp for m in ms), axis=0)
        return Monomial(exp, ms[0].coords)

    def __or__(self, other: Monomial) -> Monomial:
        """Concatenate two monomial batches."""
        if self.coords is other.coords:
            exp = np.concat((self.exp, other.exp), axis=0)
            return Monomial(exp, self.coords)
        raise NotImplementedError("`a.coords` is not `b.coords`")

    def __add__(self, other: Monomial) -> Polynomial:
        """Return a polynomial made of two monomials."""
        if self.coords is other.coords:
            return Polynomial(self | other)

    def __mul__(self, other: Monomial) -> Monomial:
        """Multiply two monomial batches."""
        if self.coords is other.coords:
            if self.exp.shape[0] is other.exp.shape[0]:
                return Monomial(self.exp + other.exp, self.coords)
            else:
                raise NotImplementedError("shapes of `a.exp` and `b.exp` don't match")
        raise NotImplementedError("`a.coords` is not `b.coords`")

    def __pow__(self, k: int) -> Monomial:
        """Integer power of monomials."""
        return Monomial(self.exp * k, self.coords)

    def __repr__(self) -> str:
        out = "Monomial:\n"
        exp = "- exp: " + repr(self.exp).replace("\n", "\n" + " " * 7)
        coords = "- coords: " + repr(self.coords).replace("\n", "\n" + " " * 10)
        return out + exp + "\n" + coords

    def C(self) -> MonomialC:  # noqa: N802
        return MonomialC(self.exp, self.coords)


class Polynomial:
    """
    Batched polynomials, aggregating monomials with a coefficient matrix.

    Given a d-variate `Monomial` instance `M` and a `(k, n)` coefficient matrix `C`,
    evaluating `P = Polynomial(M, C)` on a `(b, d)` point cloud `x` yields
    `batch_size * k` scalars given by::

        P(x)[s, i] = sum(C[i, j] * M(x)[s, j] for j in range(n))

    """

    def __init__(
        self,
        monomials: Monomial,
        coef: jax.Array | None = None,
        shape: tuple | None = None,
    ):
        """
        Create polynomials from array of monomials and coefficient matrix.

        Parameters
        ----------
        monomials : `Monomial`
            an array of `n` monomials
        coef : `jax.Array | None`
            an optional `(k, n)` matrix of coefficients, defaults to ones.
        shape : `tuple | None`
            an optional leading shape, mostly useful to arrange outputs of
            polynomial differentials. Defaults to `(k,)`.
        """
        self.monomials = monomials
        if coef is None:
            coef = np.ones((1, monomials.exp.shape[0]))
        if coef.ndim > 2:
            shape = coef.shape[:-1]
            coef = coef.reshape((-1, coef.shape[-1]))
        self.coef = coef
        self.coef_t = coef.transpose((-1, -2))
        if shape is None:
            shape = tuple(coef.shape[:-1])
        self.shape = shape

    @classmethod
    def concat(cls, ps: Polynomial | Monomial, axis: int = -1) -> Polynomial:
        """
        Concatenate polynomial instances.
        """
        ps = tuple(p if isinstance(p, Polynomial) else Polynomial(p) for p in ps)
        if not len(ps):
            raise ValueError("Cannot concatenate empty polynomial sequence.")
        # stack monomials if necessary
        if any(not (p.monomials is ps[0].monomials) for p in ps[1:]):
            monomials = Monomial.concat(tuple(p.monomials for p in ps))
        else:
            monomials = ps[0].monomials
        # block diagonal coefficients
        coef = block_diag(*(p.coef for p in ps))
        return Polynomial(monomials, coef)

    @classmethod
    def stack(cls, ps: Polynomial, axis: int = -1) -> Polynomial:
        """
        Stack polynomials along a given axis.
        """
        shapes = [p.shape for p in ps]
        if not len(ps):
            raise ValueError("Cannot stack empty polynomial sequence.")
        if any(sh != shapes[0] for sh in shapes[1:]):
            raise ValueError(f"Cannot stack inconsistent shapes {shapes}")
        shape1 = shapes[0]
        shape = (*shape1[:axis], len(ps), *shape1[axis:])
        cat_ps = cls.concat(ps, axis=0)
        coef = cat_ps.coef.reshape((len(ps), *shape1, -1))
        if axis < 0:
            axis = axis + 1 + len(shape1)
        if axis != 0:
            t = (*range(1, axis + 1), 0, *range(axis + 1, len(shape1) + 2))
            coef = np.transpose(coef, t)
        return Polynomial(cat_ps.monomials, coef, shape)

    def coalesce(self, sort=False) -> Polynomial:
        """
        Aggregate coefficients on equal monomials.
        """
        # sort monomials
        m = self.monomials
        exp, inv = np.unique(m.exp, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        monomials = Monomial(exp, m.coords)
        # aggregate coefficients
        m, n = exp.shape[0], self.exp.shape[0]
        ones = np.ones(n)
        agg = sparse.BCOO((ones, np.stack((np.arange(n), inv), axis=1)), shape=(n, m))
        coef = self.coef @ agg
        return Polynomial(monomials, coef, self.shape)

    @functools.cached_property
    def diff(self) -> Polynomial:
        """
        Differential of polynomials.
        """
        # Escape the tracer context:
        # caching the differential is a side-effect
        with jax.ensure_compile_time_eval():
            m, c = self.exp.T, self.coef
            diffs = []
            # Loop over input dimensions
            for i in range(m.shape[0]):
                mi = m[i : i + 1]
                mask_i = mi[0] > 0
                # d(xi^mi) / dxi = mi . xi^(mi - 1)
                dmi = np.concat((m[:i], (mi - 1), m[i + 1 :])).T
                dmi = np.where(mask_i[:, None], dmi, 0)
                dci = mi[0] * c
                Mi = Monomial(dmi, self.coords)
                dPi = Polynomial(Mi, dci, self.shape)
                diffs.append(dPi)
            # Stack the dP / dxi derivatives
            return Polynomial.stack(diffs, axis=-1).coalesce()

    @property
    def exp(self) -> jax.Array:
        return self.monomials.exp

    @property
    def coords(self) -> jax.Array:
        return self.monomials.coords

    def __or__(self, other: Polynomial | Monomial) -> Polynomial:
        """Stack two batches of polynomials."""
        if isinstance(other, Monomial):
            return self | Polynomial(other)
        if self.monomials is other.monomials:
            coef = np.stack((self.coef, other.coef), axis=0)
            return Polynomial(self.monomials, coef)
        # stack monomials
        monomials = self.monomials | other.monomials
        # block diagonal coefficients
        coef = block_diagonal(self.coef, other.coef)
        return Polynomial(monomials, coef)

    def __add__(self, other: Polynomial) -> Polynomial:
        """Add two batches of polynomials."""
        if self.monomials is other.monomials:
            return Polynomial(self.monomials, self.coef + other.coef)
        monomials = self.monomials | other.monomials
        coef = np.concat((self.coef, other.coef), axis=-1)
        return Polynomial(monomials, coef)

    def __sub__(self, other: Polynomial) -> Polynomial:
        """Substract two batches of polynomials."""
        if self.monomials is other.monomials:
            return Polynomial(self.monomials, self.coef - other.coef)
        monomials = self.monomials | other.monomials
        coef = np.concat((self.coef, -other.coef), axis=-1)
        return Polynomial(monomials, coef)

    def __mul__(self, other: Polynomial) -> Polynomial:
        """Multiply two polynomials."""
        raise NotImplementedError("unimplemented polynomial product")

    def __rmul__(self, other: float | complex | jax.Array) -> Polynomial:
        """Rescale polynomials"""
        return Polynomial(self.monomials, other * self.coef)

    def aggregate(self, mx: jax.Array) -> jax.Array:
        return mx @ self.coef_t

    @CustomJVP
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Evaluate polynomials on a point cloud.

        Parameters
        ----------
            x (`jax.Array`):
                a point cloud of shape `[..., d]`
        Returns
        -------
            px (`jax.Array`):
                evaluated polynomials, of shape `[..., k]`
                where `k = p.coef.shape[0]`.
        """
        bshape = x.shape[:-1]
        mx = self.monomials(x)
        px = self.aggregate(mx)
        if len(self.shape) > 1:
            px = px.reshape((*bshape, *self.shape))
        return px

    def _custom_jvp(
        self,
        primals: tuple[jax.Array],
        tangents: tuple[jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        """Custom Jacobian-Vector product: forward-mode A.D."""
        x, dx = primals[0], tangents[0]
        Px = self(x)
        dPx = self.diff(x)
        # broadcast dx with dPx: (N, *shape, 3, 1)
        # - N : batch dimension, common to dPx and dx
        # - shape : additional dimensions for e.g. gradient coordinates
        # - 1 : dummy dimension for 3x3 @ 3x1 batch_matmul
        axes = (*range(1, len(self.shape)), -1)
        dx_ = np.expand_dims(dx @ self.coords, axis=axes)
        return Px, (dPx @ dx_).squeeze(-1)

    def __repr__(self) -> str:
        m = self.monomials
        out = "Polynomial:\n"
        out += ("- monomials:\n" + repr(m.exp)).replace("\n", "\n" + " " * 4) + "\n"
        out += ("- coef:\n" + repr(self.coef)).replace("\n", "\n" + " " * 4)
        return out

    def C(self) -> PolynomialC:  # noqa: N802
        return PolynomialC(
            Monomial.C(self.monomials),
            self.coef,
            self.shape,
        )


class MonomialC(Monomial):
    """
    Complex monomials.

    The `MonomialC` class uses a different exponentiation strategy to
    exponentiate *complex* linear forms, e.g. harmonic polynomials of
    minimal/maximal magnetic momentum:

        Ym,±m(x, y, z) = (x ± 1j * y) ** m

    The real and imaginary part of `monomial.coords` are split and concatenated,
    while complex exponentiation in pulled back to polar coordinates using
    `np.arctan2` and `np.linalg.norm`.
    """

    def __init__(self, exp: jax.Array, coords: Optional[jax.Array] = None):
        self.exp = exp
        self.coords = np.concat((coords.real, coords.imag), axis=-1)

    def eval_coords(self, r: jax.Array) -> jax.Array:
        if self.coords == "harmonic":
            x, y, z = r[..., 0], r[..., 1], r[..., 2]
            o = np.zeros_like(z)
            return np.stack([x, x, z, y, -y, o], axis=-1)
        return r if r is None else r @ self.coords

    @jax.profiler.annotate_function
    def exponentiate_and_multiply(self, x: jax.Array) -> jax.Array:
        """
        Exponentiate coordinates.

        Parameters
        ----------
            x (`jax.Array`):
                point cloud of complex coordinates `[b, 2c]`
        Returns
        -------
            mx (`jax.Array`):
                evaluated monomials of shape `[b, 2n]`
        """
        a, b = x[:, :3], x[:, 3:]
        # complex module
        r = np.sqrt(a * a + b * b)
        # r ** m : (..., monomials)
        r = np.expand_dims(r, -2)
        rm = np.prod(r**self.exp, axis=-1)
        # complex phase
        phi = np.arctan2(a, b)
        # phi : (..., monomials)
        phi = np.expand_dims(phi, -2)
        mphi = np.sum(phi * self.exp, axis=-1)
        # mx : (..., 2 * monomials)
        cosm, sinm = np.cos(mphi), np.sin(mphi)
        return np.concat(
            (
                rm * cosm,
                rm * sinm,
            ),
            axis=-1,
        )


class PolynomialC(Polynomial):

    def aggregate(self, mx: jax.Array) -> jax.Array:
        n_real = self.coef_t.shape[0]
        real, imag = mx[:, :n_real], mx[:, n_real:]
        return real @ self.coef_t
        return np.concat((real @ self.coef_t, imag @ self.coef_t), axis=-1)
