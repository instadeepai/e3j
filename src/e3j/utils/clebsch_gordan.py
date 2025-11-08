from enum import Enum
from typing import Iterable, NamedTuple, TypeAlias

import numpy as np


class BCOO(NamedTuple):
    indices: np.ndarray
    values: np.ndarray


class Convention(Enum):
    E3NN = "E3NN"
    EIGVEC = "EIGVEC"


def clebsch_gordan(L: int, l1: int, l2: int, convention: Convention = "E3NN") -> BCOO:
    """Return all eigenvectors |LM⟩ f J² and Jz, i.e. Clebsch-Gordan coefficients.

    Args:
        L: target degree, which must satisfy |l1-l2| <= L <= l1+l2.
        l1: l.h.s. degree,
        l2: r.h.s. degree,

    Returns:
        a 3D sparse BCOO array of Clebsch-Gordan coefficients.

    If convention='EIGVEC', returns a sparse decomposition of the generating
    eigenvectors |LM⟩ in the basis of pure tensors |lm,l'm'⟩, i.e. such that

        J²|LM⟩ = L(L+1) |LM⟩    and     Jz |LM⟩ = M |LM⟩

    If convention='E3NN', returns a sparse BCOO representation of the Clebsch-
    Gordan coefficients agreeing with `e3nn.clebsch_gordan(L, l1, l2)` and
    corresponding to decompositions in *real* spherical harmonics, which
    are not eigenvalues of the spin operator Jz.
    """
    if convention == Convention.EIGVEC:
        basis = _eigvecs(l1, l2, L)
        idx = np.concatenate([kij for kij, _ in basis], axis=-1)
        val = np.concatenate([ckij for _, ckij in basis])
        return BCOO(idx, val)

    # FIXME: need to change basis. Up to a phase and √2 factors,
    #        the relationship with e3nn generators should be:
    #
    #     Ylm = { |lm⟩ + |l(-m)⟩ if m > 0
    #           { |lm⟩           if m == 0
    #           { |lm⟩ - |l(-m)⟩ if m < 0
    #
    #     |lm⟩ = { Ylm + Yl(-m)  if m > 0
    #            { Ylm           if m == 0
    #            { Ylm - Yl(-m)  if m < 0
    #
    if convention == Convention.E3NN:
        idx, val = clebsch_gordan(L, l1, l2, convention=Convention.EIGVEC)
        idx += np.array([L, l1, l2])[:, None]
        val /= np.sqrt(2 * L + 1)
        return BCOO(idx, val)

    return clebsch_gordan(L, l1, l2, Convention(convention))


clebsch_gordan.Convention = Convention

# Private


def _c_minus(l, m):
    """Ladder coefficient for J-."""
    return np.sqrt((l + m) * (l - m + 1))


def _c_plus(l, m):
    """Ladder coefficient for J+."""
    return np.sqrt((l - m) * (l + m + 1))


def _max_spin_eigvec(l1: int, l2: int, L: int) -> BCOO:
    """Return the max-spin eigenvector |LL⟩ of J² in sparse format.

    By the 2nd order Leibniz rule, the operator J₊J₋ acts on a dyadic
    tensor product as:

        J₊J₋  ~  (J₊J₋⊗ 1) + (1⊗ J₊J₋)
                + (J₊⊗ J₋) + (J₋⊗ J₊)

    Since J² = J₊J₋ + Jz + Jz², the action of squared angular momentum
    on a pure state |lm,l'm'⟩ is given by:

                     J² on each summand + 2 Jz⊗ Jz
                       /----------^----------\

        J²|lm,l'm'⟩ = {l(l+1) + l'(l'+1) + 2mm'} |lm,l'm'⟩

                    + c₋(m)c'₊(m') |l(m-1),l'(m'+1)⟩

                    + c₊(m)c'₋(m') |l(m+1),l'(m'-1)⟩


    Assume V = ∑ V[m,m'] |lm,l'm'⟩ is an eigenvector |LL⟩ for some L.
    By additivity of z-spin M=L, we can write

        V = ∑ V[m] |lm,l'(L-m)⟩ = ∑ V[m] |m,(L-m)⟩

    The eigenvalue equations thus reads:

        L(L+1) V[m] = {l(l+1) + l'(l'+1) + 2mm'} V[m]
                    + c-(m+1)c'+(m'-1) V[m+1]
                    + c+(m-1)c'-(m'+1) V[m-1]
    """
    if L == l1 + l2 or l1 == 0:
        return BCOO(np.array([[l1], [l2]]), np.ones(1))

    if l1 > l2:
        (m2, m1), val = _max_spin_eigvec(l2, l1, L)
        return BCOO(np.stack((m1[::-1], m2[::-1])), val[::-1])

    assert l2 - l1 <= L <= l1 + l2

    # lexicographically ordered support satisfying
    #   m1 + m2 = M = L => m2 = L - m1
    m1 = np.arange(L, l1 + l2 + 1) - l2
    m2 = L - m1
    val = np.ones(m1.size)
    # diagonal terms independent of m1, m2
    coef_l12 = l1 * (l1 + 1) + l2 * (l2 + 1)
    coef_L = L * (L + 1) - coef_l12
    # initialize val[0:2], m1 is minimal
    v1 = val[0] * (coef_L - 2 * m1[0] * m2[0])
    val[1] = v1 / (_c_minus(l1, m1[1]) * _c_plus(l2, m2[1]))
    # propagate 2nd order recurrence
    for i in range(1, m1.size - 1):
        vi = val[i] * (coef_L - 2 * m1[i] * m2[i])
        vi -= val[i - 1] * _c_plus(l1, m1[i - 1]) * _c_minus(l2, m2[i - 1])
        val[i + 1] = vi / (_c_minus(l1, m1[i + 1]) * _c_plus(l2, m2[i + 1]))

    return BCOO(np.stack((m1, m2)), val)


def _ladder_down(l1: int, l2: int, vec: BCOO) -> BCOO:
    """Apply the ladder operator J_ on a BCOO dyadic product.

    It is assumed that coordinates (m1, m2) are increasing and
    decreasing respectively, i.e. (m1, m2) is lexicographically
    ordered (by m1 + m2 = M, which implies m2 = M - m1).
    """
    idx, val = vec
    m1, m2 = idx
    M = (m1 + m2)[0]

    if m1[0] > -l1 and m2[-1] > -l2:
        # support increases
        m1_down = np.append(m1 - 1, m1[-1])
        m2_down = np.append(m2, m2[-1] - 1)
        v1 = np.append(_c_minus(l1, m1), 0)
        v2 = np.append(0, _c_minus(l2, m2))
    elif m1[0] == -l1 and m2[-1] > -l2:
        # support is constant
        m1_down = m1
        m2_down = m2 - 1
        v1 = np.append(_c_minus(l1, m1[1:]), 0)
        v2 = _c_minus(l2, m2)
    elif m1[0] > -l1 and m2[-1] == -l2:
        # support is constant
        m1_down = m1 - 1
        m2_down = m2
        v1 = _c_minus(l1, m1)
        v2 = np.append(0, _c_minus(l2, m2[:-1]))
    else:
        # support decreases
        m1_down, m2_down = m1[:-1], m2[:-1] - 1
        v1 = _c_minus(l1, m1[1:])
        v2 = _c_minus(l2, m2[:-1])

    idx_down = np.stack((m1_down, m2_down))
    val_down = v1 + v2
    return BCOO(idx_down, val_down)


def _eigvecs(l1: int, l2: int, L: int, normalize: bool = True) -> list[BCOO]:
    """Return the 2L+1 generators |LM⟩ as a list of 3D BCOO arrays."""

    def embed_3d(vec: BCOO, M: int) -> BCOO:
        """Return a 3D coef in BCOO format with output spin M."""
        (m1, m2), val = vec
        m0 = np.full(m1.size, M)
        return BCOO(np.stack((m0, m1, m2), axis=0), val)

    def normalize_bcoo(vec: BCOO) -> BCOO:
        """Project a sparse eigenvector on the unit sphere."""
        idx, val = vec
        norm = np.sqrt(np.sum(val * val))
        return BCOO(idx, val / norm)

    vec = _max_spin_eigvec(l1, l2, L)
    basis = [embed_3d(vec, L)]
    for i in range(2 * L):
        M = L - i - 1
        vec = _ladder_down(l1, l2, vec)
        basis.append(embed_3d(vec, M))
    # sort by increasing M = -L...L
    if normalize:
        return [normalize_bcoo(vec) for vec in basis[::-1]]
    return basis[::-1]
