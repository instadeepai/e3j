Overview
========

The `e3j`_ package implements Euclid-equivariant operations for JAX,
it may be used as a replacement for `e3nn_jax`_.
As an equivariant tensor network library, it is composed of the following basic
building blocks:

* `Spherical harmonics <api/_core/e3j.core.Harmonics.html>`_
* `Linear transforms <api/_linen/e3j.linen.Linear.html>`_
* `Tensor products <api/_core/e3j.core.TensorProduct.html>`_

Most `e3j`_ modules expose a `source` and `target` property,
describing the input and output representations, which are typically
of type :class:`e3j.spaces.o3.O3Array`_

Currently, parameterized operations are only integrated within the `flax.linen`_
framework for JAX neural networks, within the `e3j.linen <api/linen.html>`_ submodule.

.. _handbook: handbook.html

.. _flax.linen: https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/index.html

Spherical Harmonics
^^^^^^^^^^^^^^^^^^^

Spherical harmonic polynomials $Y^l_m(\vec r)$ play a central role in quantum
chemistry, as they provide the generating basis of `atomic orbitals`_.

.. _atomic orbitals: https://wikipedia.org/wiki/atomic_orbitals

Within `e3j`_, spherical harmonics are described as actual polynomials,
sometimes called *solid harmonics*. Each harmonic polynomial $Y^l_m$ is
of degree $l$, and $m$ is a relative integer in the range
$[-l, l]$.

For now, one may simply think of them as
orthonormal generators for the space of functions on the sphere, or
"spherical activation functions".
They are widely used in equivariant GNNs to encode edge directions
(jointly used with RBFs for the radial encoding of edge lengths).

For more background on harmonic polynomials, see the `handbook`_.
A remarkable fact is that they provide the generators for all
possible representations of $SO_3$ (group of rotations,
translations and reflections exluded), which in physicist notation may
be written:

$$|lm \rangle = Y^l_m$$

Example
*******

.. code:: python

    >>> import jax.numpy as np
    >>> import e3j
    >>> Ylm = e3j.core.Harmonics(3)

    ### Evaluate on a 3D-point cloud
    >>> r = np.linspace(0, 1, 30).reshape((10, 3))
    >>> Ylm(r).shape
    (10,)

    ### Each degree l spans a (2l+1)-dimensional subspace
    >>> monomials = Ylm.polynomial.monomials
    >>> np.sum(monomials.exp, axis=-1)
    Array([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3], dtype=int32)


Linear Transforms
^^^^^^^^^^^^^^^^^

While harmonic polynomials provide the $|lm\rangle$ generators of $SO_3$ representations,
multiple copies of a given representation may be concatenated. This leads to the
`e3nn`_-like string description of representations (called *irreps*, to be understood as
*list of* irreducibles!):

.. code:: python

    >>> rep = e3nn.Irreps("64x0e + 32x1o + 16x2e")
    ### dimension = sum(mul * (2l + 1))
    >>> rep.dim == 64 + 32 * (3) + 16 * (5)
    True

which means that `rep` is a representation obtained
by concatenating 64 scalars (degree-0 polynomials), 32 vectors (degree-1 polynomials spanned
by $x,y,z$) and 16 degree-2 harmonic polynomials. Assuming the parity
`e|o` always matches that of $(-1)^l$, coordinates in `rep` could be labeled
with only three indices:

$${\bf x}=({\bf x}_{k,l,m}) \quad{\rm with}\quad
k\leq K_l,\;l \leq l_{max},\;-l\leq m \leq l.$$

The numbers of copies are called *multiplicities*, which could be thought of as channel dimensions
(especially when the multiplicity is constant over $l$, i.e. $K_l = K$).

*Linear equivariant functions* consist of a fundamental building blocks of equivariant neural
networks. They simply mix the multiplicities together via a (learnable) weight matrix,
without transforming the content of inputs that only depends on the spin $m$.
More precisely, a linear transformation may be written

$${\bf y}_{k',l,m} = \sum_{k \leq K_l} W^l_{k', k} \cdot {\bf x}_{k,l,m}.$$

Example
*******

.. code:: python

    >>> lin = e3j.linen.Linear(
    ...     "64x0e + 32x1o + 16x2e",
    ...     "16x0e + 8x1o + 4x2e",
    ...)
    ### Block-diagonal weight matrices
    >>> [block.shape for block in lin.blocks]
    [(64, 16), (32, 8), (16, 4)]

    ### Mix channels of random inputs
    >>> rng = random.key(123)
    >>> x = random.randn(rng, (10, lin.source.dim))
    >>> params = lin.init(rng, x)
    >>> y = lin.apply(params, x)
    >>> y.shape == (10, lin.target.dim)
    True

Tensor Products
^^^^^^^^^^^^^^^

Tensor products of representations, i.e. *bilinear transforms*
of equivariant inputs, are crucial to describe coupling and interactions.

The Clebsch-Gordan rules provide a canonical basis for the tensor product
$\bf z = x \otimes y$ of two equivariant arrays. They consist in
rewriting or *developing* ${\bf z}$ as:

$${\bf z}_{kk', L, M} = \sum_{\substack{l,m\\l',m'}}
C^{LM}_{lm,l'm'} \cdot {\bf x}_{k,l,m} \cdot {\bf y}_{k',l',m'},$$

where $kk' \leq K_l K'_l$, see the `handbook`_ for more details.
The tensor product may also be performed channel-wise by stacking multiplicities
as a batch dimension. The above operation could be implemented as a
dense `einsum` on two axes of `x` and `y`, against a 3D-array `C`.

In `e3j`_, coefficients are described by sparse arrays, via
the `jax.experimental.sparse.BCOO` subclass of `jax.Array` for now.
This lets us account for the fine-grained sparsity pattern (~1%)
of Clebsch-Gordan coefficients.
Melding the representation coordinates in single opaque indices,
this means `e3j`_ basically computes the tensor product of
`x` and `y` as:

.. code:: python

    k, i, j = coef.indices.T
    z = zeros.at[:,k].add(coef.values * x[:,i] * y[:,j])

.. note::

    The default `jax.lax.scatter_add` scales quite bad to
    large batch sizes (> 10k).

    Use the `e3j_ops` CUDA kernels on GPU instead for critical operations.
    See :class:`e3j.utils.config.Config` for implementation options.

Example
*******

.. code:: python

    >>> otimes = e3j.core.TensorProduct(
    ...     source=("0e+1o+2e", "1o"),
    ...     target=None, # no output filter
    ... )
    ### Output dimension comes first
    >>> otimes.coef.shape
    (27, 9, 3)

    ### Bilinear pairing of inputs
    >>> from jax import random
    >>> rng = random.key(314)
    >>> x = random.normal(rng, (10, 1+3+5))
    >>> y = random.normal(rng, (10, 3))
    >>> z = otimes(x, y)
    >>> z.shape
    (10, 27)


.. _e3j: https://github.com/instadeepai/e3j

.. _e3nn: https://e3nn.org/
.. _e3nn_jax: https://e3nn-jax.readthedocs.io/en/latest
.. _cuequivariance: https://github.com/NVIDIA/cuEquivariance
.. _openequivariance: https://github.com/vbharadwaj-bk/OpenEquivariance
.. _e3x: https://e3x.readthedocs.io/stable/
