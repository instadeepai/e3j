import e3nn_jax as e3nn
import numpy as np
import pytest
from numpy.testing import assert_allclose

from e3j.utils.clebsch_gordan import clebsch_gordan


@pytest.mark.parametrize(
    "l0,l1,l2",
    [(2, 2, 0), (2, 0, 2), (1, 1, 0)],
    # [(1, 1, 1)]
    # [(0, 2, 2), (2, 2, 0), (2, 0, 2), (1, 2, 1), (1, 1, 1), (2, 3, 1), (3, 2, 4)]
)
def test_clebsch_gordan(l0: int, l1: int, l2: int):
    expect = e3nn.clebsch_gordan(l0, l1, l2)
    kij, ckij = clebsch_gordan(l0, l1, l2)
    (k, i, j) = kij
    k, i, j = k, i, j
    result = np.zeros(expect.shape)
    result[k, i, j] = ckij
    try:
        assert_allclose(expect, result)
    except AssertionError as err:
        print("\n", l1, "x", l2, "->", l0)
        print("\ne3nn:\n", expect)
        print("\ne3j:\n", result)
        raise (err)
