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

"""Coefficient helpers shared across Pallas kernels."""

from collections import defaultdict
from typing import NamedTuple, TypeAlias

import numpy as np


class ZiContribution(NamedTuple):
    """A single term contributing to output index `zi`.

    Attributes:
        xi: First x-input index of the contributing pair.
        yi: Second y-input index of the contributing pair.
        value: Coefficient weighting this `(xi, yi)` product.
    """

    xi: int
    yi: int
    value: float


class XiContribution(NamedTuple):
    """A single term contributing to the gradient of input index `xi`.

    Attributes:
        zi: Output index the contribution flows back from.
        yi: y-input index of the contributing pair.
        value: Coefficient weighting this `(zi, yi)` term.
    """

    zi: int
    yi: int
    value: float


#: Contributions grouped by output index `zi`, sorted by key.
CoefByZi: TypeAlias = dict[int, list[ZiContribution]]

#: Contributions grouped by x-input index `xi`, sorted by key.
CoefByXi: TypeAlias = dict[int, list[XiContribution]]


def group_coef_by_zi(indices: np.ndarray, values: np.ndarray) -> CoefByZi:
    """Group coefficients by z index for static unrolling.

    Args:
        indices: (nnz, 3) COO index array; each row is (zi, xi, yi).
        values: (nnz,) coefficient values aligned with indices.

    Returns:
        Mapping `zi -> [ZiContribution(xi, yi, value), ...]` of
        contributions, sorted by `zi` for deterministic iteration.
    """
    by_zi: CoefByZi = defaultdict(list)
    for (zi, xi, yi), v in zip(indices.tolist(), values.tolist()):
        by_zi[zi].append(ZiContribution(xi, yi, float(v)))
    return dict(sorted(by_zi.items()))


def group_coef_by_xi(indices: np.ndarray, values: np.ndarray) -> CoefByXi:
    """Group coefficients by x-input index for dx accumulation.

    Args:
        indices: `(nnz, 3)` COO index array; each row is `(zi, xi, yi)`.
        values: `(nnz,)` coefficient values aligned with `indices`.

    Returns:
        Mapping `xi -> [XiContribution(zi, yi, value), ...]` of
        contributions, sorted by `xi` for deterministic iteration.
    """
    by_xi: CoefByXi = defaultdict(list)
    for (zi, xi, yi), v in zip(indices.tolist(), values.tolist()):
        by_xi[xi].append(XiContribution(zi, yi, float(v)))
    return dict(sorted(by_xi.items()))
