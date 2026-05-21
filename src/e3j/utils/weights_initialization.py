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

import jax.numpy as np

from e3j.utils.options import LinearIndexwiseInitialization, LinearInitialization

_InitializationOption = str | LinearInitialization | LinearIndexwiseInitialization

_FAN_IN = (LinearInitialization.FAN_IN, LinearIndexwiseInitialization.FAN_IN, "FAN_IN")
_FAN_OUT = (
    LinearInitialization.FAN_OUT,
    LinearIndexwiseInitialization.FAN_OUT,
    "FAN_OUT",
)


def _get_weights_std_and_scaling(
    m_out: int,
    m_in: int,
    weights_normalization: _InitializationOption = "FAN_IN",
    rescale_gradients: bool = True,
    num_indices: int | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return (std, scale) pair for weight initialization of Linear modules.

    When `rescale_gradients` is True, normal initial weights are rescaled within
    the computation graph, effectively rescaling gradients by the standard deviation
    of final weights.

    Args:
        m_out: effective output multiplicities (may include channels in count with
            TRAILING_CHANNELS and LEADING_CHANNELS layout).
        m_in: effective input multiplicities (idem).
        weights_normalization: see :class:`e3j.utils.options.LinearIndexwiseInitialization`.
        rescale_gradients: whether to rescale gradients in the computation graph. The
            default is True.
        num_indices: optional number of indices for LinearIndexwise, used with "FAN_IN_FCTP"
            fan-in intialization where an additional `sqrt(num_indices)` factor is
            included to mimic `e3nn.FullyConnectedTensorProduct` against one-hot
            encoded species.

    Returns:
        A pair (std, scale) respectively passed as standard deviation of the
        normal initalizer, and as downstream scaling factor.
        Either (1, s) or (s, 1) with/without gradient scaling.
    """
    if m_out == 0 or m_in == 0:
        return (1.0, 1.0)

    if weights_normalization in _FAN_IN:
        alpha = 1 / m_in
    elif weights_normalization in _FAN_OUT:
        alpha = 1 / m_out
    elif weights_normalization in (
        LinearIndexwiseInitialization.FAN_IN_FCTP,
        "FAN_IN_FCTP",
    ):
        assert num_indices is not None, "FAN_IN_FCTP requires num_indices"
        alpha = 1 / (m_in * num_indices)
    else:
        raise ValueError(f"Unknown weights_normalization: {weights_normalization}")

    return (1.0, np.sqrt(alpha)) if rescale_gradients else (np.sqrt(alpha), 1.0)
