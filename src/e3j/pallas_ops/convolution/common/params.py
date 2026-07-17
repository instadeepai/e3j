"""Shared parameter container for Pallas message-passing convolution kernels."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from e3j.spaces import O3Space
from e3j.utils import options

FwdConfig = TypeVar("FwdConfig")
BwdConfig = TypeVar("BwdConfig")


@dataclass
class PallasMessagePassingConvolutionParams(Generic[FwdConfig, BwdConfig]):
    indices: np.ndarray
    values: np.ndarray
    layout: options.Layout
    x_space: O3Space
    y_space: O3Space
    z_space: O3Space
    fwd_config: FwdConfig | None = None
    bwd_config: BwdConfig | None = None
