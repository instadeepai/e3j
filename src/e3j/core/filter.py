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

from e3j.spaces import O3Space


class Filter:
    """Keep only features of selected momenta and parities."""

    def __init__(self, source, target_filter, axis: int = -1):
        self.source = O3Space(source)
        self.axis = axis
        # Filter source irreducible blocks
        indices = []
        target = []
        keep_ir_out = set(ir for _, ir in O3Space(target_filter))
        for (m, ir), slc in zip(self.source, self.source.slices()):
            if ir not in keep_ir_out:
                continue
            target.append((m, ir))
            indices.append(np.arange(slc.start, slc.stop))
        self.target = O3Space(target)
        self.indices = np.concat(indices)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Filter out features not in `target`."""
        return np.take(x, self.indices, self.axis)
