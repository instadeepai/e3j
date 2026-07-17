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


class cache:
    """A writable cache descriptor.

    Should behave like `functools.cache` with additional setter suport.

    Use as a decorator around the getter method or property:

    .. code-block:: python

        class TensorProduct:

            @cache
            def coef(self) -> np.ndarray: ...

    Note
    ----
    This class is intended to improve support of `flax.linen` and automatic
    differentiation, by avoiding nested function closures or undesired copies,
    when stopping gradients on cached coefficients indices (dtype checks may
    fail even with `jax.lax.stop_gradient`).
    """

    def __init__(self, getter):
        self.getter = getter
        self.name = getter.__name__
        self._name = "_" + self.name
        self.__doc__ = getter.__doc__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        cached = getattr(obj, self._name)
        if cached is not None:
            return cached
        else:
            attr = self.getter(obj)
            setattr(obj, self._name, attr)
            return attr

    def __set_name__(self, obj, name):
        self.name = name
        self._name = "_" + name
        setattr(obj, self._name, None)

    def __set__(self, obj, value):
        setattr(obj, self._name, value)
