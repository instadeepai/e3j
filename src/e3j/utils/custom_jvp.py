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
from typing import Callable

import jax


class CustomJVP:
    """Register custom JVP rules for bound callables.

    The `CustomJVP` class implements the descriptor protocol to allow
    for the definition of `jax.custom_jvp` rules on bound methods,
    e.g. it does morally:

        self.__call__ = jax.custom_jvp(self.__call__)

    Note the bound callable `self.__call__` actually resolves to
    `cls.__call__.__get__(obj)`, and that Python descriptors are
    precisely there to customize the behaviour of method binding.
    """

    def __init__(self, method: Callable, jvp: str = "_custom_jvp"):
        self._method = method
        self._jvp_name = jvp

    def __set_name__(self, objtype, name):
        self._name_raw = name
        self._name = "_jvp_" + name

    def __get__(self, obj, objtype=None):
        if obj is not None:
            # Return cached `jax.custom_jvp` instance
            if hasattr(obj, self._name):
                return getattr(obj, self._name)
            # Decorate bound method with `jax.custom_jvp`
            f_obj_raw = functools.partial(self._method, obj)
            f_obj = jax.custom_jvp(f_obj_raw)
            f_obj.defjvp(getattr(obj, self._jvp_name))
            return f_obj
        else:
            # Return unbound method
            return self._method
