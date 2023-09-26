# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Aliased object classes"""

from typing import Optional, TYPE_CHECKING
from .exceptions import LibraryError

if TYPE_CHECKING:
    from arraylias.alias import Alias


class AliasedModule:
    """Aliased library module.

    This is used to alias a module path for registered libraries of
    an :class:`.Alias`, and as the base module for automatic dispatch
    of an :class:`.Alias`. Use attributes access submodules or
    functions within this aliased module.
    """

    def __init__(self, alias: "Alias", lib: str, path: Optional[str] = None):
        self._alias = alias
        self._lib = lib
        self._path = path or ""
        self._cached = []

    def cache_clear(self):
        """Clear cached attributes"""
        self._cached = [delattr(self, attr) for attr in self._cached]

    def __repr__(self):
        label = "auto" if self._lib == "auto" else f"lib={repr(self._lib)}"
        if self._path:
            # Remove trailing . from path
            label += f", path={repr(self._path)}"
        return f"<AliasedModule({label})>"

    def __getattr__(self, attr: str):
        # Cache attribute to speed up attribute based access on subsequent
        # calls for aliased modules
        path = f"{self._path}.{attr}" if self._path else attr
        obj = self._alias._dispatch((self._lib,), path)
        self._cached.append(attr)
        setattr(self, attr, obj)
        return obj


class AliasedPath(AliasedModule):
    """Aliased library path.

    This is used to alias an unspecified object that cannot be inferred
    as a module or ``Callable`` until it is evaluated.

    If this aliases a function it should be used via call method to
    dispatch to the called function.

    If this aliases a module it should be used via get attribute to
    return the :class:`.AliasedPath` to the specified sub module or module
    function.
    """

    def __init__(self, alias: "Alias", path: str):
        super().__init__(alias, lib="auto", path=path)

    def __repr__(self):
        return f"<AliasedPath: {repr(self._path)}>"

    def __call__(self, first_arg, *args, **kwargs):
        # Infer array lib from type of the first argument
        libs = self._alias._libs_from_type(type(first_arg)) or self._alias.infer_libs(first_arg)
        try:
            func = self._alias._dispatch(libs, self._path)
            return func(first_arg, *args, **kwargs)

        except LibraryError as ex:
            raise LibraryError(
                f"{type(first_arg)} is not a registered type for any registered array libraries."
            ) from ex

    def __getattr__(self, attr: str):
        # Check to avoid dispatching for dunder methods or objects
        # This to prevent errors that can occur when inspecting
        # AliasedPath in wrapper methods that treat it like a function
        # such as with jax.jit
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(f"AliasedPath cannot alias dunder attribute ({attr})")
        return super().__getattr__(attr)
