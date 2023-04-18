# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Dispatcher class"""

import functools
from typing import Optional, Union, Callable, Tuple
from types import ModuleType, FunctionType

from arraylias.aliased import AliasedModule, AliasedPath
from arraylias.exceptions import AliasError, LibraryError


_AUTOLIB = ("auto",)


@functools.wraps(functools.lru_cache)
def method_lru_cache(maxsize: Optional[int] = 128, typed: bool = False) -> Callable:
    """Least-recently-used cache decorator for methods.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(3.0) and f(3) will be treated as distinct calls with
    distinct results.

    Arguments to the cached function must be hashable.

    See:  https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)
    """

    # Construct the functools LRU cache decorator
    lru_cache = functools.lru_cache(maxsize=maxsize, typed=typed)

    def cache_method_decorator(method: Callable) -> Callable:
        """Decorator for caching method.

        Args:
            method: A method to cache.

        Returns:
            The wrapped cached method.
        """

        def _get_cache(self):
            """Retrieve the cache"""
            try:
                return getattr(self, "_method_lru_cache")
            except AttributeError:
                setattr(self, "_method_lru_cache", {})
                return getattr(self, "_method_lru_cache")

        @functools.wraps(method)
        def _cached_method(self, *args, **kwargs):
            cache = _get_cache(self)
            key = method.__name__
            try:
                # Return the previously cached function
                meth = cache[key]
            except KeyError:
                # Create new cached function and return it
                meth = cache[key] = lru_cache(functools.partial(method, self))

            return meth(*args, **kwargs)

        return _cached_method

    return cache_method_decorator


class Alias:
    """Array library aliasing class.

    This class aliasing of multiple libraries and uses a  single-dispatch
    mechanism to dispatch to the correct library function based on the
    type of the first function argument.

    Full library sub-module paths are supported, and libraries can be
    extended by registering custom modules, and modules can be extended by
    registering custom functions.

    See the Arraylias :ref:`User Guide <arraylias-userguide>` for details
    on building and using an Alias.
    """

    __slots__ = [
        "_libs",
        "_types",
        "_modules",
        "_functions",
        "_fallbacks",
        "_defaults",
        "_method_lru_cache",
    ]

    def __init__(self):
        # Set of registered library names for dispatching
        # Note that we use a dict instead of a set so that the order backends
        # are registered will be preserved.
        self._libs = {}

        # Map of library types to library names for dispatching
        self._types = {}

        # Map of library names to list of modules for checking for functions
        self._modules = {}

        # Map of library names to function map for dispatching names
        # to specific functions for that library
        self._functions = {}

        # Map of function names to fallback implementations if one is
        # not provided by the registered library
        self._fallbacks = {}

        # Map of function names to default implementation if the aliased
        # function can't match the array type to a registered type
        self._defaults = {}

        # Cache for LRU cached methods
        self._method_lru_cache = {}

    def __call__(
        self, path: Optional[str] = None, like: Optional[Union[str, type, any]] = None
    ) -> Union[AliasedModule, AliasedPath, Callable]:
        """Return aliased library path.

        Args:
            path: A function or module path to alias. If ``None`` the base module
                  will be returned.
            like: Infer library based on type or object and statically dispatch to
                  that libraries function or module.

        Returns:
            The aliased module or function.

        Raises:
            ValueError: if a specific array library is specified in both path
                        and the like kwarg is used.
        """
        if like is not None:
            libs = self.infer_libs(like)
        else:
            libs = _AUTOLIB
        return self._dispatch(libs, path)

    @method_lru_cache(1)
    def registered_libs(self) -> Tuple[str, ...]:
        """Return all registered library names for dispatching."""
        return tuple(self._libs)

    @method_lru_cache(1)
    def registered_types(self) -> Tuple[type, ...]:
        """Return all registered types for dispatching."""
        return tuple(self._types)

    def register_function(
        self,
        func: Optional[Callable] = None,
        lib: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Optional[Callable]:
        """Register an array function for aliasing.

        .. note::
            This method makes a copy of the function being registered, so
            if the definition is modified you must re-register the function.

        Args:
            func: The function to dispatch to for the specified array library.
                If None this will return a decorator to apply to a function.
            lib: Optional, a name string to identify the array library.
                 If ``None`` this will be set as the base module name of the
                 arrays module.
            path: Optional, the path for dispatching to this function. If ``None``
                  the name of the input function will be used.

        Returns:
            If func is None returns a decorator for registering a function.
            Otherwise returns None.
        """
        decorator = self._register_function_decorator(lib=lib, path=path)
        if func is None:
            return decorator
        return decorator(func)

    def register_fallback(
        self,
        func: Optional[Callable] = None,
        path: Optional[str] = None,
    ) -> Optional[Callable]:
        """Register a fallback array function for aliasing.

        This function will be used if no matches for this name are found in a
        registered array libraries modules and functions.

        Args:
            func: The function to dispatch to for the specified array library.
                  If None this will return a decorator to apply to a function.
            path: Optional, the path for dispatching to this function. If None
                  the name of the input function will be used.

        Returns:
            If func is None returns a decorator for registering a function.
            Otherwise returns None.
        """
        decorator = self._register_fallback_decorator(path=path)
        if func is None:
            return decorator
        return decorator(func)

    def register_default(
        self,
        func: Optional[Callable] = None,
        path: Optional[str] = None,
    ) -> Optional[Callable]:
        """Register a default function alias for un-registered types.

        This function will be used by aliased functions for unregistered array
        types where the array library cannot be inferred.

        Args:
            func: The function to dispatch to for the specified array library.
                  If None this will return a decorator to apply to a function.
            path: Optional, the path for dispatching to this function. If None
                  the name of the input function will be used.

        Returns:
            If func is None returns a decorator for registering a function.
            Otherwise returns None.
        """
        decorator = self._register_default_decorator(path=path)
        if func is None:
            return decorator
        return decorator(func)

    def register_module(
        self,
        module: ModuleType,
        lib: Optional[str] = None,
        path: Optional[str] = None,
        prefer: bool = False,
    ):
        """Register a module for looking up array functions.

        Args:
            module: A module, namespace, or class to look for attributes
                    corresponding to the dispatched function name.
            lib: Optional, a name string to identify the array library.
                 If ``None`` this will be set as the base module name of the
                 arrays module.
            path: Optional, the path for the module. If empty this module
                  will be added to the base path for the library.
            prefer: Prioritize searching this module before previously
                    registered modules for the current path (Default: ``False``).

        .. note::

            The order modules with the same path are added sets the priority
            for looking up dispatched functions, with the first match in a
            module being returned.
        """
        # Infer default lib from module name or path prefix
        if lib is None:
            lib = _lib_from_object(module)
        path = path or ""

        # Register library and paths
        self._register_lib(lib)
        self._register_paths(lib, path)

        # Register module
        mods = self._modules[lib]
        if path not in mods:
            mods[path] = [module]
        elif prefer:
            mods[path] = [module] + mods[path]
        else:
            mods[path].append(module)

        self.cache_clear()

    def register_type(
        self,
        array_type: type,
        lib: Optional[str] = None,
        prefer: bool = False,
    ):
        """Register an array type for dispatching array functions.

        Args:
            array_type: An array type to register for the array library.
            lib: Optional, a name string to identify the array library.
                 If None this will be set as the base module name of the
                 arrays module.
            prefer: prioritize this lib when dispatching on this type if the type
                    is registered to multiple libraries (Default: False).
        """
        if lib is None:
            lib = _lib_from_object(array_type)
        self._register_lib(lib)
        if array_type not in self._types:
            self._types[array_type] = (lib,)
        elif prefer:
            self._types[array_type] = (lib,) + self._types[array_type]
        else:
            self._types[array_type] = self._types[array_type] + (lib,)
        self.cache_clear()

    def infer_libs(
        self, obj: Union[str, type, any], allow_sequence: bool = True
    ) -> Tuple[str, ...]:
        """Infer the registered library name for an object.

        Args:
            obj: array object to check.
            allow_sequence: If True recursively check the element type of
                            list and tuple objects (Default: True).

        Returns:
            A tuple of library names registered for the input object type.
        """
        if obj is None:
            return tuple()

        if isinstance(obj, str):
            if obj in self._libs:
                return (obj,)
            return tuple()

        if isinstance(obj, type):
            return self._libs_from_type(obj)

        libs = self._libs_from_type(type(obj))
        if not libs and allow_sequence and isinstance(obj, (tuple, list)) and obj:
            libs = self.infer_libs(obj[0])

        return libs

    def cache_clear(self):
        """Clear cached dispatched calls."""
        # Clear LRU cached functions
        for func in self._method_lru_cache.values():
            func.cache_clear()
        self._method_lru_cache.clear()

    def _register_lib(self, lib: str):
        """Register an array library name.

        Args:
            lib: The name string to identify the array library.
        """
        if lib not in self._libs:
            self._libs[lib] = None
            self._functions[lib] = {}
            self._modules[lib] = {}

    def _register_paths(self, lib: str, path: str):
        """Register path tree for aliasing.

        Args:
            lib: The name string to identify the array library.
            path: the path to register.
        """
        # Register intermediate paths so that path traversal during
        # dispatching can work similar to actual modules
        mods = self._modules[lib]
        split_path = path.split(".")
        if len(split_path) > 1:
            accum_paths = [".".join(split_path[: i + 1]) for i in range(len(split_path))]
        else:
            accum_paths = split_path
        for sub_path in accum_paths:
            if sub_path not in mods:
                mods[sub_path] = []

    def _register_function_decorator(
        self, lib: Optional[str] = None, path: Optional[str] = None
    ) -> Callable:
        """Return a decorator to register a function.

        Args:
            lib: Optional, the name string to identify the array library.
                 If None the base function module will be used.
            path: Optional, the aliased path for this function.
                  If ``None`` the function name will be used.

        Returns:
            A function decorator to register a function if dispatched_function
            is None.
        """

        def decorator(func):
            if lib is None:
                func_lib = _lib_from_object(func)
            else:
                func_lib = lib
            if func_lib not in self._libs:
                raise LibraryError(f"Array library {func_lib} is not a registered library.")

            if path is None:
                func_path = self._trim_lib_from_path(func.__name__)
            else:
                func_path = path

            # Register sub paths if this is a module function
            split_path = func_path.rsplit(".", 1)
            if len(split_path) > 1:
                self._register_paths(func_lib, split_path[0])

            self._functions[func_lib][func_path] = func
            self.cache_clear()
            return func

        return decorator

    def _register_fallback_decorator(self, path: Optional[str] = None) -> Callable:
        """Return a decorator to register a fallback function.

        Args:
            path: Optional, the aliased path to this function.
                  If None the function name will be used.

        Returns:
            A function decorator to register a function if dispatched_function
            is None.
        """

        def decorator(func):
            if path is None:
                func_path = self._trim_lib_from_path(func.__name__)
            else:
                func_path = path
            self._fallbacks[func_path] = func
            self.cache_clear()
            return func

        return decorator

    def _register_default_decorator(self, path: Optional[str] = None) -> Callable:
        """Return a decorator to register a default function.

        Args:
            path: Optional, the path for dispatching to this function.
                  If None the function name will be used.

        Returns:
            A function decorator to register a function if dispatched_function
            is None.
        """

        def decorator(func):
            if path is None:
                func_path = self._trim_lib_from_path(func.__name__)
            else:
                func_path = path
            self._defaults[func_path] = func
            self.cache_clear()
            return func

        return decorator

    @method_lru_cache(512)
    def _dispatch(
        self, libs: Tuple[str, ...], path: Union[str, None]
    ) -> Union[AliasedModule, AliasedPath, Callable]:
        # pylint: disable = too-many-return-statements
        # Check if path is a default function for unmatched library
        if not libs:
            if path in self._defaults:
                return self._defaults[path]
            raise LibraryError(f"No default function registered for {path}.")

        # Return base library module
        if not path:
            return AliasedModule(self, lib=libs[0])

        # Auto-dispatching path
        if libs == _AUTOLIB:
            return AliasedPath(self, path)

        # Static dispatching for given libs and path
        for lib in libs:
            dispatched = self._static_dispatch(lib, path)
            if dispatched is not None:
                return dispatched

        # Check if path has a registered fallback function
        if path in self._fallbacks:
            return self._fallbacks[path]

        # Couldn't find a matching function
        raise AliasError(
            f"Unable to resolve path '{path}' for array libraries '{libs}'"
            " and no fallback is registered."
        )

    def _static_dispatch(self, lib: str, path: Union[str, None]) -> Optional[Callable]:
        """Try static dispatching for given lib and path."""
        # Check if path is a registered function for module
        if path in self._functions[lib]:
            return self._functions[lib][path]

        # Check if path is a registered module
        modules = self._modules[lib]
        if path in modules:
            return AliasedModule(self, lib=lib, path=path)

        # Since we could not match with registered functions and modules
        # we search for path in registered modules
        split_path = path.split(".")
        accum_paths = [""] + [".".join(split_path[: i + 1]) for i in range(len(split_path) - 1)]

        # Find deepest registered module path for given path
        search_mods = None
        search_path = None
        for i, sub_path in enumerate(reversed(accum_paths)):
            if sub_path in modules:
                search_mods = modules[sub_path]
                search_path = split_path[-1 - i :]
                break

        # Try looking up based on sub module path
        if search_mods:
            for sub_mod in search_mods:
                try:
                    current_path = getattr(sub_mod, search_path[0])
                    for sub_path in search_path[1:]:
                        current_path = getattr(current_path, sub_path)
                    if isinstance(current_path, ModuleType):
                        return AliasedModule(self, lib=lib, path=path)
                    else:
                        return current_path
                except AttributeError:
                    pass

        # Static dispatch failed to match anything for this path and library
        return None

    @method_lru_cache()
    def _libs_from_type(self, obj_type: type) -> Tuple[str, ...]:
        """Return the library name of a type if registered"""
        if obj_type is type(None):
            return tuple()

        if obj_type in self._types:
            return self._types[obj_type]

        # Look via subclass
        for key, libs in self._types.items():
            if issubclass(obj_type, key):
                return libs

        return tuple()

    @method_lru_cache()
    def _trim_lib_from_path(self, path: Optional[str]) -> str:
        """Split lib from path string if present."""
        if path is None:
            return ""

        split = path.split(".", 1)
        if split[0] in self._libs:
            tail = "" if len(split) == 1 else split[1]
            return tail

        return path


@functools.lru_cache()
def _lib_from_object(obj: any) -> str:
    """Infer array library string from base module path of an object."""
    if isinstance(obj, ModuleType):
        modname = obj.__name__
    elif isinstance(obj, (type, FunctionType)):
        modname = obj.__module__
    else:
        modname = type(obj).__module__
    return modname.split(".", maxsplit=1)[0]
