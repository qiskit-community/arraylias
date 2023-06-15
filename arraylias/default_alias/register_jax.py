# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Register jax library for Dispatch"""
# pylint: disable=import-error


def register_jax_types(alias):
    """Register JAX array types if installed.

    Args:
        alias (Alias): The alias dispatcher to register with.

    Returns:
        bool: True if jax is installed and was successfully registered.
    """
    try:
        import jax
        from jax.core import Tracer

        # pylint: disable = invalid-name
        JAX_TYPES = (Tracer, jax.Array)
        if jax.__version__ <= "0.4.10":
            JAX_TYPES += (jax.interpreters.xla.DeviceArray,)

        # Register jax types
        for atype in JAX_TYPES:
            alias.register_type(atype, "jax")

        return True

    except ModuleNotFoundError:
        return False


def register_jax_numpy(alias):
    """Register jax.numpy if JAX is installed.

    Args:
        alias (Alias): The alias dispatcher to register with.

    Returns:
        bool: ``True`` if JAX is installed and was successfully registered, otherwise ``False``.
    """
    if register_jax_types(alias):
        import jax

        # Register jax numpy modules
        alias.register_module(jax.numpy, lib="jax")

        # Jax doesn't implement a copy method, so we add one using the
        # jax numpy.array constructor which implicitly copies
        # pylint: disable=unused-variable
        @alias.register_function(lib="jax")
        def copy(array, order="K"):
            return jax.numpy.array(array, copy=True, order=order)

        # Register Jax linalg functions
        alias.register_module(jax.numpy.linalg, lib="jax", path="linalg")

        return True

    return False


def register_jax_scipy(alias):
    """Register jax.scipy if JAX is installed.

    Args:
        alias (Alias): The alias dispatcher to register with.

    Returns:
        bool: ``True`` if JAX is installed and was successfully registered, otherwise ``False``.
    """
    if register_jax_types(alias):
        import jax

        alias.register_module(jax.scipy, lib="jax")

        return True
    return False
