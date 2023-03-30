# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Register jax library for Dispatch"""
# pylint: disable=import-error


def register_jax(alias):
    """Register default implementation of JAX if installed.

    Args:
        alias (Alias): The alias dispatcher to register with.

    Returns:
        bool: True if jax is installed and was successfully registered.
    """
    try:
        import jax
        from jax.interpreters.xla import DeviceArray
        from jax.core import Tracer

        lib = "jax"

        if jax.__version__ >= "0.4.6":
            from jaxlib.xla_extension import ArrayImpl

            # pylint: disable = invalid-name
            JAX_TYPES = (DeviceArray, Tracer, ArrayImpl)
        else:
            # pylint: disable = invalid-name
            JAX_TYPES = (DeviceArray, Tracer)

        # Register jax types
        for atype in JAX_TYPES:
            alias.register_type(atype, lib)

        # Register jax numpy modules
        alias.register_module(jax.numpy, lib)

        # Jax doesn't implement a copy method, so we add one using the
        # jax numpy.array constructor which implicitly copies
        # pylint: disable=unused-variable
        @alias.register_function(lib=lib)
        def copy(array, order="K"):
            return jax.numpy.array(array, copy=True, order=order)

        # Register Jax linalg functions
        alias.register_module(jax.numpy.linalg, path="linalg", lib=lib)
        alias.register_module(jax.scipy.linalg, path="linalg", lib=lib)

        return True

    except ModuleNotFoundError:
        return False
