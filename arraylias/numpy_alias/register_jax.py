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
        from jax.random import PRNGKey, uniform, normal
        from jax.core import Tracer

        lib = "jax"

        if jax.__version__ > "0.4.10":
            # pylint: disable = invalid-name
            JAX_TYPES = (Tracer, jax.Array)
        else:
            from jax.interpreters.xla import DeviceArray

            # pylint: disable = invalid-name
            JAX_TYPES = (DeviceArray, Tracer, jax.Array)

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

        # JAX provides a jax.random module, so we add custom functions
        # which is similar with numpy.random module

        @alias.register_function(lib=lib)
        def random_seed(seed=None):
            # pylint: disable=global-variable-undefined
            global _JAX_RANDOM_KEY
            if seed is None:
                import random

                seed = random.randint(-(2**63), 2**63 - 1)
            _JAX_RANDOM_KEY = PRNGKey(seed)

        def random_key():
            # pylint: disable=global-variable-undefined
            global _JAX_RANDOM_KEY
            if "_JAX_RANDOM_KEY" not in globals() and "_JAX_RANDOM_KEY" not in locals():
                _JAX_RANDOM_KEY = None
            if _JAX_RANDOM_KEY is None:
                random_seed()
            _JAX_RANDOM_KEY, res_key = jax.random.split(_JAX_RANDOM_KEY)
            return res_key

        @alias.register_function(lib=lib)
        def random_normal(loc=0.0, scale=1.0, size=None):
            if size is None:
                size = ()
            return normal(random_key(), shape=size) * scale + loc

        @alias.register_function(lib=lib)
        def random_uniform(low=0.0, high=1.0, size=None):
            if size is None:
                size = ()
            return uniform(random_key(), shape=size, minval=low, maxval=high)

        # Register Jax linalg functions
        alias.register_module(jax.numpy.linalg, path="linalg", lib=lib)
        alias.register_module(jax.scipy.linalg, path="linalg", lib=lib)

        return True

    except ModuleNotFoundError:
        return False
