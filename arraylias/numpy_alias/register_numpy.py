# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Register numpy library for Dispatch"""
# pylint: disable=import-error


def register_numpy(alias, register_numbers=True, prefer_scipy=False):
    """Register default implementation of numpy and scipy if installed.

    Args:
        alias (Alias): The alias dispatcher to register with.
        register_numbers (bool): If True register python scalar number types
                                 int, float, and complex as numpy array types
                                 for aliasing (Default: True).
        prefer_scipy (bool): If set to True the SciPy linear algebra module will
                             be preferred over the NumPy linear algebra module
                             for aliased functions in both modules. If False SciPy
                             will only be used for functions not in NumPy.
                             (Default: False).

    Returns:
        bool: True if numpy is installed and was successfully registered.
    """
    try:
        import numpy

        lib = "numpy"

        # Register numpy ndarray
        alias.register_type(numpy.ndarray, lib)
        alias.register_type(numpy.number, lib)

        if register_numbers:
            # Register scalar number types
            alias.register_type(int, lib)
            alias.register_type(float, lib)
            alias.register_type(complex, lib)

        # Register numpy lib module
        alias.register_module(numpy)

        # Add asarray and array as defaults for unregistered types
        alias.register_default(numpy.array, path="array")
        alias.register_default(numpy.asarray, path="asarray")

        try:
            import scipy.linalg

            alias.register_module(numpy.linalg, path="linalg", lib=lib)
            alias.register_module(scipy.linalg, path="linalg", lib=lib, prefer=prefer_scipy)

        except ModuleNotFoundError:
            pass

        return True

    except ModuleNotFoundError:
        return False
