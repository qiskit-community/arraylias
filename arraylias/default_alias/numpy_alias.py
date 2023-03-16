# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Default NumPy alias"""

from arraylias.alias import Alias
from .register_numpy import register_numpy
from .register_jax import register_jax_numpy
from .register_tensorflow import register_tensorflow_numpy


def numpy_alias(register_numbers=True) -> Alias:
    """Return a pre-configured Alias with numpy like syntax.

    This includes registered libs ``numpy``, ``jax``, and ``tensorflow``
    if the respective packages are installed.

    Args:
        register_numbers (bool): If True register python scalar number types
            int, float, and complex as numpy array types for aliasing
            (Default: True).

    Returns:
        A numpy-syntax :class:`.Alias`.
    """
    alias = Alias()
    register_numpy(alias, register_numbers=register_numbers)
    register_jax_numpy(alias)
    register_tensorflow_numpy(alias)
    return alias
