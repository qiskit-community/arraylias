# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Register tensorflow library for Dispatch"""
# pylint: disable=import-error


def register_tensorflow(alias):
    """Register default implementation of Tensorflow if installed.

    Args:
        alias (Alias): The alias dispatcher to register with.

    Returns:
        bool: True if tensorflow is installed and was successfully registered.
    """
    try:
        import tensorflow as tf
        import tensorflow.experimental.numpy as tnp

        lib = "tensorflow"

        # Register Tensor type
        alias.register_type(tf.Tensor, lib)

        # Register Tensorflow modules with preference given to the
        # numpy module
        alias.register_module(tnp, lib)
        alias.register_module(tf, lib)
        alias.register_module(tf.math, lib)

        return True

    except ModuleNotFoundError:
        return False
