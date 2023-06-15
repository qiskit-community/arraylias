# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Register tensorflow library for Dispatch"""
# pylint: disable=import-error


def register_tensorflow_numpy(alias):
    """Register tensorflow.numpy if TensorFlow is installed.

    Args:
        alias (Alias): The alias dispatcher to register with.

    Returns:
        bool: True if tensorflow is installed and was successfully registered.
    """
    try:
        import tensorflow as tf
        import tensorflow.experimental.numpy as tnp

        alias.register_type(tf.Tensor, "tensorflow")
        alias.register_module(tnp, "tensorflow")

        return True

    except ModuleNotFoundError:
        return False


def register_tensorflow_scipy(alias):
    """Register tensorflow.scipy.linalg if TensorFlow is installed.

    Args:
        alias (Alias): The alias dispatcher to register with.

    Returns:
        bool: ``True`` if TensorFlow is installed and was successfully registered, otherwise
        ``False``.
    """
    try:
        import tensorflow as tf

        alias.register_type(tf.Tensor, "tensorflow")
        alias.register_module(tf.linalg, lib="tensorflow", path="linalg")

        return True

    except ModuleNotFoundError:
        return False
