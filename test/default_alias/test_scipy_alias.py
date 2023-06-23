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
"""scipy_alias tests"""

import unittest
import numpy as np
import scipy
import tensorflow as tf
from arraylias import scipy_alias

unp = scipy_alias()()

try:
    import jax.numpy as jnp
except ImportError:
    pass


class TestNumpyBase(unittest.TestCase):
    """Test Outputs when the inputs are numpy array of scipy_alias"""

    def setUp(self):
        self.arr_2d = np.array([[1.0, 2.0], [2.0, 1.0]])

    def test_eigh(self):
        """Test outputs of scipy.linalg.eigh and linalg.eigh func
        by numpy_alias are identical"""
        w_unp, v_unp = unp.linalg.eigh(self.arr_2d)
        w, v = scipy.linalg.eigh(self.arr_2d)
        self.assertTrue(unp.allclose(w, w_unp))
        self.assertTrue(unp.allclose(v, v_unp))


class TestJax(TestNumpyBase):
    """Test Outputs when the inputs are jax.numpy array of scipy_alias"""

    @classmethod
    def setUpClass(cls):
        # skip tests of JAX not installed
        try:
            # pylint: disable=import-outside-toplevel
            import jax

            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
        except Exception as err:
            raise unittest.SkipTest("Skipping jax tests.") from err

    def setUp(self):
        self.arr_2d = jnp.array([[1.0, 2.0], [2.0, 1.0]])


class TestTensorflow(TestNumpyBase):
    """Test Outputs when the inputs are tensorflow array of scipy_alias"""

    @classmethod
    def setUpClass(cls):
        # skip tests of tensorflow not installed
        try:
            # pylint: disable=reimported, unused-import
            import tensorflow
        except Exception as err:
            raise unittest.SkipTest("Skipping tensorflow tests.") from err

    def setUp(self):
        self.arr_2d = tf.constant([[1.0, 2.0], [2.0, 1.0]])
