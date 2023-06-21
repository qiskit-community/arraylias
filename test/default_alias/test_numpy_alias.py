# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""numpy_alias tests"""

import unittest
import numpy as np
import tensorflow as tf
from arraylias import numpy_alias

unp = numpy_alias()()


try:
    import jax.numpy as jnp
except ImportError:
    pass


class TestNumpyBase(unittest.TestCase):
    def setUp(self):
        self.arr = self.array([1.0, 2.0, 3.0, 4.0])
        self.arr_2d = self.array([[1.0, 2.0], [2.0, 1.0]])

    def array(self, arr):
        return np.array(arr)

    def test_sin(self):
        arr_unp = unp.sin(self.arr)
        self.assertTrue(type(self.arr) == type(arr_unp))
        self.assertTrue(unp.allclose(np.sin(self.arr), arr_unp))

    def test_eig(self):
        w_unp, v_unp = unp.linalg.eig(self.arr_2d)
        w, v = np.linalg.eig(self.arr_2d)
        self.assertTrue(unp.allclose(w, w_unp))
        self.assertTrue(unp.allclose(v, v_unp))


class TestJax(TestNumpyBase):
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

    def array(self, arr):
        return jnp.array(arr)


class TestTensorflow(TestNumpyBase):
    @classmethod
    def setUpClass(cls):
        # skip tests of tensorflow not installed
        try:
            # pylint: disable=import-outside-toplevel
            import tensorflow as tf
        except Exception as err:
            raise unittest.SkipTest("Skipping tensorflow tests.") from err

    def array(self, arr):
        return tf.constant(arr)

    def test_sin(self):
        arr_unp = unp.sin(self.arr)
        self.assertTrue(type(self.arr) == type(arr_unp))
        self.assertTrue(unp.allclose(np.sin(self.arr), arr_unp.numpy()))

    def test_eig(self):
        # skip because tensorflow.linalg is not registered by numpy_alias
        pass
