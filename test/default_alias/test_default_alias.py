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
"""numpy_alias tests"""

import unittest
import numpy as np
import scipy
import tensorflow as tf
from arraylias import numpy_alias, scipy_alias

unp = numpy_alias()()
usp = scipy_alias()()


try:
    import jax.numpy as jnp
    from jax import jit, grad, vmap
except ImportError:
    pass


class NumpyBase:
    """Base class for testing numpy dispatching."""

    def array(self, arr):
        """convert list to numpy.array
        Args:
            arr (list): The input array of List.

        Returns:
            Numpy.array converted from the input
        """
        return np.array(arr)


class JaxBase:
    """Base class for testing jax dispatching."""

    @classmethod
    def setpClass(cls):
        # skip tests of JAX not installed
        try:
            # pylint: disable=import-outside-toplevel
            import jax

            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
        except Exception as err:
            raise unittest.SkipTest("Skipping jax tests.") from err

    def array(self, arr):
        """convert list to jax.numpy.array
        Args:
            arr (list): The input array of List.

        Returns:
            Jax.numpy.array converted from the input
        """
        return jnp.array(arr)


class TensorflowBase:
    """Base class for testing tensorflow dispatching."""

    @classmethod
    def setUpClass(cls):
        # skip tests of tensorflow not installed
        try:
            # pylint: disable=reimported, unused-import
            import tensorflow
        except Exception as err:
            raise unittest.SkipTest("Skipping tensorflow tests.") from err

    def array(self, arr):
        """convert list to tensorflow.Tensor
        Args:
            arr (list): The input array of List.

        Returns:
            Tensorflow.Tensor converted from the input
        """
        return tf.constant(arr)


class TestNumpyAlias(unittest.TestCase, NumpyBase):
    """Test outputs when the inputs are numpy array of numpy_alias."""

    def setUp(self):
        self.arr = self.array([1.0, 2.0, 3.0, 4.0])
        self.arr_2d = self.array([[1.0, 2.0], [2.0, 1.0]])

    def test_sin(self):
        """Test outputs of numpy.sin and sin func by numpy_alias are identical."""
        arr_unp = unp.sin(self.arr)
        self.assertTrue(isinstance(arr_unp, type(self.arr)))
        self.assertTrue(unp.allclose(np.sin(self.arr), arr_unp))

    def test_eig(self):
        """Test outputs of numpy.linalg.eig and linalg.eig func by numpy_alias are identical."""
        w_unp, v_unp = unp.linalg.eig(self.arr_2d)
        w, v = np.linalg.eig(self.arr_2d)
        self.assertTrue(unp.allclose(w, w_unp))
        self.assertTrue(unp.allclose(v, v_unp))


class TestJaxAlias(JaxBase, TestNumpyAlias):
    """Test outputs when the inputs are jax numpy array of numpy_alias."""

    def test_direct_jit(self):
        """Test jit directly on dispatched function."""
        jit_sin = jit(unp.sin)
        self.assertTrue(unp.allclose(unp.sin(1.42), jit_sin(1.42)))

    def test_direct_grad(self):
        """Test grad directly on dispatched function."""
        grad_sin = jit(grad(unp.sin))
        self.assertTrue(unp.allclose(unp.cos(1.23), grad_sin(1.23)))

    def test_direct_vmap(self):
        """Test vmap directly on dispatched function."""
        vmap_sin = vmap(unp.sin)
        self.assertTrue(unp.allclose(unp.sin([1.23, 1.423]), vmap_sin(jnp.array([1.23, 1.423]))))


class TestTensorflowAlias(TensorflowBase, TestNumpyAlias):
    """Test Outputs when the inputs are tensorflow array of numpy_alias"""

    def test_sin(self):
        """Test outputs of numpy.sin and tensorflow.experimental.numpy.sin func
        by numpy_alias are identical.
        """
        arr_unp = unp.sin(self.arr)
        self.assertTrue(isinstance(arr_unp, type(self.arr)))
        self.assertTrue(unp.allclose(np.sin(self.arr), arr_unp.numpy()))

    def test_eig(self):
        # skip because tensorflow.experimental.numpy.linalg is not registered
        # by numpy_alias
        pass


class TestScipyAlias(unittest.TestCase, NumpyBase):
    """Test Outputs when the inputs are numpy array of scipy_alias"""

    def setUp(self):
        self.arr_2d = self.array([[1.0, 2.0], [2.0, 1.0]])

    def test_eigh(self):
        """Test outputs of scipy.linalg.eigh and linalg.eigh func
        by numpy_alias are identical.
        """
        w_usp, v_usp = usp.linalg.eigh(self.arr_2d)
        w, v = scipy.linalg.eigh(self.arr_2d)
        self.assertTrue(usp.allclose(w, w_usp))
        self.assertTrue(usp.allclose(v, v_usp))


class TestJaxScipyAlias(JaxBase, TestScipyAlias):
    """Test outputs when the inputs are jax.numpy array of scipy_alias."""

    pass


class TestTensorflowScipyAlias(TensorflowBase, TestScipyAlias):
    """Test outputs when the inputs are tensorflow array of scipy_alias."""

    def setUp(self):
        self.arr_2d = tf.constant([[1.0, 2.0], [2.0, 1.0]])
