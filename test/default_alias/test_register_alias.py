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
import tensorflow as tf
import scipy
from arraylias.alias import Alias
from arraylias.default_alias.register_numpy import register_numpy, register_scipy
from arraylias.default_alias.register_jax import register_jax_numpy, register_jax_scipy
from arraylias.default_alias.register_tensorflow import (
    register_tensorflow_numpy,
    register_tensorflow_scipy,
)

try:
    import jax
    from jax.core import Tracer
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp
except ImportError:
    pass


class TestRegisterNumpy(unittest.TestCase):
    def setUp(self):
        self.unp = Alias()
        self.lib = "numpy"
        self.types = [np.ndarray, np.number, int, float, complex]
        self.module_path = {np: ""}

        self.register(self.unp)

    def register(self, alias):
        register_numpy(alias, register_numbers=True)

    def test_register_type(self):
        for t in self.types:
            self.assertTrue(t in self.unp._types)
            self.assertTrue(self.unp._types[t], self.lib)

    def test_register_module(self):
        for m in self.module_path.keys():
            self.assertTrue(m, self.unp._modules[self.lib][self.module_path[m]])


class TestRegisterScipy(TestRegisterNumpy):
    def setUp(self):
        self.unp = Alias()
        self.lib = "numpy"
        self.module_path = {scipy: ""}

        self.register(self.unp)

    def register(self, alias):
        register_scipy(alias, register_numbers=True)

    def test_register_type(self):
        pass


class TestRegisterJax(TestRegisterNumpy):
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
        self.unp = Alias()
        self.lib = "jax"
        self.types = [Tracer, jax.Array]
        self.module_path = {jax.numpy: "", jax.numpy.linalg: "linalg"}

        self.register(self.unp)

    def register(self, alias):
        register_jax_numpy(alias)


class TestRegisterJaxScipy(TestRegisterNumpy):
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
        self.unp = Alias()
        self.lib = "jax"
        self.module_path = {jax.scipy: ""}

        self.register(self.unp)

    def register(self, alias):
        register_jax_scipy(alias)

    def test_register_type(self):
        pass


class TestRegisterTensorflow(TestRegisterNumpy):
    @classmethod
    def setUpClass(cls):
        # skip tests of tensorflow not installed
        try:
            # pylint: disable=import-outside-toplevel
            import tensorflow as tf
        except Exception as err:
            raise unittest.SkipTest("Skipping tensorflow tests.") from err

    def setUp(self):
        self.unp = Alias()
        self.lib = "tensorflow"
        self.types = [tf.Tensor]
        self.module_path = {tnp: ""}

        self.register(self.unp)

    def register(self, alias):
        register_tensorflow_numpy(alias)


class TestRegisterTensorflowJax(TestRegisterNumpy):
    @classmethod
    def setUpClass(cls):
        # skip tests of tensorflow not installed
        try:
            # pylint: disable=import-outside-toplevel
            import tensorflow as tf
        except Exception as err:
            raise unittest.SkipTest("Skipping tensorflow tests.") from err

    def setUp(self):
        self.unp = Alias()
        self.lib = "tensorflow"
        self.types = [tf.Tensor]
        self.module_path = {tf.linalg: "linalg"}

        self.register(self.unp)

    def register(self, alias):
        register_tensorflow_scipy(alias)

    def test_register_type(self):
        pass
