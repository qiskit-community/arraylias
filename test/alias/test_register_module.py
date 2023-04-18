# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Alias tests"""


import unittest
import numpy
import scipy.linalg
from arraylias import Alias


class TestRegisterFunction(unittest.TestCase):
    """Test methods for registering modules

    These cover tests for the following Alias method
    * register_module
    """

    def test_register_module_default(self):
        """Test register_module method with default args"""
        alias = Alias()
        alias.register_type(numpy.ndarray)
        alias.register_module(numpy)
        func = alias("sin", like=numpy.ndarray)
        self.assertEqual(func, numpy.sin)

    def test_register_module_path_default(self):
        """Test register_module method with custom path"""
        alias = Alias()
        alias.register_type(numpy.ndarray)
        alias.register_module(numpy, path="custom.path")
        func = alias("custom.path.sin", like=numpy.ndarray)
        self.assertEqual(func, numpy.sin)

    def test_register_module_path(self):
        """Test register_module method with custom path inc lib"""

        alias = Alias()
        alias.register_type(numpy.ndarray)
        alias.register_module(numpy, path="custom.path")
        func = alias("custom.path.sin", like=numpy.ndarray)
        self.assertEqual(func, numpy.sin)

    def test_register_module_lib(self):
        """Test register_module method with custom lib"""
        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")
        alias.register_module(numpy, lib="test_lib")
        func = alias("sin", like=numpy.ndarray)
        self.assertEqual(func, numpy.sin)

    def test_register_module_path_and_lib(self):
        """Test register_module method with path and lib"""
        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")
        alias.register_module(numpy, lib="test_lib", path="custom.path")
        func = alias("custom.path.sin", like=numpy.ndarray)
        self.assertEqual(func, numpy.sin)

    def test_register_module_same_path_and_lib(self):
        """Test register_module method with path and lib"""
        alias = Alias()
        alias.register_type(numpy.ndarray)
        alias.register_module(numpy, path="numpy")
        func = alias("numpy.sin", like=numpy.ndarray)
        self.assertEqual(func, numpy.sin)

    def test_register_module_flatten(self):
        """Test using register_module to flatten modules"""
        alias = Alias()
        alias.register_type(numpy.ndarray)
        alias.register_module(numpy)
        alias.register_module(numpy.linalg, lib="numpy")
        alias.register_module(scipy.linalg, lib="numpy")
        func1 = alias("eig", like=numpy.ndarray)
        func2 = alias("expm", like=numpy.ndarray)
        self.assertEqual((func1, func2), (numpy.linalg.eig, scipy.linalg.expm))

    def test_register_module_prefer(self):
        """Test prefer when using register_module to flatten modules"""
        alias = Alias()
        alias.register_type(numpy.ndarray)
        alias.register_module(numpy)
        alias.register_module(numpy.linalg, lib="numpy")
        alias.register_module(scipy.linalg, lib="numpy", prefer=True)
        func1 = alias("eig", like=numpy.ndarray)
        func2 = alias("expm", like=numpy.ndarray)
        self.assertEqual((func1, func2), (scipy.linalg.eig, scipy.linalg.expm))
