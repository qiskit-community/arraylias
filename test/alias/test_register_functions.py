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
from arraylias import Alias


class TestRegisterFunction(unittest.TestCase):
    """Test methods for registering functions

    These cover tests for the following Alias methods
    * register_function
    * register_fallback
    * register_default
    """

    def test_register_function_default(self):
        """Test register_function with default args"""

        alias = Alias()
        alias.register_type(numpy.ndarray)
        alias.register_function(numpy.sin)
        func = alias("sin", like=numpy.ndarray)
        self.assertEqual(func, numpy.sin)

    def test_register_function_name(self):
        """Test register_function with custom name and lib"""

        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")
        alias.register_function(numpy.sin, path="np_sin", lib="test_lib")
        func = alias("np_sin", like="test_lib")
        self.assertEqual(func, numpy.sin)

    def test_register_function_path(self):
        """Test register_function with custom name and lib"""

        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")
        alias.register_function(numpy.sin, path="some.long.path.sin", lib="test_lib")
        func = alias("some.long.path.sin", like="test_lib")
        self.assertEqual(func, numpy.sin)

    def test_register_function_lib_path(self):
        """Test register_function with custom name and lib"""

        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")
        alias.register_function(numpy.sin, lib="test_lib", path="some.path.sin")
        func = alias("some.path.sin", like="test_lib")
        self.assertEqual(func, numpy.sin)

    def test_register_function_custom(self):
        """Test register_function"""
        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")

        def myfunc(arg):
            return arg

        alias.register_function(myfunc, lib="test_lib")

        func = alias("myfunc", like="test_lib")
        self.assertEqual(func, myfunc)

    def test_register_function_custom_name(self):
        """Test register_function with custom name"""
        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")

        def myfunc(arg):
            return arg

        alias.register_function(myfunc, path="supermyfunc", lib="test_lib")
        func = alias("supermyfunc", like="test_lib")
        self.assertEqual(func, myfunc)

    def test_register_function_custom_path(self):
        """Test register_function with custom name"""
        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")

        def myfunc(arg):
            return arg

        alias.register_function(myfunc, path="some.module.supermyfunc", lib="test_lib")
        func = alias("some.module.supermyfunc", like="test_lib")
        self.assertEqual(func, myfunc)

    def test_register_function_decorator(self):
        """Test register_function with custom name"""

        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")

        @alias.register_function(path="supermyfunc", lib="test_lib")
        def myfunc(arg):
            return arg

        func = alias("supermyfunc", like="test_lib")
        self.assertEqual(func, myfunc)

    def test_register_function_decorator_default(self):
        """Test register_function"""
        alias = Alias()
        lib = self.__module__.split(".", maxsplit=1)[0]
        alias.register_type(numpy.ndarray, lib)

        @alias.register_function
        def myfunc(arg):
            return arg

        func = alias("myfunc", like=lib)
        self.assertEqual(func, myfunc)

    def test_register_fallback_default(self):
        """Test register_fallback with default args"""

        alias = Alias()
        alias.register_type(numpy.ndarray)
        alias.register_fallback(numpy.sin)
        func = alias("sin", like=numpy.ndarray)
        self.assertEqual(func, numpy.sin)

    def test_register_fallback_name(self):
        """Test register_fallback with custom name"""

        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")
        alias.register_fallback(numpy.sin, path="np_sin")
        func = alias("np_sin", like="test_lib")
        self.assertEqual(func, numpy.sin)

    def test_register_fallback_path(self):
        """Test register_fallback with custom path"""

        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")
        alias.register_fallback(numpy.sin, path="some.module.path.sin")
        func = alias("some.module.path.sin", like="test_lib")
        self.assertEqual(func, numpy.sin)

    def test_register_fallback_custom(self):
        """Test register_fallback"""
        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")

        def myfunc(arg):
            return arg

        alias.register_fallback(myfunc)

        func = alias("myfunc", like="test_lib")
        self.assertEqual(func, myfunc)

    def test_register_fallback_custom_name(self):
        """Test register_fallback with custom name"""
        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")

        def myfunc(arg):
            return arg

        alias.register_fallback(myfunc, path="supermyfunc")
        func = alias("supermyfunc", like="test_lib")
        self.assertEqual(func, myfunc)

    def test_register_fallback_decorator(self):
        """Test register_fallback with custom name"""

        alias = Alias()
        alias.register_type(numpy.ndarray, "test_lib")

        @alias.register_fallback(path="supermyfunc")
        def myfunc(arg):
            return arg

        func = alias("supermyfunc", like="test_lib")
        self.assertEqual(func, myfunc)

    def test_register_fallback_decorator_default(self):
        """Test register_fallback"""
        alias = Alias()
        lib = self.__module__.split(".", maxsplit=1)[0]
        alias.register_type(numpy.ndarray, lib)

        @alias.register_fallback
        def myfunc(arg):
            return arg

        func = alias("myfunc", like=lib)
        self.assertEqual(func, myfunc)

    def test_register_default_default(self):
        """Test register_default with default args"""

        alias = Alias()
        alias.register_default(numpy.sin)
        func = alias("sin", like=numpy.ndarray)
        self.assertEqual(func, numpy.sin)

    def test_register_default_name(self):
        """Test register_default with custom path"""

        alias = Alias()
        alias.register_default(numpy.sin, path="np_sin")
        func = alias("np_sin", like="test_lib")
        self.assertEqual(func, numpy.sin)

    def test_register_default_path(self):
        """Test register_default with custom path"""

        alias = Alias()
        alias.register_default(numpy.sin, path="some.module.path.sin")
        func = alias("some.module.path.sin", like="test_lib")
        self.assertEqual(func, numpy.sin)

    def test_register_default_custom(self):
        """Test register_default"""
        alias = Alias()

        def myfunc(arg):
            return arg

        alias.register_default(myfunc)

        func = alias("myfunc", like="test_lib")
        self.assertEqual(func, myfunc)

    def test_register_default_custom_name(self):
        """Test register_default with custom name"""
        alias = Alias()

        def myfunc(arg):
            return arg

        alias.register_default(myfunc, path="supermyfunc")
        func = alias("supermyfunc", like="test_lib")
        self.assertEqual(func, myfunc)

    def test_register_default_decorator(self):
        """Test register_default with custom name"""
        alias = Alias()

        @alias.register_default(path="supermyfunc")
        def myfunc(arg):
            return arg

        func = alias("supermyfunc", like="test_lib")
        self.assertEqual(func, myfunc)

    def test_register_default_decorator_default(self):
        """Test register_default"""
        alias = Alias()

        @alias.register_default
        def myfunc(arg):
            return arg

        func = alias("myfunc", like=numpy.ndarray)
        self.assertEqual(func, myfunc)
