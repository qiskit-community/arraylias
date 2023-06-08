# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Alias tests"""


import unittest
from numbers import Number
from arraylias import Alias


class TestRegisterType(unittest.TestCase):
    """Test methods for registering types and libraries

    These cover tests for the following Alias methods
    * register_type
    * registered_types
    * registered_libs
    * infer_libs
    """

    def test_registered_types_order(self):
        """Test order of registered_types"""
        alias = Alias()
        alias.register_type(Number, "number_lib")
        alias.register_type(int, "int_lib")
        alias.register_type(complex, "complex_lib")
        self.assertEqual(alias.registered_types(), (Number, int, complex))

    def test_registered_libs_order(self):
        """Test order of registered_libs"""
        alias = Alias()
        alias.register_type(Number, "number_lib")
        alias.register_type(int, "int_lib")
        alias.register_type(complex, "complex_lib")
        self.assertEqual(alias.registered_libs(), ("number_lib", "int_lib", "complex_lib"))

    def test_infer_lib_types(self):
        """Test exact class matching for infer_libs"""

        alias = Alias()
        alias.register_type(int, "int_lib")
        alias.register_type(complex, "complex_lib")

        self.assertEqual(alias.infer_libs(int), ("int_lib",))
        self.assertEqual(alias.infer_libs(complex), ("complex_lib",))
        self.assertEqual(alias.infer_libs(float), tuple())

    def test_infer_lib_instances(self):
        """Test exact class matching for infer_lib"""

        alias = Alias()
        alias.register_type(int, "int_lib")
        alias.register_type(complex, "complex_lib")

        self.assertEqual(alias.infer_libs(1), ("int_lib",))
        self.assertEqual(alias.infer_libs(1j), ("complex_lib",))
        self.assertEqual(alias.infer_libs(1.0), tuple())

    def test_infer_lib_str(self):
        """Test exact class matching for infer_lib"""

        alias = Alias()
        alias.register_type(int, "int_lib")
        alias.register_type(complex, "complex_lib")

        self.assertEqual(alias.infer_libs("int_lib"), ("int_lib",))
        self.assertEqual(alias.infer_libs("complex_lib"), ("complex_lib",))
        self.assertEqual(alias.infer_libs("other_lib"), tuple())

    def test_infer_lib_subtype(self):
        """Test exact class and subclass matching for infer_lib"""

        alias = Alias()
        alias.register_type(Number, "number_lib")
        alias.register_type(int, "int_lib")
        alias.register_type(complex, "complex_lib")

        self.assertEqual(alias.infer_libs(int), ("int_lib",))
        self.assertEqual(alias.infer_libs(complex), ("complex_lib",))
        self.assertEqual(alias.infer_libs(float), ("number_lib",))

    def test_infer_lib_subclass(self):
        """Test exact class and subclass matching for infer_lib"""

        alias = Alias()
        alias.register_type(Number, "number_lib")
        alias.register_type(int, "int_lib")
        alias.register_type(complex, "complex_lib")

        self.assertEqual(alias.infer_libs(1), ("int_lib",))
        self.assertEqual(alias.infer_libs(1j), ("complex_lib",))
        self.assertEqual(alias.infer_libs(1.0), ("number_lib",))

    def test_infer_lib_allow_sequence(self):
        """Test allow sequence for inferring libs"""

        alias = Alias()
        alias.register_type(Number, "number_lib")
        alias.register_type(int, "int_lib")

        tup = (1, 1.0, 1j)
        self.assertEqual(alias.infer_libs(tup), ("int_lib",))
        self.assertEqual(alias.infer_libs(tup, allow_sequence=False), tuple())
        self.assertEqual(alias.infer_libs(list(tup)), ("int_lib",))
        self.assertEqual(alias.infer_libs(list(tup), allow_sequence=False), tuple())

        tup = (1.0, 1, 1j)
        self.assertEqual(alias.infer_libs(tup), ("number_lib",))
        self.assertEqual(alias.infer_libs(tup, allow_sequence=False), tuple())
        self.assertEqual(alias.infer_libs(list(tup)), ("number_lib",))
        self.assertEqual(alias.infer_libs(list(tup), allow_sequence=False), tuple())

    def test_infer_lib_allow_sequence_nested(self):
        """Test nested allow sequence for inferring libs"""

        alias = Alias()
        alias.register_type(Number, "number_lib")
        alias.register_type(int, "int_lib")

        nested = [[[1, 1.0], [1j, 1]], [[-1.0, 1.0], [1j, 1]]]
        self.assertEqual(alias.infer_libs(nested), ("int_lib",))
        self.assertEqual(alias.infer_libs(nested, allow_sequence=False), tuple())
        self.assertEqual(alias.infer_libs(tuple(nested)), ("int_lib",))
        self.assertEqual(alias.infer_libs(tuple(nested), allow_sequence=False), tuple())

    def test_register_type__AUTOLIB(self):
        """Test automatic inference of lib from base module name"""
        alias = Alias()
        alias.register_type(complex)
        alias.register_type(Number)
        self.assertEqual(alias.infer_libs(1j), ("builtins",))
        self.assertEqual(alias.infer_libs(2.0), ("numbers",))

    def test_register_multiple_libs(self):
        """Test registering 1 type to multiple libs"""
        alias = Alias()
        alias.register_type(complex, "lib1")
        alias.register_type(complex, "lib2")
        self.assertEqual(alias.infer_libs(1j), ("lib1", "lib2"))

    def test_register_multiple_libs_prefer(self):
        """Test registering 1 type to multiple libs with prefer arg"""
        alias = Alias()
        alias.register_type(complex, "lib1")
        alias.register_type(complex, "lib2", prefer=True)
        self.assertEqual(alias.infer_libs(1j), ("lib2", "lib1"))
