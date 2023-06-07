# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for JAX transformations."""


import unittest

# pylint: disable=consider-using-from-import
import scipy.stats as stats
from arraylias import numpy_alias

unp = numpy_alias()("jax")


class TestJAXRandomModule(unittest.TestCase):
    """Tests that JAX transformations perform as expected on dispatched functions."""

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

    def test_random_seed(self):
        """Test the same result by seeting random seed"""
        seed = 123
        unp.random_seed(seed)
        result1 = unp.random_normal()
        unp.random_seed(seed)
        result2 = unp.random_normal()
        self.assertEqual(result1, result2)

    def test_random_normal(self):
        """Test random output according to a normal distribution"""
        for _ in range(1000):
            self.assertTrue(unp.random_uniform() <= 1.0 and unp.random_uniform() >= 0.0)

    def test_random_uniform(self):
        """Test random output according to a uniform distribution"""
        _, p_val = stats.normaltest([unp.random_normal() for _ in range(1000)])
        self.assertTrue(p_val > 0.05)
