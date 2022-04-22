# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for JAX transformations."""


import unittest
from arraylias import numpy_alias

unp = numpy_alias()()

try:
    from jax import jit, grad, vmap
    import jax.numpy as jnp
except ImportError:
    pass


class TestJaxTransformations(unittest.TestCase):
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
