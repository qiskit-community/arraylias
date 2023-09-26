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

"""
============================
Arraylias (:mod:`arraylias`)
============================

.. currentmodule:: arraylias

This module contains general tools for building aliased libraries, as well as default aliases for
NumPy and SciPy that provide a common interface for working with multiple underlying array
libraries.


Classes for building aliased libraries
======================================

.. autosummary::
    :toctree: ../stubs/

    Alias
    AliasedModule
    AliasedPath
    AliasError
    LibraryError


Constructors for default NumPy and SciPy aliases
================================================

.. autosummary::
    :toctree: ../stubs/

    numpy_alias
    scipy_alias

"""
from arraylias.version import __version__
from arraylias.alias import Alias
from arraylias.aliased import AliasedModule, AliasedPath
from arraylias.exceptions import LibraryError, AliasError
from arraylias.default_alias import numpy_alias, scipy_alias
