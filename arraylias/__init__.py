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

This module contains a common interface for working with array types from
multiple array libraries.


Functions
=========
.. autosummary::
    :toctree: ../stubs/

    numpy_alias
    scipy_alias


Classes
=======

.. autosummary::
    :toctree: ../stubs/

    Alias
    AliasedModule
    AliasedPath
    AliasError
    LibraryError
"""
from arraylias.version import __version__
from arraylias.alias import Alias
from arraylias.aliased import AliasedModule, AliasedPath
from arraylias.exceptions import LibraryError, AliasError
from arraylias.default_alias import numpy_alias, scipy_alias
