.. _arraylias-userguide:

.. module:: arraylias_userguide

====================
Arraylias User Guide
====================

The main class of the *arraylias* package is the :class:`.Alias` class.
This is used to build an aliased module structure for 1 or more
libraries and write generic code that can automatically dispatch to
the appropriate function in one of those libraries based on the type
of the first argument of the called function.

Full library sub-module paths are supported, and libraries can be
extended by registering custom modules, and modules can be extended by
registering custom functions.

Default NumPy and SciPy Aliases
===============================

The simplest way to use Arraylias is to use the :func:`.numpy_alias`
and :func:`scipy_alias` functions to return a pre-configured
:class:`.Alias` with NumPy module or SciPy module syntax respectively.

If the appropriate Python libraries are installed on the system
this will include pre-registered support for the following libraries
using the NumPy module interface

* `NumPy <https://numpy.org/>`_, extended with the `SciPy <https://scipy.org/>`_ linear algebra module.
* `JAX <https://github.com/google/jax>`_
* `Tensorflow <https://www.tensorflow.org/>`_

See the :ref:`Using an Alias <using>` section for a guide on using an Alias
in your code.

For extending with support for additional libraries refer to the
:ref:`Building an Alias <building>` section.


.. _using:

Using an Alias
==============

Given a configured :class:`.Alias` instance, an alias to a given path
is returned using the :meth:`.Alias.__call__` method. This will return an
:class:`.AliasedPath` or :class:`.AliasedModule` to the specified path.


Automatic dispatching
---------------------

An :class:`.AliasedPath` stores an unresolved path to a function.
Whether this function exists in any libraries is not resolved until
it is called. When called the type of the first argument will be used
to infer the library to dispatch on, and if a function exists for that
path it will be returned and called.

For example

.. code-block:: python

   # Return the default NumPy Alias
   alias = numpy_alias()
   eig = alias("linalg.eig")
   evals, evecs = eig(a)

If an :class:`.Alias` is called with no arguments this will return an
:class:`.AliasedModule` to the base path of the alias. This can be
traversed to submodules or functions using attribute based access such as:

.. code-block:: python

   unp = alias()  # Returns AliasedModule to base
   evals, evecs = unp.linalg.eig(a)

Aliased paths can also be treated as modules until they are resolved:

.. code-block:: python

   la = alias("linalg")
   evals, evecs = la.eig(a)


Dispatching to a specific library
---------------------------------

Dispatching to a specific library function or module can be done
by specifying a path including the library, or using the ``like``
kwarg of the call method.

.. code-block:: python

   # The following are equivalent
   tensordot = alias("tensordot", like="numpy")

   np = alias(like="numpy")
   tensordot = np.tensordot

The ``like`` kwarg can also be passed a type or object to infer the
library from, for example

.. code-block:: python

   tensordot = alias("tensordot", like=a)

When dispatching to a specific library the returned function will be
the actual library function or registered function for that path
instead of an :class:`.AliasedPath` instance. Note however that for
modules the return type will be a :class:`.AliasedModule` instance
instead of the actual library module. This is to allow access to
custom registered functions and sub-modules in the :class:`.Alias`
for that library.


.. _building:

Building an Alias
=================

Registering Types
-----------------
Types are registered to a specific library for dispatching using the
:meth:`.Alias.register_type` method. Registering a type a second time
will override the previous library with the new library. Any subclasses
of registered types will also be matched for dispatching if they are not
separately registered.

For example to register NumPy ndarrays:

.. code-block:: python

   alias.register_type(numpy.ndarray, "numpy")

The :meth:`.Alias.registered_types` and :meth:`.Alias.registered_libs`
methods can be used to return aa tuple of  all registered types and
libraries respectively.

Registering Modules
-------------------
The :meth:`.Alias.register_module` method can be used to register
a module for dispatching aliased functions and modules for a library.
By default modules are registered to the base path of that library if
a custom path is not provided.

For example to register the base NumPy module, which will also allow
path based access to all sub-modules accessible from ``numpy``.

.. code-block:: python

   alias.register_module(numpy, "numpy")

We can also use this method to modify the default NumPy path, for
example to add SciPy linear algebra functions to the NumPy linear
algebra path:

.. code-block:: python

   alias.register_module(scipy.linalg, "numpy", path="linalg")

Note that the default :func:`.numpy_alias` does not include SciPy functions.
There is a separate :func:`.scipy_alias` that can be used to initialize a
SciPy alias.


Registering Functions
---------------------
Individual functions are registered using the
:meth:`.Alias.register_function` method.

.. code-block:: python

   alias.register_function(some_function, lib="library")

The :meth:`.Alias.register_function` can also be used as a decorator like

.. code-block:: python

   @alias.register_function(lib="numpy")
   def foo(a, x, b):
      return a * x + b

By default the name of the function will be used as its path, a custom
name can be provided by using the ``path`` kwarg

.. code-block:: python

   @alias.register_function(lib="numpy", path="line")
   def _(a, x, b):
      return a * x + b

Note that a function can be registered to a specific submodule by
including it in the path. These modules do not even need to exist in
the library, they will still be traversable by the alias. Eg

.. code-block:: python

   @alias.register_function(lib="numpy", path="objectives.linear.line")
   def _(a, x, b):
      return a * x + b

   # Evaluate added function
   unp = alias("numpy")
   y = unp.objectives.linear.line(a, x, b)

If the ``path`` kwarg is not provided, the name of the function will be
used as the path. Functions can

Registering Fallback Functions
------------------------------

The :meth:`.Alias.register_fallback` can be used to register a fallback
function that will be invoked if a match to a specific function path
cannot be found for the dispatched library. Like :meth:`.Alias.register_function`
it can also be used as a function decorator.

Typically this would be used to implement a generic method for a
custom function that works for all registered libraries, and then
also registering a specialized version of the function for a
specific library using the :meth:`.Alias.register_function` method.

Registering Default Functions
-----------------------------

The :meth:`.Alias.register_default` can be used to register a default function
that will be invoked if the type of the first argument of a called
:class:`.AliasedPath` function does not match any registered library
types.

Typically this would be used to register a default implementation
of a function that may take other types than arrays as its first
argument, for example this is used by :func:`.numpy_alias` to
register ``numpy.array`` and ``numpy.asarray`` as default
functions:

.. code-block:: python

   alias.register_default(numpy.array)

   unp = alias()
   a = unp.array(SomeCustomClass())
