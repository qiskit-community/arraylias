#######################
Arraylias Documentation
#######################

Arraylias is an open-source Python library providing single-dispatching tools centred around the
construction of an aliased module. Aliased modules are built by initially registering "libraries"
consisting of a collection of types, then registering different versions of a given function in the
aliased module for each underlying type library. When using the aliased module, function calls are
automatically dispatched to version of the function for the correct library based on the type of the
first argument.

Arraylias contains default pre-built aliased versions of both 
`NumPy <https://github.com/numpy/numpy>`_ and `SciPy <https://github.com/scipy/scipy>`_, with
additional registration of the `JAX <https://github.com/google/jax>`_ and 
`Tensorflow <https://github.com/tensorflow>`_ array libraries. This enables writing `NumPy
<https://github.com/numpy/numpy>`_ and `SciPy <https://github.com/scipy/scipy>`_ like code that will
that will execute on `NumPy <https://github.com/numpy/numpy>`_,
`JAX <https://github.com/google/jax>`_, and `Tensorflow <https://github.com/tensorflow>`_ array
objects as if it had been written in the respective native libraries. If necessary, these default
aliases can be further extended to fit the needs of the application.

.. warning::

   This package is still in the early stages of development and it is very likely
   that there will be breaking API changes in future releases.
   If you encounter any bugs please open an issue on
   `Github <https://github.com/Qiskit-Extensions/arraylias/issues>`_


.. toctree::
  :maxdepth: 2

  User Guide <userguide/index>
  API References <apidocs/index>
  Tutorial <tutorials/index>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`