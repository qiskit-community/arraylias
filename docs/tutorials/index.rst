.. _arraylias-tutorial:

.. module:: arraylias_tutorial


==================
Arraylias Tutorial
==================



Arraylias provides a way to configure aliased modules for some libraries, thereby eliminating the need for us
to write separate code for each library.
In this tutorial, we will learn how to write generic code with an example of numerical computation by using 
`NumPy <https://numpy.org/>`_ and `JAX <https://github.com/google/jax>`_.
We'll use a code example that employs the Runge-Kutta method to numerically simulate the Schrödinger equation.
In addition, we will demonstrate the utility of :meth:`.Alias.register_function` method for efficiently solving it.

In this section, we will go through the following steps:

1. Import the required libraries
2. Initialize the numpy alias
3. Register custom functions using :meth:`.Alias.register_function` method
4. Define Runge-Kutta method using generic code with Arraylias
5. Solve using the NumPy array
6. Solve using the JAX array


1. Import the required libraries
--------------------------------

Here, we import the necesary libraries.

The :func:`.numpy_alias` function returns a pre-registered :class:`.Alias` instance with modules such as 
`NumPy <https://numpy.org/>`_, `JAX <https://github.com/google/jax>`_, and `Tensorflow <https://www.tensorflow.org/>`_,
setting a bridge between these libraries.

.. jupyter-execute::
    :hide-code:

    import warnings
    warnings.filterwarnings('ignore', message='', category=Warning, module='', lineno=0, append=False)

.. jupyter-execute::

    from arraylias import numpy_alias
    import numpy as np
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp

    # set some variables
    k = 5
    sigma = 0.4
    N = 201

2. Initialize the numpy alias
-----------------------------

We initialize Arraylias using the :func:`.numpy_alias` function.

.. jupyter-execute::

    alias = numpy_alias()

3. Register custom functions using :meth:`.Alias.register_function`
-------------------------------------------------------------------

The Runge-Kutta method solves differential equations by approximating power series up to
the foruth order terms. It provides to find solutions by numerically calculations.
In this section, we define the Runge-Kutta method to solve the Schrödinger equation using each NumPy and 
JAX.

JAX's strength lies in vectorized operations and parallel computations. In some cases, Python for loops
may not effectively use JAX's optimizations. Therefore, it is recommended to avoid using the Python for 
loops to effectively maximize JAX's strength. Using ``jax.lax.scan`` function could enable efficiently
parallel computations.

We can create custom functions for each library by using :meth:`.Alias.register_function` method.

In Numpy case, we define the function ``runge_kutta`` for Numpy.


.. jupyter-execute::

    @alias.register_function(lib="numpy", path="runge_kutta")
    def _(phi, U, dt, N):
        for n in range(N-1):
            k1 = dt * np.matmul(phi,U)
            k2 = dt * np.matmul(phi + 0.5*k1, U)
            k3 = dt * np.matmul(phi + 0.5*k2, U)
            k4 = dt * np.matmul(phi + 1.0*k3, U)

            phi+= (k1 + 2*k2 + 2*k3 + k4) / 6.
        return phi

This function of ``runge_kutta`` is registered using the decorator ``@alias.register_function`` under 
NumPy.

In JAX, we want to use ``jax.lax.scan`` function instead of Python for loops.

.. jupyter-execute::

    @alias.register_function(lib="jax", path="runge_kutta")
    def _(phi, U, dt, N):
        def step_for_jax_scan(phi, _):
            k1 = dt * unp.matmul(phi, U)
            k2 = dt * unp.matmul(phi + 0.5*k1, U)
            k3 = dt * unp.matmul(phi + 0.5*k2, U)
            k4 = dt * unp.matmul(phi + 1.0*k3, U)

            phi_new = phi + (k1 + 2*k2 + 2*k3 + k4) / 6.
            return phi_new, None
        phi, _ = jax.lax.scan(step_for_jax_scan, phi, jnp.arange(N-1))
        return phi

We need to initialize :class:`.Alias` to reflect on the above registered functions.

.. jupyter-execute::

    unp = alias()


4. Define Runge-Kutta method using generic code with Arraylias
--------------------------------------------------------------

Next, we solve the Schrödinger equation by Runge-Kutta method.
The equation we solve is the single-particle time-dependent Schrödinger equation:

.. math::

    i\hbar\frac{\partial\psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi + V(x)\psi

In this tutorial, we set :math:`\hbar` and :math:`m` to 1.

.. math::

    i\frac{\partial\psi}{\partial t} = -\frac{1}{2}\nabla^2\psi + V(x)\psi


Furthermore, we consider a potential well, where inside the well, the potential is assumed
to zero. The wave function is initialized as a Gaussian wave packet:

.. math::

    \sqrt{\frac{1}{{\sqrt{\pi}} \sigma}} \cdot \exp\left(-\frac{x^2}{{2 \sigma^2}}\right) \cdot \exp\left(i k x\right)

Here, :math:`sigma` and :math:`k` are the standard deviation and momentum, respectively.

In this section, we solve the time evolution of this wave function by Runge-Kutta method and we can write generic code by Arraylias.

.. jupyter-execute::

    def solve_with_RungeKutta(x, V, dt):
        phi0 = unp.sqrt(1 / unp.sqrt(np.pi) / sigma) * unp.exp(-x**2 / (2 * sigma**2)) * unp.exp(1j * k * x)
        x_size = x.size
        dx = x[1] - x[0]

        V = unp.diag(V)
        T = unp.diag(unp.ones(x_size-1),-1) + unp.diag(-2 * unp.ones(x_size), 0) + unp.diag(unp.ones(x_size-1), 1)

        T *= (-1 / (2 * dx ** 2))
        U = -1j*(T + V)
        

        return unp.runge_kutta(phi0, U, dt, N)

For instance, we can use Arraylias like ``unp.exp(x)``. Arraylias uses automatically the module corresponding to
the type of the input. The defined custom function ``unp.runge_kutta`` is called in this function.
The type of ``phi0`` determines which custom function is used.


5. Solve using the NumPy array
------------------------------

We just completed writing the generic code to solve the Schrödinger equation.
First, we solve the equation by using Numpy as the input.

.. jupyter-execute::

    x = np.linspace(-10, 10, 101)
    dx = x[1] - x[0]
    V = unp.zeros_like(x)

The initial wave function of the equation is chosen as a wave packet of a free electron, 
which gives 

.. jupyter-execute::

    phi0 = unp.sqrt(1 / np.sqrt(np.pi) / sigma) * np.exp(-x**2 / (2 * sigma**2)) * np.exp(1j * k * x)
    plt.plot(x, phi0)


We evolve the wave function over time in ``N-1`` steps and obtain the time evolved spatial distribution of this wave function.

.. jupyter-execute::

    phi_final = solve_with_RungeKutta(x,V, 0.0005)
    plt.plot(x, phi_final)

.. jupyter-execute::

    %timeit solve_with_RungeKutta(x,V, 0.0005)


6. Solve using the JAX array
----------------------------

Second, we solve the equation by using JAX as the input.

.. jupyter-execute::

    x = jnp.linspace(-10, 10, 101)
    dx = x[1] - x[0]
    V = unp.zeros_like(x)

All the above variables' types are JAX array and we can jit this function.

.. jupyter-execute::

    solve_with_RungeKutta_jit = jax.jit(solve_with_RungeKutta)

We solve the time evolved spatial distribution of the wave function by Runge-Kutta method 
using ``jax.lax.scan``.

.. jupyter-execute::

    phi_final = solve_with_RungeKutta_jit(x,V, 0.0005)


.. jupyter-execute::

    %timeit solve_with_RungeKutta_jit(x,V, 0.0005)


By following these steps, we've learned how to leverage Arraylias to 
write versatile numerical code that can efficiently switch between different arrays.
We've also explored how to accelerate our code's execution using JAX's JIT compilation.
By understanding the strengths of different numerical libraries and harnessing their capabilities through Arraylias, we can create high-performance code for various computational tasks. 
Apply these concepts to your own projects to unlock their full potential.