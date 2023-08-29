.. _arraylias-tutorial:

.. module:: arraylias_tutorial


==================
Arraylias Tutorial
==================



Arraylias provides configurating an aliased module for some libraries, thereby eliminating the need for us
 to write code for each library.
In this tutorial, we will learn how to write generic code with an example of numerically computation by using 
`NumPy <https://numpy.org/>`_ and `JAX <https://github.com/google/jax>`_.
We'll use a code example that employs the Runge-Kutta method to numerically simulate the Schrödinger equation.
In addition, we will show the utility of :meth:`.Alias.register_function` method to help efficiently solving it.

In this section, we will go through the following steps:

1. Import the required libraries
2. Return the numpy alias
3. Register custom functions using :meth:`.Alias.register_function`
4. Define Runge-Kutta method using generic code with Arraylias
5. Solve with NumPy array
6. Solve with JAX array


1. Import the required libraries
--------------------------------

Here, we import the necesary libraries.
The :func:`.numpy_alias` is a function which returns a pre-registered :class:`.Alias` with moudles such as 
`NumPy <https://numpy.org/>`_, `JAX <https://github.com/google/jax>`_, and `Tensorflow <https://www.tensorflow.org/>`_ and
sets up the bridge between these libraries. 

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

2. Return the numpy alias
-------------------------

We initialize Arraylias using the :func:`.numpy_alias` function.

.. jupyter-execute::
    alias = numpy_alias()

3. Register custom functions using :meth:`.Alias.register_function`
-------------------------------------------------------------------

The Runge-Kutta method is solving differential equations by approximating the power series to
the foruth order terms. It provides to find solutions by numerically calculations.
In this part, we define the Runge-Kutta method to solve the Schrödinger equation for each NumPy and 
JAX.

JAX's strength lies in vectorized operations and parallel computations. In some cases, Python for loops
may not benefit from JAX's optimizations. Therefore, it is recommended to avoid using the Python for 
loops to effectively maximize JAX's strength. Using ``jax.lax.scan`` function could enable efficiently
parallel computations.

We can prepare your own custom function for each library by using :meth:`.Alias.register_function` method.

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
In this part, we can write generic code by Arraylias.

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
the type of the input. The just defined function ``unp.runge_kutta`` is called in this function.
This custom function is also called depending on the type of ``phi0``.

5. Solve with NumPy array
-------------------------

We just finished writing the generic code to solve the Schrödinger equation.
First, we solve the equation by using Numpy as the input.

.. jupyter-execute::
    x = np.linspace(-10, 10, 101)
    dx = x[1] - x[0]
    V = unp.zeros_like(x)

The initial wave function of the equation is chosen as a wave packet of a free electron, 
which gives 

.. jupyter-execute::
    phi0 = unp.sqrt(1 / np.sqrt(np.pi) / sigma) * np.exp(-x**2 / (2 * sigma**2)) * np.exp(1j * k * x)
    #空間部分であることのコメント
    plt.plot(x, phi0)


We evolve the wave function in time by ``N-1`` steps and get the time evolved spatial distribution of this wave function.

.. jupyter-execute::
    phi_final = solve_with_RungeKutta(x,V, 0.0005)
    plt.plot(x, phi_final)

.. jupyter-execute::
    %timeit RungeKutta(x,V, 0.0005)


6. Solve with JAX array
-----------------------

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