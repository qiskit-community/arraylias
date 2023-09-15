.. _rungekutta:

.. module:: rungekutta


Solving the Schrodinger equation with either NumPy or JAX using the default NumPy alias
=======================================================================================

In this tutorial, we will learn how to write generic code with examples of numerical computation by using 
`NumPy <https://numpy.org/>`_ and `JAX <https://github.com/google/jax>`_.
We'll use a code example that employs the Runge-Kutta method to numerically simulate the matrix evolution of the Schrödinger equation.
In addition, we will showcase the utility of :meth:`.Alias.register_function` method for efficient problem-solving.

In this section, we will go through the following steps:

1. Import the required libraries and initialize the numpy alias.
2. Define the equation using :class:`.Alias`.
3. Solve the right-hand side function using existing solvers.
4. Register custom solvers using :meth:`.Alias.register_function` method.
5. Solve using the custom function.


1. Import the required libraries and initialize the numpy alias
---------------------------------------------------------------

Here, we import the necesary libraries.

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

    from scipy.integrate import solve_ivp
    from jax.experimental.ode import odeint

    # set some variables
    dt = 0.01
    N = 1001


The :func:`.numpy_alias` function returns a pre-registered :class:`.Alias` instance with modules such as 
`NumPy <https://numpy.org/>`_, `JAX <https://github.com/google/jax>`_, and `Tensorflow <https://www.tensorflow.org/>`_,
setting a bridge between these libraries.

We initialize Arraylias using the :func:`.numpy_alias` function and :class:`.Alias`.

.. jupyter-execute::

    alias = numpy_alias()
    unp = alias()

2. Define the equation using :class:`.Alias`
--------------------------------------------

We solve the Schrödinger equation using the Runge-Kutta method in this tutorial.
The Schrödinger equation is written as

.. math:: \frac{\partial\psi}{\partial t} = -i * H \psi

The equation we are solving represents a qubit's state as a two-level system. The Hamiltonian is

.. math:: H = \frac{1}{2} \times 2 \pi \nu_z {Z} + 2 \pi \nu_x \cos(2 \pi \nu_d t){X},

where :math:`\{X,Y,Z\}` are the Pauli matrices.

We can express the right-hand side (RHS) function of this equation as follows:

.. jupyter-execute::

    Z = np.array([[1,0],[0,-1]])
    X = np.array([[0,1],[1,0]])

    def rhs(t,y):
        return unp.matmul(-1j * (5 * Z -  unp.cos(10 * t) * X ), y)

We can confirm that the rhs function outputs the type corresponding to the input type.

.. jupyter-execute::

    # Numpy input
    rhs(0.1, np.array([0., 1.]))


.. jupyter-execute::

    # Jax.numpy input
    rhs(jnp.array(0.1), jnp.array([0., 1.]))

We eventually want to find the probability of existence of this qubit state, so we will prepare the following function.

.. jupyter-execute::

    def state_probabilities(state):
        return unp.abs(state) ** 2

3. Solve the right-hand side function using existing solvers
------------------------------------------------------------

Here we show how the rhs function can be passed to numerical ODE solvers in both SciPy and JAX as if the function had been natively written in either library.

First, we solve the equation by using Numpy as the input and ``scipy.integrate.solve_ivp`` as a solver.
We define the initial state, the time span for the simulation, and time point we want to simulate.

.. jupyter-execute::

    init_state = np.array([1. + 0j,0. + 0j])

    t_span = [0,(N-1) * dt]
    T = np.linspace(0,(N-1) * dt,N)

We solve by using ``scipy.integrate.solve_ivp`` and plot the probabilities of each state.

.. jupyter-execute::

    sol = solve_ivp(rhs,t_span,init_state,method='RK45',t_eval=T)
    probabilities = state_probabilities(sol.y)

    plt.plot(sol.t, probabilities[0], label="0")
    plt.plot(sol.t, probabilities[1], label="1")
    plt.xlabel('T')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()


Second, we solve the equation by using Jax.array as the input and ``jax.experimental.ode.odeint`` as a solver.

.. jupyter-execute::

    init_state = jnp.array([1. + 0j,0. + 0j])

    t_span = [0,(N-1) * dt]
    T = jnp.linspace(0,(N-1) * dt,N)

    sol = odeint(lambda y,t: rhs(t,y), init_state, T)
    probabilities = state_probabilities(sol.T)
    plt.plot(T, probabilities[0], label="0")
    plt.plot(T, probabilities[1], label="1")
    plt.xlabel('T')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()



4. Register custom solvers using :meth:`.Alias.register_function`
-----------------------------------------------------------------

In this section, we introduce how to register the function, taking the 4th order Runge-Kutta method
to solve ordinary differential equations as an example.

The 4th order Runge-Kutta fomula is here:

.. math::

    k_1 &= h \cdot f(t_n, y_n) \\
    k_2 &= h \cdot f(t_n + \frac{h}{2}, y_n + \frac{k_1}{2}) \\
    k_3 &= h \cdot f(t_n + \frac{h}{2}, y_n + \frac{k_2}{2}) \\
    k_4 &= h \cdot f(t_n + h, y_n + k_3) \\
    y_{n+1} &= y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)

, where :math:`y_{n}`, :math:`t_{n}`, and :math:`h` are current solution, current time, and time step size, respectively.

We define the Runge-Kutta method to be used later here:

.. jupyter-execute::

    def runge_kutta_step(n, state):
        k1 = dt * rhs(n * dt, state)
        k2 = dt * rhs(n * dt + 0.5 * dt, state + 0.5*k1)
        k3 = dt * rhs(n * dt + 0.5 * dt, state + 0.5*k2)
        k4 = dt * rhs(n * dt + dt, state + k3)
        return (k1 + 2*k2 + 2*k3 + k4) / 6.

When writing the custom solver, we write and register a version for both NumPy and JAX. 
The NumPy version uses standard Python loops, whereas the JAX version uses the JAX looping construct ``jax.lax.scan``.

In Numpy case, we define the function ``runge_kutta`` for Numpy.

.. jupyter-execute::

    @alias.register_function(lib="numpy", path="runge_kutta")
    def _(state, N):
        probabilities = []
        for n in range(N):
            probabilities.append(state_probabilities(state))
            state+= runge_kutta_step(n, state)
        return probabilities


This custom function of ``runge_kutta`` is registered using the decorator ``@alias.register_function`` under 
NumPy.

In the case of JAX,

.. jupyter-execute::

    @alias.register_function(lib="jax", path="runge_kutta")
    def _(state, N):
        def runge_kutta_step_scan(carry, probabilities):
            n, state = carry
            probabilities = state_probabilities(state)
            state+= runge_kutta_step(n, state)
            return (n + 1, state), probabilities
        _, probabilities = jax.lax.scan(runge_kutta_step_scan, (0, state), jnp.zeros((N,2)))
        return probabilities

5. Solve using the custom function
----------------------------------

We have just completed writing the generic code to solve the Schrödinger equation.
We will now demonstrate two cases of solving the equation using NumPy and JAX as inputs.

First, solve with NumPy:

.. jupyter-execute::

    init_state = np.array([1. + 0j,0. + 0j])

    probabilities = unp.array(unp.runge_kutta(init_state, N))

    T = np.linspace(0,(N-1) * dt,N)
    plt.plot(T, probabilities, label = ["0", "1"])
    plt.xlabel('T')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

.. jupyter-execute::

    %timeit unp.array(unp.runge_kutta(init_state, N))


Second case is JAX:

.. jupyter-execute::

    init_state = jnp.array([1. + 0j,0. + 0j])
    probabilities = unp.array(unp.runge_kutta(init_state, N))

    T = np.linspace(0,(N-1) * dt,N)

    plt.plot(T, probabilities, label=["0", "1"])
    plt.xlabel('T')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

Lastly, we verify that the function ``unp.runge_kutta`` behaves as expected under JAX transformations.

.. jupyter-execute::

    from functools import partial

    @partial(jax.jit, static_argnums=(1,))
    def solve_with_RungeKutta_jit(init_state, N):
        return unp.array(unp.runge_kutta(init_state, N))

.. jupyter-execute::

    %timeit solve_with_RungeKutta_jit(init_state, N)

