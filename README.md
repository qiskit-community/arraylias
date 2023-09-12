# arraylias

[![License](https://img.shields.io/github/license/Qiskit/arraylias.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)

**This repo is still in the early stages of development, there will be breaking API changes**

Arraylias is an open-source Python library providing single-dispatching tools centred around the
construction of an aliased module. Aliased modules are built by initially registering "libraries"
consisting of a collection of types, then registering different versions of a given function in the
aliased module for each underlying type library. When using the aliased module, function calls are
automatically dispatched to version of the function for the correct library based on the type of the
first argument.

Arraylias contains default pre-built aliased versions of both
[NumPy](https://github.com/numpy/numpy) and [Scipy](https://github.com/scipy/scipy), with additional
registration of the [JAX](https://github.com/google/jax) and
[Tensorflow](https://github.com/tensorflow) array libraries. This enables writing
[NumPy](https://github.com/numpy/numpy) and [Scipy](https://github.com/scipy/scipy) like code that
will that will execute on [NumPy](https://github.com/numpy/numpy),
[JAX](https://github.com/google/jax), and [Tensorflow](https://github.com/tensorflow) array objects
as if it had been written in the respective native libraries. If necessary, these default aliases
can be further extended to fit the needs of the application.


## Installation

Arraylias is installed by using `pip`:

```
pip install arraylias
```

## Contribution Guidelines

If you'd like to contribute to Arraylias, please take a look at our 
[contribution guidelines](CONTRIBUTING.md). This project adheres to Qiskit's 
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit-Extensions/arraylias/issues) for tracking
requests and bugs. For questions that are more suited for a forum we use the Qiskit tag in the 
[Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Authors and Citation

## License

[Apache License 2.0](LICENSE.txt)

