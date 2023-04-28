
Implementation of IAPWS-95 in Python, providing an interface to the Helmholtz
potential, its first and second derivatives, and common thermodynamic functions.
In the IAPWS-95 formulation, the Helmholtz energy of pure-phase water is fit to
a large collection of experimental data. The nondimensional Helmholtz potential,
ϕ (phi), written as a function of specific volume and temperature, uniquely
determines the pressure, entropy, and other properties such as the sound speed
and the heat capacity of the pure phase. Other potentials such as specific
energy, specific enthalpy, and the Gibbs potential are determined uniquely up
to a constant. In mixtures of liquid and vapour phases at thermodynamic
equilibrium, quantities such as specific volume and energy and can be computed
as a mass-weighted sum of the saturated liquid and saturated vapour quantities.
The saturated liquid and vapour quantities depend on the saturation curve, which
can be recovered from the Helmholtz potential by a Gibbs equilibrium condition,
sometimes called the Maxwell condition.

The implementation of the equation of state depends on the implementation of the
nondimensional Helmholtz function, composed of a sum of 56 terms.

Both a pure Python implementation and a Cython implementation are provided. The
Python implementation is implemented in numpy and treats inputs and outputs as
type `np.array`. Coefficients to the terms in the Helmholtz function are padded
and evaluated in a vector fashion. The Cython implementation is optimized for
speed, but requires some installation as described below. **The Python
implementation is not dependent on the Cythin implementation.**

## Installation (pure python)

This package depends on `numpy`. If necessary, install numpy using pip. Python >=3.9.1

## Usage (pure python)

```python
import numpy as np
import iapws95_light

# Print computed values and values from the IAPWS95 reference document
iapws95_light.print_verification_values()
# Compute pressure from density (kg/m^3) and temperature (K)
iapws95_light.p(100, 1000)
```

## Installation (_Cython_)

Cython usage requires a C compiler, and installation of cython. Installing
Cython can be done through pip. For instructions on both C installation and
Cython installation, see [Cython docs.](https://cython.readthedocs.io/en/latest/src/quickstart/install.html).

## Compilation (_Cython_)

A setup file for building the Cython module is provided. With Cython installed,
run
```python setup.py build_ext --inplace```
which produces an importable python module in the local directory.

## Usage (_Cython_)

The module `float_phi_functions` contains functions that can be called as a
typical python function. The module `iapws95_light_perf` provides wrappers for
testing the setup of Cython functions.

```python
import float_phi_functions
import iapws95

# Print computed values and values from the IAPWS95 reference document using
#   Cython backend
iapws95_light_perf.print_verification_values()
float_phi_functions.p(100, 1000)
```

## Performance timing

Using `%timeit` in an iPython notebook, example of performance is shown below
for one pair of input (rho, T).

```
Saturation curve evaluation
172 µs ± 19.3 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
Mixed-phase pressure
203 µs ± 36.8 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
Mixed-phase energy
245 µs ± 57 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
Pure phase pressure (supercritical)
5.66 µs ± 1.09 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
Pure phase energy (supercritical)
4.46 µs ± 528 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```

## Performance diagnostics

To compile with .html output highlighting python interactions, run

```cython -a float_phi_functions.pyx```

with `cython` in the system path variable. Alternatively, include the `annotate=True` flag in `cythonize` in `setup.py`.
