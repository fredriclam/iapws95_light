

Compilation

```python setup.py build_ext --inplace```

Usage

```python
import float_phi_functions
# Print value of phir_d for scalar inputs
d, t = 1.0, 0.5
print(float_phi_functions.phir_d(d, t))
```

Diagnostics

```cython -a float_phi_functions.pyx```