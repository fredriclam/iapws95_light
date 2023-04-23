from setuptools import setup
from Cython.Build import cythonize

# from Cython.Compiler.Options import get_directive_defaults
# directive_defaults = get_directive_defaults()
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

# setup(
#   ext_modules = cythonize(["float_phi_functions.pyx"], language_level="3")
# )

from distutils.extension import Extension
extensions = [
  Extension("float_phi_functions",
    ["float_phi_functions.pyx"]), 
  Extension("float_mix_functions",
    ["float_mix_functions.pyx"]), 
]
setup(
    ext_modules = cythonize(extensions)
)

# # Optimize only O1
# from distutils.extension import Extension
# extensions = [
#   Extension("float_phi_functions",
#   ["float_phi_functions.pyx"],
#   extra_compile_args=["/O1"])
# ]
# 
# setup(
#     ext_modules = cythonize(extensions)
# )

# from distutils.core import setup
# from Cython.Build import cythonize
# from Cython.Compiler.Options import directive_defaults
# from distutils.extension import Extension

# directive_defaults['profile'] = True
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

# sourcefiles = ['test.pyx']

# extensions = [Extension("test", sourcefiles, define_macros=[('CYTHON_TRACE', '1')])]

# setup(
#     ext_modules = cythonize(extensions)
# )
