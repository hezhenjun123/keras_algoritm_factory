import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("utilities/bbox_overlap.pyx", compiler_directives={'language_level' : "3"}), include_dirs=[numpy.get_include()])