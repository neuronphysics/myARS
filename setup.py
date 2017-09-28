from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext
extra_compile_args = ['-fPIC']
extra_link_args = ['-Wall']
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules=[
        Extension("ars", 
                  sources=["ars.pyx"],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args)
    ],
gdb_debug=True)


