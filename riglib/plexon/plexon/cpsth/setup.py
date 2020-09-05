#! /usr/bin/env python

# System imports
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# ezrange extension module
_psth = Extension("cpsth",
                   ["cpsth.pyx", 'psth.c'],
#                   define_macros = [('DEBUG', None)],
#                   extra_compile_args=["-g"],
#                   extra_link_args=["-g"],
                   )

# ezrange setup
setup(  name        = "psth generator",
        description = "Generates the PSTH from the raw buffer made by plexnet",
        author      = "James Gao",
        version     = "1.0",
        cmdclass = {'build_ext': build_ext},
        ext_modules = [_psth]
        )
