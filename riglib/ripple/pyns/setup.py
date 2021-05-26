#!/usr/bin/env python
from distutils.core import setup

classifiers =[
              'Development Status :: 4 - Beta',
              'Operating System :: OS Independent',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering',
              'Topic :: Software Development :: Libraries :: Python Modules',
              'License :: OSI Approved :: GNU General Public License (GPL)'
              ]

setup(name="Py Neuroshare",
      version="0.4.2",
      description="Python port of the Neuroshare API",
      author="Ripple LLC",
      author_email="support@rppl.com",
      packages=["pyns"],
      classifiers=classifiers
)
        




