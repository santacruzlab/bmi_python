brain-python-interface (a.k.a. bmi3d)
====================================
**This is the unstable python 3 branch under development.** 

This package contains Python code to run brain-machine interface (BMIs) tasks as well as other neurophysiology experiments. This package differs from other similar packages (e.g., BCI2000--http://www.schalklab.org/research/bci2000) in that it is primarily intended for intracortical BMI experiments. 

This package was originally written for Python 2. The code has been updated to run with Python 3, but is currently unstable.

HTML documentation for the code can be found at http://carmenalab.github.io/bmi3d_docs/

Getting started 
---------------
# Dependencies
## Linux/OS X
(none at this time)

## Windows
Visual C++ Build tools (for the 'traits' package)

# Installation
```bash
git clone -b unstable_py3 https://github.com/carmenalab/brain-python-interface.git
cd brain-python-interface
pip3 install -r requirements.txt
pip3 install -e .
```
