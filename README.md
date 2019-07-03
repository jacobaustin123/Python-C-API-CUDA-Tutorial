# CUDA-Python-CAPI-Tutorial

The Python C-API lets you write functions in C and call them like normal Python functions. This is super useful for computationally heavy code, and it can even be used to call CUDA kernels from Python. There don’t seem to be a good tutorial about how to do this, so we’re going to walk through the process of defining a couple C functions with support for CUDA and calling them natively from Python.

# Usage:

This repository contains everything you need to compile a C extension for Python with CUDA and Numpy support. Just clone the repository and run make python to build and install the Python module. 
