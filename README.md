# Python C-API CUDA Tutorial

The Python C-API lets you write functions in C and call them like normal Python functions. This is super useful for computationally heavy code, and it can even be used to call CUDA kernels from Python. There don’t seem to be a good tutorial about how to do this, so we’re going to walk through the process of defining a couple C functions with support for CUDA and calling them natively from Python.

# Usage:

Run `make` to compile the module and run tests. It should complain if it's unable to find the correct dependencies.

# More Information

This repository accompanies a number of blog posts about CUDA. Read part 1 [here](https://medium.com/@jacobaustin132/creating-c-extensions-for-python-with-numpy-and-cuda-part-1-4941118982cd) and part 2 [here](https://medium.com/@jacobaustin132/creating-c-extensions-for-python-with-numpy-and-cuda-part-2-46abfc392e07). 
