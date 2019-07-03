# Python C-API CUDA Tutorial

The Python C-API lets you write functions in C and call them like normal Python functions. This is super useful for computationally heavy code, and it can even be used to call CUDA kernels from Python. There don’t seem to be a good tutorial about how to do this, so we’re going to walk through the process of defining a couple C functions with support for CUDA and calling them natively from Python.

# Usage:

```bash
git clone https://github.com/ja3067/Python-C-API-CUDA-Tutorial.git
cd Python-C-API-CUDA-Tutorial
make
```

Run `make` to compile the module and run tests. It should complain if it's unable to find the correct dependencies. To use some of the example functions, run

```python
>>> import myModule
>>> myModule.helloworld()
Hello World
>>> import numpy as np
>>> myModule.trace(np.random.randn(4, 4))
0.9221652081491398
>>> myModule.vector_add(np.random.randn(500), np.random.randn(500))
[0.01, ..., 0.54]
>>> myModule.vector_add_cpu(np.random.randn(500), np.random.randn(500))
[0.07, ..., 0.24]
```

# More Information

This repository accompanies a number of blog posts about CUDA. Read part 1 [here](https://medium.com/@jacobaustin132/creating-c-extensions-for-python-with-numpy-and-cuda-part-1-4941118982cd) and part 2 [here](https://medium.com/@jacobaustin132/creating-c-extensions-for-python-with-numpy-and-cuda-part-2-46abfc392e07). 
