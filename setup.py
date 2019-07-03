import os
from distutils.core import setup, Extension
import numpy as np

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

setup(name = 'myModule', version = '1.0',  \
   ext_modules = [
      Extension('myModule', ['myModule.c'], 
      include_dirs=[np.get_include(), "/usr/local/cuda-10.0/include"],
      libraries=["vectoradd", "cudart"],
      library_dirs = [".", "/usr/local/cuda-10.0/lib64/"]
      )])