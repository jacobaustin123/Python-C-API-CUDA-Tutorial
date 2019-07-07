import os
from distutils.core import setup, Extension
import numpy as np

setup(name = 'myModule', version = '1.1',  \
   ext_modules = [
      Extension('myModule', ['myModule.c'], 
      include_dirs=[np.get_include()],
)])
