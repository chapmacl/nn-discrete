#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages


setup(name='discrete_nn',
      version='1.0',
      description='Discrete neural networks with pytorch',
      author='Clay Chapman and Thiago Bell',
      packages=find_packages(),
      install_requires=["scikit-learn", "torch", "tqdm", "numpy", "torchvision"])

