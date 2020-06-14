#!/usr/bin/env python
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

# explicitly config
test_args = [
    '--cov-report=term',
    '--cov-report=html',
    '--cov=vanidl',
    'tests'
]


class PyTest(TestCommand):
    user_options = []

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = test_args

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(name='vanidl',
      version='0.0.1',
      description='Deep Learning Analyzer for HPC Applications',
      author='Hariharan Devarajan',
      author_email='hdevarajan@hawk.iit.edu',
      url='https://github.com/hariharan-devarajan/dlprofiler',
      download_url='https://github.com/hariharan-devarajan/dlprofiler/tarball/0.0.1',
      license='MIT',
      packages=find_packages(exclude=['tests*']),
      install_requires=[
          'numpy',
          'pandas',
          'h5py',
          'tensorflow'
      ],
      test_suite='tests',
      cmdclass={'test': PyTest},
      classifiers=[
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      keywords=[
          'VaniDL',
          'Darshan Analysis',
          'Scientific',
          'TensorFlow',
          'Deep Learning',
          'Machine Learning',
          'Neural Networks',
          'AI'
      ]
      )
