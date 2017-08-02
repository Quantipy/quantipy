#!/usr/bin/env python
# coding: utf-8

import sys
from setuptools import setup, find_packages

versions = dict(numpy='1.11.3',
                scipy='0.18.1',
                pandas='0.19.2',
                ftfy='4.4.3')

precisions = dict(numpy='==',
                  scipy='==',
                  pandas='==',
                  ftfy='==')

libs = ['numpy',
        'scipy',
        'pandas',
        'ftfy',
        'xmltodict',
        'lxml',
        'xlsxwriter',
        'pillow',
        'prettytable',
        'decorator',
        'watchdog',
        'requests',
        'python-pptx']

def version_libs(libs, precisions, versions):
    return [lib + precisions[lib] + versions[lib]
            if lib in versions.keys() else lib
            for lib in libs]

if sys.platform == 'win32':
    INSTALL_REQUIRES = version_libs(libs[2:], precisions, versions)
else:
    INSTALL_REQUIRES = version_libs(libs, precisions, versions)

setup(name='quantipy',
      version='0.1.1',
      # author='',
      # author_email='',
      packages=find_packages(exclude=['tests', 'quantipy.sandbox']),
      include_package_data=True,
      install_requires=INSTALL_REQUIRES,
      )
