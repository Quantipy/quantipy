#!/usr/bin/env python
# coding: utf-8

import sys
from setuptools import setup, find_packages

versions = dict(
    numpy='1.13.0',
    pandas='0.19.2',
    ftfy='4.4.3',
    watchdog='0.10.3',
    decorator='5.0.0'
)

precisions = dict(
    numpy='==',
    pandas='==',
    ftfy='==',
    watchdog='==',
    decorator='<'
)

libs = [
    'numpy',
    'scipy',
    'pandas',
    'ftfy',
    'xmltodict',
    'lxml',
    'xlsxwriter',
    'prettytable',
    'decorator',
    'watchdog',
    'requests',
    'python-pptx',
    'functools32'
]

def version_libs(libs, precisions, versions):
    return [lib + precisions[lib] + versions[lib]
            if lib in versions.keys() else lib
            for lib in libs]

if sys.platform == 'win32':
    INSTALL_REQUIRES = version_libs(libs[2:], precisions, versions)
else:
    INSTALL_REQUIRES = version_libs(libs, precisions, versions)

setup(
    name='quantipy',
    version='0.1.1',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
)
