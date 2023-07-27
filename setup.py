#!/usr/bin/env python
#-*- coding: utf-8 -*-

from setuptools import setup, find_packages

short_descr = "Riemannian L-Systems"
readme = open("README.md").read()

# find packages
pkgs = find_packages('src')

setup_kwds = dict(
    name='riemann_lsystems',
    version="0.1.0",
    description=short_descr,
    long_description=readme,
    author="Christophe Godin",
    author_email="christophe.godin@inria.fr",
    url='',
    license='LGPL-3.0',
    zip_safe=False,

    packages=pkgs,

    package_dir={'': 'src'},
    entry_points={},
    keywords='',
)

setup(**setup_kwds)