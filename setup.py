# -*- coding: utf-8 -*-
# Copyright (C) 2021 Jesus Renero
# Licence: Apache 2.0

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


with open('README.md', encoding="utf-8") as f:
    long_description = f.read()


def setup_package():
    setup(name='pyground',
          version='0.1.0',
          description='Utils for python projects',
          packages=find_packages(),
          url='https://github.com/renero/ground',
          package_data={'': ['**/*.data', '**/*.csv']},
          install_requires=['numpy', 'pandas'],
          include_package_data=True,
          author='Jesus Renero',
          author_email='hergestridge@gmail.com',
          license='Apache 2.0')


if __name__ == '__main__':
    setup_package()
