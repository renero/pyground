# -*- coding: utf-8 -*-
# Copyright (C) 2021 Jesus Renero
# Licence: Apache 2.0
import codecs
import os.path

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

with open('README.md', encoding="utf-8") as f:
    long_description = f.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def setup_package():
    setup(name='pyground',
          version=get_version("pyground/__init__.py"),
          description='Utils for python projects',
          packages=find_packages(),
          url='https://github.com/renero/pyground',
          package_data={'': ['**/*.data', '**/*.csv']},
          install_requires=['numpy', 'pandas', 'prettytable', 'argparse',
            'joblib','sklearn','scipy','scikit','setuptools','pytest',
            'PyYAML','networkx','ipython','pyground','pydot','boto3',
            'sagemaker','matplotlib','pydotplus'],
          include_package_data=True,
          author='J. Renero',
          author_email='hergestridge@gmail.com',
          license='Apache 2.0')


if __name__ == '__main__':
    setup_package()
