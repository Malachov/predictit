from __future__ import print_function
from setuptools import setup, find_packages
import io
import os
import sys

import predict-it

here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md')

setup(
    name='predict-it',
    version=predict-it.__version__,
    url='https://github.com/Malachov/predict-it',
    license='mit',
    author='Daniel Malachov',
    install_requires=[],
    author_email='malachovd@seznam.cz',
    description='Library/framework for making predictions.',
    long_description=' Automatically choose best of 20 models (ARIMA, regressions, LSTM...). Preprocess data and chose optimal parameters of predictions.',
    packages=['predict-it'],
    include_package_data=True,
    platforms='any',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Environment :: Other Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        ],
    extras_require={
    }
)
