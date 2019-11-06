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
    name='predictit',
    version=0.1,
    url='https://github.com/Malachov/predict-it',
    download_url='https://github.com/Malachov/predict-it/archive/0.1.tar.gz',
    license='mit',
    author='Daniel Malachov',
    install_requires=[
            'numpy',
            'pandas',
            'scipy',
            'scikit_learn'
            'SQLAlchemy',
            'statsmodels',
            'pyodbc',
            'sklearn_extensions',
            'prettytable',
            'matplotlib',
            'plotly',
            'cufflinks',
            'seaborn'
      ],
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
