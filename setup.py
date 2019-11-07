from __future__ import print_function
from setuptools import setup, find_packages
import io
import os
import sys

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

with open('README.md') as readme_file:
    readme = readme_file.read()

# TODO - txt to list
with open('requirements.txt') as reqs_file:
    reqs = reqs_file.read()

setup(
    name='predictit',
    version=0.23,
    url='https://github.com/Malachov/predictit',
    download_url='https://github.com/Malachov/predictit/archive/0.11.tar.gz',
    license='mit',
    author='Daniel Malachov',
    install_requires=[
            'numpy',
            'pandas',
            'scipy',
            'scikit_learn',
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
    long_description=readme,
    packages=['predictit'],
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
