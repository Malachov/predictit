from setuptools import setup, find_packages
import io
import os
import sys

import predictit

version = predictit.__version__

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

with open('requirements.txt', 'r') as f:
    myreqs = [line.strip() for line in f]

setup(
    name='predictit',
    version=version,
    url='https://github.com/Malachov/predictit',
    download_url='https://github.com/Malachov/predictit/archive/0.11.tar.gz',
    license='mit',
    author='Daniel Malachov',
    install_requires=myreqs,
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
