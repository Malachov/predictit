#%%
from setuptools import setup
import sys
import pathlib

lib_path = pathlib.Path(__file__).resolve().parents[0]
lib_path_str = lib_path.as_posix()
sys.path.insert(0, lib_path_str)

import predictit

version = predictit.__version__

with open(lib_path / 'README.md') as readme_file:
    readme = readme_file.read()

with open(lib_path / 'requirements.txt', 'r') as f:
    myreqs = [line.strip() for line in f]

setup(
    name='predictit',
    version=version,
    url='https://github.com/Malachov/predictit',
    license='mit',
    author='Daniel Malachov',
    author_email='malachovd@seznam.cz',
    install_requires=myreqs,
    description='Library/framework for making predictions.',
    long_description_content_type='text/markdown',
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
