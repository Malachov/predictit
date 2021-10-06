from setuptools import setup, find_packages
import pkg_resources
import predictit

version = predictit.__version__

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as f:
    used_requirements = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

setup(
    name="predictit",
    version=version,
    url="https://github.com/Malachov/predictit",
    license="mit",
    author="Daniel Malachov",
    author_email="malachovd@seznam.cz",
    install_requires=used_requirements,
    description="Library/framework for making predictions.",
    long_description_content_type="text/markdown",
    long_description=readme,
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    platforms="any",
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Natural Language :: English",
        "Environment :: Other Environment",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
    ],
    extras_require={},
)
