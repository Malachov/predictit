#!/bin/sh

cd ../docs

sphinx-apidoc -f -e -o source/ ../predictit

cd ..

python setup.py sdist bdist_wheel
twine upload dist/*

rm -r -f dist
rm -r -f predictit.egg-info
rm -r -f build

ECHO Job done
