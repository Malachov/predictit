#!/bin/sh

cd ../docs/source

rm -v !("conf.py"|"index.rst"|"tooc.rst")

cd ..

sphinx-apidoc -f -e -o source/ ../predictit
rm predictit.tempCodeRunnerFile.rst

cd ..

python setup.py sdist bdist_wheel
twine upload dist/*

rm -r -f dist
rm -r -f predictit.egg-info
rm -r -f build

echo 'Job done'
