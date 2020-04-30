#!/bin/sh

cd ../docs/source

shopt -s extglob

find . -type f -not \( -name "conf.py" -o -name "index.rst" -o -name "tooc.rst" \) -delete

sphinx-apidoc -f -e -o ../source/ ../../predictit

rm predictit.tempCodeRunnerFile.rst

cd ../..

python setup.py sdist bdist_wheel
twine upload dist/*

rm -r -f dist
rm -r -f predictit.egg-info
rm -r -f build

echo 'Job done'

$SHELL
