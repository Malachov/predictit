#!/bin/bash

cd ../docs/source

shopt -s extglob

rm -v !("conf.py"|"index.rst"|"tooc.rst") 

sphinx-apidoc -f -e -o ../source/ ../../predictit

rm predictit.tempCodeRunnerFile.rst

cd ..

make html

cd ..

# python setup.py sdist bdist_wheel
# twine upload dist/* -p ${TWINE_PASSWORD}

# rm -r -f dist
# rm -r -f predictit.egg-info
# rm -r -f build

echo 'Job done'

$SHELL
