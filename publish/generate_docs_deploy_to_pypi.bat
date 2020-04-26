@ECHO OFF

cd ..\docs

sphinx-apidoc -f -e -o source/ ../predictit

cd ..

python setup.py sdist bdist_wheel
twine upload dist/*

rmdir /s /Q dist
rmdir /s /Q predictit.egg-info
rmdir /s /Q build

ECHO Job done

PAUSE