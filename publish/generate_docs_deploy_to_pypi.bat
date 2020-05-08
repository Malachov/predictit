@ECHO OFF


cd ..\docs\source

for %%i in (*.*) do if not "%%i"=="doc1.txt" if not "%%i"=="doc2.txt" del /q "%%i"

sphinx-apidoc -f -e -o ../source/ ../../predictit

rm predictit.tempCodeRunnerFile.rst

cd ../..

python setup.py sdist bdist_wheel
twine upload dist/*

rmdir /s /Q dist
rmdir /s /Q predictit.egg-info
rmdir /s /Q build

ECHO Job done

PAUSE
