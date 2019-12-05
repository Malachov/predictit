#%%

import sys
import os
import inspect
import pathlib

script_dir = pathlib.Path(__file__).resolve()
lib_path_str = str(pathlib.Path(script_dir).parents[1]
sys.path.insert(0, lib_path_str)

import predictit

results = predictit.main.predict()

print(desktop)
