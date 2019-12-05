
#%%
import sys
import os
import inspect
import pathlib

script_dir = pathlib.Path(__file__).resolve()

lib_path_str = str(pathlib.Path(script_dir).parents[1])

sys.path.insert(0, lib_path_str)

import predictit

results = predictit.main.predict()
#desktop = os.path.normpath(os.path.expanduser("~/Desktop")) + '/plot.html'

#print(desktop)https://stackoverflow.com/questions/5606719/share-my-git-repository
