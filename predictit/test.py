#%%
data = 2
import datetime
if not isinstance(data, np.ndarray):
    print(21)
#%%
import pandas as pd
import numpy as np

import predictit_library.predictit as predictit
from predictit.data_prep import make_sequences
from predictit import config

from pathlib import Path
from sklearn import linear_model
import sklearn
import numpy as np
from sklearn.multioutput import MultiOutputRegressor

from predictit.data_prep import make_sequences, make_x_input

data = np.random.randn(1, 100)

import warnings
import traceback


from predictit import test_data
data_length = 1000
predicted_column_index = 0
data_all_pickle = {'sin': test_data.data_test.gen_sin(data_length), 'sign': test_data.data_test.gen_sign(data_length)}#, 'random': test_data.data_test.gen_random(data_length)}

if config.piclkeit:
    from predictit.test_data.pickle_test_data import pickle_data_all
    import pickle
    pickle_data_all(data_all_pickle, datalength=data_length)

results = {}

if config.from_pickled:
    data_all = {}

    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / "test_data" / "pickled"

    for i, j in data_all_pickle.items():
        file_name = i + '.pickle'
        file_adress = data_folder / file_name
        try:
            with open(file_adress, "rb") as input_file:
                data = pickle.load(input_file)
            data_all[i] = data
        except Exception:
            warnings.warn(f"\n Error: {traceback.format_exc()} \n Warning - test data not loaded - First in config.py pickleit = 1, that save the data on disk, then load from pickled. \n")

        results[i] = predictit.main.predict(return_model_criterion=1)
        

else:
    data_all = data_all_pickle

