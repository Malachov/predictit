#%%
import pandas as pd
import numpy as np

from pathlib import Path

this_path = Path(__file__).resolve().parents[1]
this_path_string = str(this_path)

# If used not as a library but as standalone framework, add path to be able to import predictit
sys.path.insert(0, this_path_string)

import predictit
from predictit import config
import predictit.data_prep as dp
from data_prep import make_sequences

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



# If you want to test one module and don't want to comment and uncomment one after one
config.used_models = {
    "AR (Autoregression)": models.ar,

}


#data_all = {'sin': test_data.data_test.gen_sin(data_length), 'sin1': test_data.data_test.gen_sin1(data_length), 'sign': test_data.data_test.gen_sign(data_length), 'random': test_data.data_test.gen_random(data_length)}
data_all = {'sign': test_data.data_test.gen_sign(data_length)}




if config.piclkeit:
    from predictit.test_data.pickle_test_data import pickle_data_all
    import pickle
    pickle_data_all(data_all, datalength=data_length)


if config.from_pickled:

    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / "test_data" / "pickled"

    for i, j in data_all.items():
        file_name = i + '.pickle'
        file_adress = data_folder / file_name
        try:
            with open(file_adress, "rb") as input_file:
                data = pickle.load(input_file)
            data_all[i] = data
        except Exception:
            warnings.warn(f"\n Error: {traceback.format_exc()} \n Warning - test data not loaded - First in config.py pickleit = 1, that save the data on disk, then load from pickled. \n")

results = {}
for i, j in data_all.items():
    config.plot_name = i
    try:
        results[i] = predictit.main.predict(data=j, return_model_criterion=1)

    except Exception:
        pass

