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

from prettytable import PrettyTable

config.lengths = 1

data_all = {'sin': test_data.data_test.gen_sin(data_length), 'Sign': test_data.data_test.gen_sign(data_length), 'Random data': test_data.data_test.gen_random(data_length)}
#data_all = {'sign': test_data.data_test.gen_sign(data_length)}

used_models = {
    "AR (Autoregression)": predictit.models.ar,
    #"ARMA": predictit.models.arma,
    #"SARIMAX (Seasonal ARIMA)": predictit.models.sarima,
}

config.criterion = 'rmse'

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
        result = predictit.main.predict(data=j, return_model_criterion=1)
        results[i] = (result - np.nanmin(result)) / (np.nanmax(result) - np.nanmin(result))

    except Exception:
        results[i] = np.nan


results_array = np.stack((results.values()), axis=0)


all_data_average = np.nanmean(results_array, axis=0)

models_best_results = np.nanmin(all_data_average, axis=1)
best_compared_model = int(np.nanargmin(models_best_results))
best_compared_model_name = list(config.used_models.keys())[best_compared_model]

all_lengths_average = np.nanmean(all_data_average, axis=0)
best_all_lengths_index = np.nanargmin(all_lengths_average)


models_names = list(config.used_models.keys())

models_table = PrettyTable()
models_table.field_names = ["Model", "Average standardized {} error".format(config.criterion)]

# Fill the table
for i, j in enumerate(models_names):
    models_table.add_row([models_names[i], models_best_results[i]])


print(f'\n {models_table} \n')


print(f"\n\nBest model is {best_compared_model_name}")
print(f"\n\nBest data length index is {best_all_lengths_index}")

#%%

a = np.array([[9, 1], [3, 0.2], [1, 88]])
b = np.array([[1, 2], [7, 2], [1, 2]])
c = np.array([[1, 66], [3, 2], [3, 2]])
d = np.array([[10, 2], [3, 2], [1, 2]])

results_array = np.stack((a, b, c, d), axis=0)
results_array_orig = results_array.copy()




all_data_average = np.nanmean(results_array, axis=0)

all_lengths_average = np.nanmean(all_data_average, axis=0)
best_all_lengths_index = np.nanargmin(all_lengths_average)

models_best_results = np.nanmin(all_data_average, axis=1)
best_compared_model = np.nanargmin(models_best_results)


print(f"Best model is {best_all_lengths_index}")
print(f"Best data length index is {best_all_lengths_index}")
