""" Visual test on various components. Mostly for data preparation functions.
Just run and manually check results.

"""

#%%
import sys
import pathlib
import pandas as pd
import numpy as np
import time

# TODO dataprep i na dataframe a numpy zvlast

script_dir = pathlib.Path(__file__).resolve()
lib_path_str = script_dir.parents[1].as_posix()
sys.path.insert(0, lib_path_str)

import predictit
import predictit.data_preprocessing as dp

if predictit.misc._JUPYTER:
    get_ipython().run_line_magic('matplotlib', 'inline')

# Data must have 2 dimensions. If you have only one column, reshape(1, -1)!!!
data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3, 2, 4, 5, 6, 0, 0, 0, 0, 7, 3, 4, 55, 3, 2]])

column_for_prediction = data[0]

data_multi_col = np.array([[1, 22, 3, 3, 5, 8, 3, 3, 5, 8], [5, 6, 7, 6, 7, 8, 3, 3, 5, 8], [8, 9, 10, 6, 8, 8, 3, 3, 5, 8]])
#data_multi_col = data_multi_col.astype('float64')

dataframe_raw = pd.DataFrame(data_multi_col)
predicted_column_index = 0

# Define some longer functions, that is bad to compute in f strings...
seqs, Y, x_input, test_inputs = predictit.define_inputs.make_sequences(data, n_steps_in=6, n_steps_out=1, constant=1)
seqs_2, Y_2, x_input2, test_inputs2 = predictit.define_inputs.make_sequences(data, n_steps_in=4, n_steps_out=2, constant=0)
seqs_m, Y_m, x_input_m, test_inputs_m = predictit.define_inputs.make_sequences(data_multi_col, n_steps_in=4, n_steps_out=1, other_columns_length=None, constant=1)
seqs_2_m, Y_2_m, x_input2_m, test_inputs2_m = predictit.define_inputs.make_sequences(data_multi_col, n_steps_in=3, n_steps_out=2, other_columns_length=1, constant=0)

print("""
        ###############
        ### Analyze ###
        ###############\n
""")

predictit.analyze.analyze_data(column_for_prediction)

print(f"""

        #################
        ### Data_prep ###
        #################

    ##################
    ### One column ###
    ##################\n

### Remove outliers ###\n

With outliers: \n {data} \n\nWith no outliers: \n{dp.remove_outliers(data)} \n

### Difference transform ### \n
Original data: \n {column_for_prediction} \n\nDifferenced data: \n{dp.do_difference(column_for_prediction)} \n\nBackward difference: \n{dp.inverse_difference(dp.do_difference(column_for_prediction), column_for_prediction[0])} \n

### Standardize ### \n
Original: \n {data} \n\nStandardized: \n{dp.standardize(data)[0]} \n
Inverse standardization: \n{dp.standardize(data)[1].inverse_transform(dp.standardize(data)[0])} \n

### Split ### \n
Original: \n {data} \n\nsplited train: \n{dp.split(data)[0]} \n \n\nsplited test: \n{dp.split(data)[1]} \n

### Make sequences - n_steps_in = 6, n_steps_out = 1, constant = 1 ### \n
Original: \n {data} \n\nsequences: \n{seqs} \n\nY: \n{Y} \n\n\nx_input:{x_input} \n

### Make batch sequences - n_steps_in = 4, n_steps_out = 2, constant = 0 ### \n
Original: \n {data} \n\nsequences: \n{seqs_2} \n\nY: \n{Y_2} \n \nx_input: \n{x_input2} \n

    ####################
    ### More columns ###
    ####################\n

### Remove outliers ### \n
With outliers: \n {data_multi_col} \n\nWith no outliers: \n{dp.remove_outliers(data_multi_col, predicted_column_index=predicted_column_index, threshold = 1)} \n

### Standardize ### \n
Original: \n {data_multi_col} \n\nStandardized: \n{dp.standardize(data_multi_col)[0]} \n
Inverse standardization: \n {dp.standardize(data_multi_col)[1].inverse_transform(dp.standardize(data_multi_col)[0][0])} \n

### Split ### \n
Original: \n {data_multi_col} \n\nsplited train: \n{dp.split(data_multi_col, predicts=2, predicted_column_index=0)[0]} \n \n\nsplited test: \n{dp.split(data_multi_col, predicts=2, predicted_column_index=0)[1]} \n

### Make sequences - n_steps_in=4, n_steps_out=1, other_columns_length=None, constant=1 ### \n
Original: \n {data_multi_col} \n\nsequences: \n{seqs_m} \n\nY: \n{Y_m} \nx_input: \n\n{x_input_m} \n

### Make batch sequences - n_steps_in=3, n_steps_out=2, other_columns_length=1, constant=0 ### \n
Original: \n {data_multi_col} \n\nsequences: \n{seqs_2_m} \n\nY: \n{Y_2_m} \nx_input: \n\n{x_input2_m} \n

        ###########################
        ### Data_postprocessing ###
        ###########################\n

### Fitt power transform ### \n
Original: \n {data}, original std = {data.std()}, original mean = {data.mean()} \n\ntransformed: \n{dp.fitted_power_transform(data, 10, 10)} \n\ntransformed std = {dp.fitted_power_transform(data, 10, 10).std()},
transformed mean = {dp.fitted_power_transform(data, 10, 10).mean()} (shoud be 10 and 10)\n

""")
