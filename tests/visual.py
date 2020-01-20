""" Visual test on various components. Mostly for data preparation functions.
Just run and manually check results.

"""

#%%
import sys
import pathlib
import pandas as pd
import numpy as np

# TODO dataprep i na dataframe a numpy zvlast

script_dir = pathlib.Path(__file__).resolve()
lib_path_str = script_dir.parents[1].as_posix()
sys.path.insert(0, lib_path_str)

import predictit
import predictit.data_prep as dp

#data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3, 2, 4, 5, 6, 0, 0, 0, 0, 7, 3, 4, 55, 3, 2, 5, 6]])
data = np.array([1, 3, 5, 2, 3, 4, 5, 66, 3, 2, 4, 5, 6, 0, 0, 0, 0, 7, 3, 4, 55, 3, 2, 5, 6])
column_for_prediction = np.array([1, 3, 5, 2, 3, 4, 5, 66, 3, 2, 4, 5, 6, 0, 0, 0, 0, 7, 3, 4, 55, 3, 2, 5, 6])

#column_for_prediction = data[0]
data_multi_col = np.array([[1, 22, 3, 3], [5, 6, 7, 6], [8, 9, 10, 6]])
#data_multi_col = data_multi_col.astype('float64')

dataframe_raw = pd.DataFrame(data_multi_col)
predicted_column_index = 0


print('''

        #################
        ### Data_prep ###
        #################
    \n ''')

print('''
##################
### One column ###
##################\n
    ''')

print('### Remove outliers ### \n')
routl = dp.remove_outliers(data)
print(f"With outliers: \n {data} \n\nWith no outliers: \n{routl} \n")

print('### Difference transform ### \n')
differenced = dp.do_difference(column_for_prediction)
inv_differenced = dp.inverse_difference(differenced, column_for_prediction[0])
print(f"Original data: \n {column_for_prediction} \n\nDifferenced data: \n{differenced} \n\nBackward difference: \n{inv_differenced} \n")

print('### Standardize ### \n')
stand, scaler = dp.standardize(data)
print(f"Original: \n {data} \n\nStandardized: \n{stand} \n")

print('### Inverse standardization ### \n')
inv_stand = scaler.inverse_transform(stand)
print(f"Original: \n {data} \n")

print('### Split ### \n')
train, test = dp.split(data)
print(f"Original: \n {data} \n\nsplited train: \n{train} \n \n\nsplited test: \n{test} \n")

print('### Make sequences - One step ### \n')
seqs, Y = dp.make_sequences(data, n_steps_in=3, n_steps_out=1, constant=None, predicted_column_index=0, other_columns_lenght=None)
print(f"Original: \n {data} \n\nsequences: \n{seqs} \n\nY: \n{Y} \n")

print('### Make sequences - Batch ### \n')
seqs_2, Y_2 = dp.make_sequences(data, n_steps_in=2, n_steps_out=2, constant=None, predicted_column_index=0, other_columns_lenght=None)
print(f"Original: \n {data} \n\nsequences: \n{seqs_2} \n\nY: \n{Y_2} \n")

print('### Make x-input - One step ### \n')
X_input = dp.make_x_input(data, n_steps_in=3, constant=None, predicted_column_index=0, other_columns_lenght=None)
print(f"Original: \n {data} \n\nX_input: \n{X_input} \n")

print('''
####################
### More columns ###
####################\n
    ''')
print('### Clean nan values ### \n')

print('### Remove nan columns ### \n')
dataframe = dp.remove_nan_columns(dataframe_raw)
print(f"Original: \n{dataframe_raw} \n\nCleaned: \n{dataframe} \n")

print('### Remove outliers ### \n')
routl_m = dp.remove_outliers(dataframe, predicted_column_index=predicted_column_index, threshold = 1)
print(f"With outliers: \n {dataframe} \n\nWith no outliers: \n{routl_m} \n")

print('### Standardize ### \n')
stand_m, scaler_m = dp.standardize(data_multi_col)
print(f"Original: \n {data_multi_col} \n\nStandardized: \n{stand_m} \n")

print('### Inverse standardization ### \n')
inv_stand_m = scaler_m.inverse_transform(stand_m[0])
print(f"Original: \n {inv_stand_m} \n")

print('### Split ### \n')
train_m, test_m = dp.split(data_multi_col, predicts=2, predicted_column_index=0)
print(f"Original: \n {data_multi_col} \n\nsplited train: \n{train_m} \n \n\nsplited test: \n{test_m} \n")

print('### Make sequences - One step ### \n')
seqs_m, Y_m = dp.make_sequences(data_multi_col, n_steps_in=3, n_steps_out=1, constant=None, predicted_column_index=0, other_columns_lenght=2)
print(f"Original: \n {data_multi_col} \n\nsequences: \n{seqs_m} \n\nY: \n{Y_m} \n")

print('### Make sequences - One step ### \n')
seqs_m, Y_m = dp.make_sequences(data_multi_col, n_steps_in=3, n_steps_out=1, constant=None, predicted_column_index=0, other_columns_lenght=2)
print(f"Original: \n {data_multi_col} \n\nsequences: \n{seqs_m} \n\nY: \n{Y_m} \n")

print('### Make sequences - Batch ### \n')
seqs_m, Y_m = dp.make_sequences(data_multi_col, n_steps_in=2, n_steps_out=2, constant=None, predicted_column_index=0, other_columns_lenght=2)
print(f"Original: \n {data_multi_col} \n\nsequences: \n{seqs_m} \n\nY: \n{Y_m} \n")

print('''
        ###############
        ### Analyze ###
        ###############
    ''')

predictit.analyze.analyze_data(column_for_prediction)
#predictit.analyze.analyze_correlation(dataframe)











