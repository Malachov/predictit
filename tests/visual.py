""" Visual test on various components. Mostly for data preparation functions.
Just run and manually check results."""

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

data = np.array([1, 3, 5, 2, 3, 4, 5, 66, 3, 2, 4, 5, 6, 0, 0, 0, 0, 7, 3, 4, 55, 3, 2, 5, 6])
data_multi_col = np.array([[1, 22, 3, 3], [5, 6, 7, 6], [8, 9, 10, 6]])
#data_multi_col = data_multi_col.astype('float64')

dataframe_raw = pd.DataFrame(data_multi_col)
predicted_column_index = 0

print('''

        #################
        ### Data_prep ###
        #################
    \n ''')

print('### One column ### \n')

print('# Remove outliers # \n')
routl = dp.remove_outliers(data)
print(f"With outliers: \n {data} \n\nWith no outliers: \n{routl} \n")

print('# Difference transform # \n')
differenced = dp.do_difference(data)
inv_differenced = dp.inverse_difference(differenced, data[0])
print(f"Original data: \n {data} \n\nDifferenced data: \n{differenced} \n\nBackward difference: \n{inv_differenced} \n")

print('# Standardize # \n')
stand, scaler = dp.standardize(data)
print(f"Original: \n {data} \n\nStandardized: \n{stand} \n")

print('# Inverse standardization # \n')
inv_stand = scaler.inverse_transform(stand)
print(f"Original: \n {data} \n")

print('### More columns ### \n')
print('# Clean nan values # \n')

print('# Remove nan columns # \n')
dataframe = dp.remove_nan_columns(dataframe_raw)
print(f"Original: \n{dataframe_raw} \n\nCleaned: \n{dataframe} \n")

print('# Remove outliers # \n')
routl_m = dp.remove_outliers(dataframe, predicted_column_index=predicted_column_index, threshold = 1)
print(f"With outliers: \n {dataframe} \n\nWith no outliers: \n{routl_m} \n")

print('# Standardize # \n')
stand_m, scaler_m = dp.standardize(data_multi_col)
print(f"Original: \n {data_multi_col} \n\nStandardized: \n{stand_m} \n")

print('# Inverse standardization # \n')
inv_stand_m = scaler_m.inverse_transform(stand_m[0])
print(f"Original: \n {inv_stand_m} \n")

print('''
        ###############
        ### Analyze ###
        ###############
    ''')

predictit.analyze.analyze_data(data)
#predictit.analyze.analyze_correlation(dataframe)
