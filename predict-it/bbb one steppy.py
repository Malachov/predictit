#%%
import sys
sys.path.insert(0, 'C:\VSCOODE\Diplomka')
sys.path.insert(0, 'C:\VSCOODE\Diplomka\models')

a = 1
if a == 1:



    from sklearn import linear_model
    from models import pokus
    from config import *
    from pathlib import Path
    import pandas as pd
    import os
    import pickle
    import analyze
    from sklearn.multioutput import MultiOutputRegressor

    import config
    from data_prep import make_sequences
    import numpy as np
    import matplotlib.pyplot as plt
    cwd = os.getcwd()
    path = Path(cwd)
    os.chdir(path)

    data_folder = Path('test_data')

    try:
        data_for_predicts_csv = pd.read_csv(config.csv_adress, header=0, index_col=0)
    except Exception as exc:
        print("\n \t ERROR - Data load failed - Setup CSV adress in config and column name \n")

    data_shape = data_for_predicts_csv.shape

    data_for_predicts_csv.index = pd.to_datetime(data_for_predicts_csv.index)

    column_for_prediction_all = np.array(data_for_predicts_csv[config.predicted_columns_names])
    column_for_prediction = column_for_prediction_all[-config.datalength:]
    predicted_column_index = data_for_predicts_csv.columns.get_loc(config.predicted_columns_names)

    if data_shape[0] > data_shape[1]:
        data_for_predicts_nd = data_for_predicts_csv.values.T
    else:
        data_for_predicts_nd = data_for_predicts_csv.values

    data_for_predicts = np.array(data_for_predicts_nd)
    data = data_for_predicts[-config.datalength:, :]
n_steps = 3



predicts = 7



data_shape = np.array(data).shape

if len(data_shape) == 1:
    X, y = make_sequences(data, n_steps)

else:
    for_prediction = data[predicted_column_index, :] 
    data = np.delete(data, predicted_column_index, axis=0)

    # Vymaže sloupce které neobsahují čísla
    remember = 0
    for i in range(len(data)):
        try:
            int(data[i][1]) 
        except Exception as exc:
            data = np.delete(data, i, axis=0)
            remember -= 1


    sequentions = []

    for i in data:
        X_other_columns, y_other_columns = make_sequences(i, n_steps)
        sequentions.append(X_other_columns)

    X_only, y = make_sequences(for_prediction, n_steps)

    almost_serialized_sequentions = [[] for i in range(len(sequentions[0]))]
    serialized_sequentions = [[] for i in range(len(sequentions[0]))]

    for i in range(len(sequentions[0])):
        for j in sequentions:
            almost_serialized_sequentions[i].append(j[i])

    for i in range(len(almost_serialized_sequentions)):
        almost_serialized_sequentions[i].append(X_only[i])

    for i in range(len(almost_serialized_sequentions)):
        serialized_sequentions[i] = np.array(almost_serialized_sequentions[i]).reshape(-1)

    X = np.array(serialized_sequentions)

reg_model = linear_model.BayesianRidge()
reg = MultiOutputRegressor(reg_model)
reg.fit(X, y) 

predictions = []
nucolumn = []
nu_data_shape = data.shape
import models

for i in range(predicts):
    if data_shape == 1:
        x_input = data[-n_steps_in:]
        x_input = x_input.reshape((1, n_steps))

    else:
        x_input = []
        x_input.append(list(for_prediction[-n_steps_in:])[::-1])
        for j in reversed(data):
            reverseded = list(j[-n_steps_in:])[::-1]
            x_input.append(reverseded)
        x_input = np.array([x_input]).reshape(1, 9)

    yhat = reg.predict(x_input)
    predictions.append(yhat)
    
    for_prediction = np.append(for_prediction, yhat)
    
    


    for j in data:
        nucolumn.append(models.ar(j, predicts=1))


    nucolumn_T = np.array(nucolumn).reshape(nu_data_shape[0], 1)
    data = np.append(data, nucolumn_T, axis=1)
    nucolumn = []
