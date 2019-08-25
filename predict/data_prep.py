import pandas as pd
from statistics import stdev
import numpy as np
import pickle

def remove_outliers(data, predicted_column_index=0, threshold=3):
    """Remove values far from mean - probably errors
    ======

    Output:
    -------
    Cleaned data{arr, list}

    Arguments:
    -------
        data {arr, list} -- Data to be cleaned
        predicted_column_index {int} -- Number of predicted column (default: {0})
        threshold {f} -- Threshold that limit values defined by multiplier of standard deviation (default: {3})
    """
    
    data = np.array(data)
    data_shape = data.shape
    if len(data_shape) == 1:
        data = data[data > data.std()]

        return data

    else:
        data_mean = data[predicted_column_index].mean()
        data_std = data[predicted_column_index].std()

        counter = 0
        for i in range(len(data[predicted_column_index])):

            if abs(data[predicted_column_index, counter] - data_mean) > threshold * data_std:
                data = np.delete(data, counter, axis=1)
                counter -= 1
            counter += 1

    return data

def do_difference(dataset):
    """Transform data into neighbor difference. For example from [1, 2, 4] into [1, 2]
    
    Arguments:
        dataset {arr, list}
    """
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)

def inverse_difference(differenced_predictions, last_undiff_value):
    "Transform do_difference transform back"
    first = last_undiff_value + differenced_predictions[0]
    diff = [first]
    for i in range(1, len(differenced_predictions)):
        value = differenced_predictions[i] + diff[i - 1]
        diff.append(value)
    return np.array(diff)


def split(data, predicts = 7, predicted_column_index=0):
    '''Divide data set on train and test set
    Inputs
        data - dataframe, array, CSV or list
    Output
        train
        test
    example:
    train, test = split(data)
    '''

    if isinstance(data, pd.DataFrame):
        data = data.values

    data = np.array(data)
    data_shape = data.shape

    if len(data_shape) == 1: 
        train = data[:(len(data)-predicts)]
        test = data[-predicts:]

        if predicts > (len(data)):
            print('To few data, train/test not returned')
            return ([np.nan], [np.nan] * predicts)
            
    else:
        if predicts > (data_shape[1]):
            print('To few data, train/test not returned')
            return ([np.nan], np.array([np.nan] * predicts))

        train = data[:, :-predicts]

        test = data[predicted_column_index, -predicts:]

    return (train, test)

def data_clean(data):
    "Remove non-number columns"
    remember = 0
    for i in range(len(data)):
        try:
            int(data.iloc[5][remember]) 
        except Exception as exc:
            data.drop(data.columns[remember]    , axis=1, inplace=True)
            remember -= 1
    return data

def make_sequences(data, n_steps_in, n_steps_out=1, constant=None, predicted_column_index=0, other_columns_lenght=None):
    ''' From [1, 2, 3, 4, 5, 6, 7, 8] 

        Creates [1, 2, 3]  [4]
                [5, 6, 7]  [8]
    '''

    def make_seq(data, n_steps_in, n_steps_out=1, constant=None):
        X, y = list(), list()

        for i in range(len(data)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the data
            if out_end_ix > len(data):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)

        return X, y

    data = np.array(data)
    data_shape = data.shape

    if len(data_shape) == 1:
        X_list, y = make_seq(data=data, n_steps_in=n_steps_in, n_steps_out=n_steps_out, constant=constant)

    else: 

        for_prediction = data[predicted_column_index, :] 
        other_data = np.delete(data, predicted_column_index, axis=0)

        X_only, y = make_seq(for_prediction, n_steps_in=n_steps_in, n_steps_out=n_steps_out)

        other_columns = []
        X_list = []

        for i in range(len(other_data)):
            other_columns_sequentions, e = make_seq(other_data[i], n_steps_in=n_steps_in, n_steps_out=n_steps_out)
            other_columns.append(other_columns_sequentions)

        other_columns_array = np.array(other_columns)

        if other_columns_lenght:
            other_columns_array = other_columns_array[:, :, -other_columns_lenght:]

        for i in range(len(X_only)):

            other_columns_list = list(other_columns_array[:, i, :].reshape(-1))
            X_list.append(other_columns_list + list(X_only[i]))

    X = np.array(X_list)

    if constant:
        X = np.insert(X, 0, constant, axis=1)

    
    return np.array(X), np.array(y)


def make_x_input(data, n_steps_in, predicted_column_index=0, other_columns_lenght=None, constant=None):
    """Make input into model
    
    Arguments:
        data {arr, list}
        n_steps_in {int} -- Lags going into model
    
    Keyword Arguments:
        predicted_column_index {int} -- Index of predicted column (default: {0})
        other_columns_lenght {int} -- Lags of other columns (default: {None})
        constant {int} -- Can be 1 - constant added into every input (default: {None})
    """
    data = np.array(data)
    data_shape = data.shape

    if len(data_shape) == 1:

        x_input = data[-n_steps_in:]
        if constant:
            x_input = np.insert(x_input, 0, constant)

    else:
        for_prediction = data[predicted_column_index, -n_steps_in:] 
        other_data = np.delete(data, predicted_column_index, axis=0)

        if other_columns_lenght:
            other_cols = other_data[:, -other_columns_lenght:]
        else:
            other_cols = other_data[:, -n_steps_in:]

        x_input = np.concatenate((other_cols, for_prediction), axis=None)

        if constant:
            x_input = np.insert(x_input, 0, constant)

    x_input = x_input.reshape(1, -1)

    return x_input

def smooth():
    # TODO
    pass

'''
def normalize01(data):
    data = np.array(data)
    data = data.reshape(-1, 1)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data) 
    return scaled

def denormalize01(scaled):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    denormalized = scaler.inverse_transform(scaled)
    data = data.reshape(-1)
    return denormalized

def standardize(data):
    dataT = data.reshape(-1, 1)
    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(dataT)
    return scaled

def destandardize(scaled):
    scaler = preprocessing.StandardScaler()
    destandardized = scaler.inverse_transform(scaled)
    return destandardized
'''