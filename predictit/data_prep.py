"""Module for data preprocessing. Contain functions for removing outliers, NaN columns or 
function for making inputs and outputs from time series if you want to see how functions work -
working examples with printed results are in tests - visual.py.

"""

import pandas as pd
import numpy as np


def remove_nan_columns(data):
    """Remove columns that are not a numbers.

    Args:
        data (np.ndarray): Time series data.

    Returns:
        np.ndarray: Cleaned time series data.
    """

    remember = 0
    cleaned = data.copy()

    if isinstance(cleaned, pd.DataFrame):
        for i in cleaned.columns:
            try:

                int(cleaned.iloc[2, remember])
                remember += 1

            except Exception:
                cleaned.drop([i], axis=1, inplace=True)

    if isinstance(cleaned, np.ndarray):
        for i in range(len(data)):
            try:

                int(cleaned[remember, 1])
                remember +=1

            except Exception:
                cleaned = np.delete(cleaned, remember, axis=0)

    return cleaned


def keep_corelated_data(data, predicted_column_index=0, threshold=0.2):
    """Remove columns that are not corelated enough to predicted columns.

    Args:
        data (np.array, pd.DataFrame): Time series data.
        predicted_column_index (int, optional): Column that is predicted. Defaults to 0.
        threshold (float, optional): After correlation matrix is evaluated, all columns that are correlated less than threshold are deleted. Defaults to 0.2.

    Returns:
        np.array, pd.DataFrame: Data with no columns that are not corelated with predicted column.
    """

    if isinstance(data, np.ndarray):
        corr = pd.DataFrame(np.corrcoef(data))
        range_array = np.array(range(len(corr)))
        names_to_del = range_array[corr[0] <= abs(threshold)]
        data = np.delete(data, names_to_del, axis=0)

    if isinstance(data, pd.DataFrame):
        corr = data.corr()
        names_to_del = list(corr[corr[corr.columns[predicted_column_index]] <= abs(0.6)].index)
        data.drop(columns=names_to_del, inplace=True)

    return data


def remove_outliers(data, predicted_column_index=0, threshold=3):
    """Remove values far from mean - probably errors.

    Args:
        data (np.array, pd.DataFrame, list): Time series data.
        predicted_column_index (int, optional): Column that is predicted. Defaults to 0.
        threshold (int, optional): How many times must be standard deviation from mean to be ignored. Defaults to 3.

    Returns:
        np.array, pd.DataFrame: Cleaned data.

    Examples:

        >>> data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]])
        >>> print(remove_outliers(data))
        [1, 3, 5, 2, 3, 4, 5, 3]
    """

    if isinstance(data, pd.DataFrame):

        data_mean = data.iloc[:, predicted_column_index].mean()
        data_std = data.iloc[:, predicted_column_index].std()

        data = data[abs(data[data.columns[0]] - data_mean) < threshold * data_std]

    else:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        data_shape = data.shape

        if len(data_shape) == 1:
            data = data[(data - data.mean()) < threshold * data.std()]
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
    """Transform data into neighbor difference. For example from [1, 2, 4] into [1, 2].

    Args:
        dataset (np.ndarray): One-dimensional numpy array.

    Returns:
        np.ndarray: Differenced data.

    Examples:

        >>> data = np.array([1, 3, 5, 2])
        >>> print(remove_outliers(data))
        [2, 2, -3]
    """

    assert len(np.shape(dataset)) == 1, 'Data input must be one-dimensional.'
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)


def inverse_difference(differenced_predictions, last_undiff_value):
    """Transform do_difference transform back.
    
    Args:
        differenced_predictions (np.ndarray): Differenced data from do_difference function.
        last_undiff_value (float): First value to computer the rest.
    
    Returns:
        np.ndarray: Normal data, not the additive series.

    Examples:

        >>> data = np.array([1, 1, 1, 1])
        >>> print(inverse_difference(data, 1))
        [2, 3, 4, 5]
    """

    first = last_undiff_value + differenced_predictions[0]
    diff = [first]
    for i in range(1, len(differenced_predictions)):
        value = differenced_predictions[i] + diff[i - 1]
        diff.append(value)
    return np.array(diff)


def standardize(data, predicted_column_index=0, standardizer='standardize'):
    """Standardize or normalize data. More standardize methods available.

    Args:
        data (np.ndarray): Time series data.
        predicted_column_index (int, optional): Index of column that is predicted. Defaults to 0.
        standardizer (str, optional): '01' and '-11' means scope from to for normalization.
            'robust' use RobustScaler and 'standard' use StandardScaler - mean is 0 and std is 1. Defaults to 'standardize'.

    Returns:
        np.ndarray: Standardized data.
    """

    from sklearn import preprocessing

    if data.ndim == 1:
        if standardizer == '01':
            final_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        if standardizer == '-11':
            final_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        if standardizer == 'robust':
            final_scaler = preprocessing.RobustScaler()
        if standardizer == 'standardize':
            final_scaler = preprocessing.StandardScaler()

        data_for_predictions = final_scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)

    if data.ndim == 2:
        if not (data.dtype == 'float64' or data.dtype == 'float32'):
            data = data.astype('float64')
        data_for_predictions = data.copy()
        for i in range(len(data)):

            if standardizer == '01':
                scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            if standardizer == '-11':
                scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            if standardizer == 'robust':
                scaler = preprocessing.RobustScaler()
            if standardizer == 'standardize':
                scaler = preprocessing.StandardScaler()

            if i == predicted_column_index:
                final_scaler = scaler

            scaled = scaler.fit_transform(data[i, :].reshape(-1, 1))
            data_for_predictions[i, :] = scaled.reshape(-1)

    return data_for_predictions, final_scaler


def split(data, predicts=7, predicted_column_index=0):
    """Divide data set on train and test set.

    Args:
        data (pd.DataFrame, np.ndarray): Time series data.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        predicted_column_index (int, optional): Index of column that is predicted. Defaults to 0.
    
    Returns:
        np.array, np.array: Train set and test set.

    Examples:

        >>> data = np.array([1, 2, 3, 4])
        >>> train, test = (split(data, predicts=2))
        >>> print(train, test)
        [2, 3]
        [4, 5]
    """

    if isinstance(data, pd.DataFrame):
        data = data.values

    data = np.array(data)
    data_shape = data.shape

    if len(data_shape) == 1:
        train = data[:(len(data) - predicts)]
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


def make_sequences(data, n_steps_in, n_steps_out=1, constant=None, predicted_column_index=0, other_columns_lenght=None):
    """Function that create inputs and outputs to models.
    From [1, 2, 3, 4, 5, 6, 7, 8]

        Creates [1, 2, 3]  [4]
                [5, 6, 7]  [8]

    Args:
        data (np.array): Time series data.
        n_steps_in (int): Number of input members.
        n_steps_out (int, optional): Number of output members. For one-step models use 1. Defaults to 1.
        constant (bool, optional): If use bias (add 1 to first place to every member). Defaults to None.
        predicted_column_index (int, optional): [description]. Defaults to 0.
        other_columns_lenght (int, optional): Length of non-predicted columns that are evaluated in inputs. If None, than same length as predicted column. Defaults to None.

    Returns:
        np.array, np.array: X and y. Inputs and outputs (that can be used for example in sklearn models).

    """

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

    if not other_columns_lenght:
        data = data[predicted_column_index]

    data_shape = data.shape

    if data_shape[0] == 1:
        data = data.reshape(-1)
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
        other_columns_array = other_columns_array[:, :, -other_columns_lenght:]

        for i in range(len(X_only)):
            if data_shape[0] != 1:
                other_columns_list = list(other_columns_array[:, i, :].reshape(-1))

            X_list.append(other_columns_list + list(X_only[i]))

    X = np.array(X_list)

    if constant:
        X = np.insert(X, 0, constant, axis=1)

    return np.array(X), np.array(y)


def make_x_input(data, n_steps_in, predicted_column_index=0, other_columns_lenght=None, constant=None):
    """Make input into model.

    Args:
        data (np.array): Time series data.
        n_steps_in (int): Lags going into model.
        predicted_column_index (int, optional): Index of predicted column. Defaults to 0.
        other_columns_lenght (int, optional): Lags of other columns. If none, than same length as predicted column Defaults to None.
        constant (bool, optional): If use bias (add 1 to first place to every member). Defaults to None.

    Returns:
        np.array: y input in model.

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
