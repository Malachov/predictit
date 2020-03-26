"""Module for data preprocessing. Contain functions for removing outliers, NaN columns or 
function for making inputs and outputs from time series if you want to see how functions work -
working examples with printed results are in tests - visual.py.

"""

import pandas as pd
import numpy as np
from scipy.stats import yeojohnson


def remove_nan_columns(data):
    """Remove columns that are not a numbers.

    Args:
        data (pd.DataFrame): Time series data.

    Returns:
        pd.DataFrame: Cleaned time series data.
    """

    data = data.select_dtypes(include='number')

    return data


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
        corr = np.corrcoef(data)
        range_array = np.array(range(len(corr)))
        names_to_del = range_array[corr[predicted_column_index] <= abs(threshold)]
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

    if isinstance(data, np.ndarray):

        data_shape = data.shape

        if len(data_shape) == 1:
            data = data[abs(data - data.mean()) < threshold * data.std()]
            return data

        else:
            data_mean = data[predicted_column_index].mean()
            data_std = data[predicted_column_index].std()

            range_array = np.array(range(data_shape[1]))
            names_to_del = range_array[abs(data[predicted_column_index] - data_mean) > threshold * data_std]
            data = np.delete(data, names_to_del, axis=1)

        return data


def do_difference(data):
    """Transform data into neighbor difference. For example from [1, 2, 4] into [1, 2].

    Args:
        dataset (np.ndarray): Numpy on or multi dimensional array.

    Returns:
        np.ndarray: Differenced data.

    Examples:

        >>> data = np.array([1, 3, 5, 2])
        >>> print(remove_outliers(data))
        [2, 2, -3]
    """

    diff = np.diff(data)

    return diff


def inverse_difference(differenced_predictions, last_undiff_value):
    """Transform do_difference transform back.

    Args:
        differenced_predictions (np.ndarray): One dimensional!! differenced data from do_difference function.
        last_undiff_value (float): First value to computer the rest.

    Returns:
        np.ndarray: Normal data, not the additive series.

    Examples:

        >>> data = np.array([1, 1, 1, 1])
        >>> print(inverse_difference(data, 1))
        [2, 3, 4, 5]
    """

    assert differenced_predictions.ndim == 1, 'Data input must be one-dimensional.'

    differenced_predictions[0] = differenced_predictions[0] + last_undiff_value
    undiff = np.cumsum(differenced_predictions)

    return undiff


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


def fitted_power_transform(data, fitted_stdev, mean=None, fragments=10, iterations=5):
    """Function mostly for data postprocessing. Function transforms data, so it will have 
    similiar standar deviation, similiar mean if specified. It use Box-Cox power transform in SciPy lib.

    Args:
        data (np.array): Array of data that should be transformed.
        fitted_stdev (float): Standard deviation that we want to have.
        mean (float, optional): Mean of transformed data. Defaults to None.
        fragments (int, optional): How many lambdas will be used in one iteration. Defaults to 9.
        iterations (int, optional): How many iterations will be used to find best transform. Defaults to 4.

    Returns:
        np.array: Transformed data with demanded standard deviation and mean.
    """

    lmbda_low = 0
    lmbda_high = 3
    lmbd_arr = np.linspace(lmbda_low, lmbda_high, fragments)
    lbmda_best_stdv_error = 1000000

    for i in range(iterations):
        for j in range(len(lmbd_arr)):

            power_transformed = yeojohnson(data, lmbda=lmbd_arr[j])
            transformed_stdev = np.std(power_transformed)
            if abs(transformed_stdev - fitted_stdev) < lbmda_best_stdv_error:
                lbmda_best_stdv_error = abs(transformed_stdev - fitted_stdev)
                lmbda_best_id = j

        if lmbda_best_id > 0:
            lmbda_low = lmbd_arr[lmbda_best_id - 1]
        if lmbda_best_id < len(lmbd_arr) - 1:
            lmbda_high = lmbd_arr[lmbda_best_id + 1]
        lmbd_arr = np.linspace(lmbda_low, lmbda_high, fragments)

    transformed_results = yeojohnson(data, lmbda=lmbd_arr[j])

    if mean is not None:
        mean_difference = np.mean(transformed_results) - mean
        transformed_results = transformed_results - mean_difference

    return transformed_results


def smooth():
    # TODO
    pass
