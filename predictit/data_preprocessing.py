"""Module for data preprocessing. Contain functions for removing outliers, NaN columns or
function for making inputs and outputs from time series if you want to see how functions work -
working examples with printed results are in tests - visual.py.

Default output data shape is (n_features, n_sample) !!! Its different than usual convention...
"""

import pandas as pd
import numpy as np
from scipy.stats import yeojohnson


def keep_corelated_data(data, predicted_column_index=0, threshold=0.5):
    """Remove columns that are not corelated enough to predicted columns.

    Args:
        data (np.array): Time series data.
        predicted_column_index (int, optional): Column that is predicted. Defaults to 0.
        threshold (float, optional): After correlation matrix is evaluated, all columns that are correlated less than threshold are deleted. Defaults to 0.2.

    Returns:
        np.array: Data with no columns that are not corelated with predicted column.
    """

    corr = np.corrcoef(data)
    range_array = np.array(range(corr.shape[0]))
    names_to_del = range_array[abs(corr[predicted_column_index]) <= threshold]
    data = np.delete(data, names_to_del, axis=0)

    # if isinstance(data, pd.DataFrame):
    #     corr = data.corr()
    #     names_to_del = list(corr[corr[corr.columns[predicted_column_index]] <= abs(0.6)].index)
    #     data.drop(columns=names_to_del, inplace=True)

    return data


def remove_outliers(data, predicted_column_index=0, threshold=3):
    """Remove values far from mean - probably errors.

    Args:
        data (np.array): Time series data.
        predicted_column_index (int, optional): Column that is predicted. Defaults to 0.
        threshold (int, optional): How many times must be standard deviation from mean to be ignored. Defaults to 3.

    Returns:
        np.array: Cleaned data.

    Examples:

        >>> data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]])
        >>> print(remove_outliers(data))
        [1, 3, 5, 2, 3, 4, 5, 3]
    """

    # if isinstance(data, pd.DataFrame):

    #     data_mean = data.iloc[:, predicted_column_index].mean()
    #     data_std = data.iloc[:, predicted_column_index].std()

    #     data = data[abs(data[data.columns[0]] - data_mean) < threshold * data_std]


    # if data.ndim == 1:
    #     data = data[abs(data - data.mean()) < threshold * data.std()]
    #     return data

    data_mean = data[predicted_column_index].mean()
    data_std = data[predicted_column_index].std()

    range_array = np.array(range(data.shape[1]))
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

    undiff = np.insert(differenced_predictions, 0, last_undiff_value).cumsum()[1:]

    return undiff


def standardize(data, predicted_column_index=0, used_scaler='standardize'):
    """Standardize or normalize data. More standardize methods available.

    Args:
        data (np.ndarray): Time series data.
        predicted_column_index (int, optional): Index of column that is predicted. Defaults to 0.
        used_scaler (str, optional): '01' and '-11' means scope from to for normalization.
            'robust' use RobustScaler and 'standard' use StandardScaler - mean is 0 and std is 1. Defaults to 'standardize'.

    Returns:
        np.ndarray: Standardized data.
    """

    from sklearn import preprocessing

    scaler = {
        '01': preprocessing.MinMaxScaler(feature_range=(0, 1)),
        '-11': preprocessing.MinMaxScaler(feature_range=(-1, 1)),
        'robust': preprocessing.RobustScaler(),
        'standardize': preprocessing.StandardScaler()
    }[used_scaler]

    normalized = scaler.fit_transform(data.T).T

    final_scaler = scaler.fit(data[0].reshape(-1, 1))

    return normalized, final_scaler


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

    # if data.ndim == 1:
    #     train = data[:(len(data) - predicts)]
    #     test = data[-predicts:]

    #     if predicts > (len(data)):
    #         print('To few data, train/test not returned')
    #         return ([np.nan], [np.nan] * predicts)

    if predicts > (data.shape[1]):
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




def data_consolidation(data, predicted_column, datalength, other_columns, do_remove_outliers, datetime_index='', freq='', dtype='float32'):

    ### pd.Series ###

    if isinstance(data, pd.Series):

        predicted_column_name = data.name

        data_for_predictions_df = pd.DataFrame(data[-datalength:])
        data_for_predictions = data_for_predictions_df.values.reshape(1, -1)

    ### pd.DataFrame ###

    elif isinstance(data, pd.DataFrame):
        data_for_predictions_df = data.iloc[-datalength:, ]

        if isinstance(predicted_column, str):

            predicted_column_name = predicted_column
            predicted_column_index = data_for_predictions_df.columns.get_loc(predicted_column_name)
        else:
            predicted_column_index = predicted_column
            predicted_column_name = data_for_predictions_df.columns[predicted_column_index]


        if isinstance(data.index, (pd.core.indexes.datetimes.DatetimeIndex, pd._libs.tslibs.timestamps.Timestamp)):
            datetime_index = 'index'

        if datetime_index is not None:

            if datetime_index == 'index':
                pass

            elif isinstance(datetime_index, str):
                data_for_predictions_df.set_index(datetime_index, drop=True, inplace=True)

            else:
                data_for_predictions_df.set_index(data_for_predictions_df.columns[datetime_index], drop=True, inplace=True)

            data_for_predictions_df.index = pd.to_datetime(data_for_predictions_df.index)

            if freq:
                data_for_predictions_df.sort_index(inplace=True)
                data_for_predictions_df = data_for_predictions_df.resample(freq).sum()
                data_for_predictions_df = data_for_predictions_df.asfreq(freq, fill_value=0)

            else:

                try:
                    data_for_predictions_df.index.freq = pd.infer_freq(data_for_predictions_df.index)
                except Exception:
                    pass

                if data_for_predictions_df.index.freq is None:
                    data_for_predictions_df.reset_index(inplace=True)

        # Make predicted column index 0
        data_for_predictions_df.insert(0, predicted_column_name, data_for_predictions_df.pop(predicted_column_name))

        if other_columns:

            data_for_predictions_df = data_for_predictions_df.select_dtypes(include='number')
            data_for_predictions = data_for_predictions_df.values.T

        else:
            data_for_predictions = data_for_predictions_df[predicted_column_name].to_frame().values.T

    ### np.ndarray ###

    elif isinstance(data, np.ndarray):
        data_for_predictions = data
        predicted_column_name = 'Predicted column'

        if data_for_predictions.ndim == 1:
            data_for_predictions = data_for_predictions.reshape(1, -1)

        if data_for_predictions.shape[0] > data_for_predictions.shape[1]:
            data_for_predictions = data_for_predictions.T

        data_for_predictions = data_for_predictions[:, -datalength:]

        if other_columns and predicted_column != 0 and data_for_predictions.shape[0] != 1:
            # Make predicted column on index 0
            data_for_predictions[[0, predicted_column], :] = data_for_predictions[[predicted_column, 0], :]

        if not other_columns:
            data_for_predictions = data_for_predictions[0].reshape(1, -1)

    else:
        raise TypeError("Input data must be in pd.dataframe, pd.series or numpy array.\n"
                        "If you use csv or sql data source, its converted automatically. Check config comments...")

    data_for_predictions = data_for_predictions.astype(dtype, copy=False)

    if 'data_for_predictions_df' not in locals():
        data_for_predictions_df = pd.DataFrame(data_for_predictions.T)
        data_for_predictions_df.rename(columns={0: predicted_column_name})
        # data_for_predictions_df = pd.DataFrame(data_for_predictions.reshape(-1), columns=[predicted_column_name])

    return data_for_predictions, data_for_predictions_df, predicted_column_name
