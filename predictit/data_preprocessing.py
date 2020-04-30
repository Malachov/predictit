"""Module for data preprocessing. Contain functions for removing outliers,
NaN columns or data smoothing. If you want to see how functions work -
working examples with printed results are in tests - visual.py.

Default output data shape is (n_samples, n_features)

All processing funtions after data_consolidation use numpy.ndarray with ndim == 2,
so reshape(1, -1) if necessary.

"""

import pandas as pd
import numpy as np
import scipy
import warnings
from sklearn import preprocessing

from predictit.config import config
from predictit.misc import user_warning, colorize



def data_consolidation(data):
    """Input data in various formats and shapes and return data in defined shape,
    that other functions rely on.

    Args:
        data (np.ndarray, pd.DataFrame): Input data in well standardized format.

    Raises:
        KeyError, TypeError: If wrong configuration in config.py.
            E.g. if predicted column name not found in dataframe.


    Returns:
        np.ndarray, pd.DataFrame, str: Data in standardized form.

        Data array for prediction - predicted column on index 1,
        and column for ploting as pandas dataframe.

    """
    ### Pandas dataframe and series ###

    if isinstance(data, pd.Series):

        data = pd.Dataframe(data)

    if isinstance(data, pd.DataFrame):
        data_for_predictions_df = data

        if isinstance(config['predicted_column'], str):

            predicted_column_name = config['predicted_column']

            try:
                predicted_column_index = data_for_predictions_df.columns.get_loc(predicted_column_name)
            except Exception:
                raise KeyError(colorize(f"Predicted column name - '{config['predicted_column']}' not found in data. Change in config - 'predicted_column'"))

        else:
            predicted_column_index = config['predicted_column']
            predicted_column_name = data_for_predictions_df.columns[predicted_column_index]

        try:
            int(data_for_predictions_df[predicted_column_name].iloc[0])
        except Exception:
            user_warning("Predicted column is not number datatype. Setup correct 'predicted_column' in config.py")

        if isinstance(data.index, (pd.core.indexes.datetimes.DatetimeIndex, pd._libs.tslibs.timestamps.Timestamp)):
            config['datetime_index'] = 'index'

        if config['datetime_index'] is not None:

            if config['datetime_index'] == 'index':
                pass

            elif isinstance(config['datetime_index'], str):
                try:
                    data_for_predictions_df.set_index(config['datetime_index'], drop=True, inplace=True)
                except Exception:
                    raise KeyError(colorize(f"Datetime name / index from config - '{config['datetime_index']}' not found in data. Change in config - 'datetime_index'"))

            else:
                data_for_predictions_df.set_index(data_for_predictions_df.columns[config['datetime_index']], drop=True, inplace=True)

            data_for_predictions_df.index = pd.to_datetime(data_for_predictions_df.index)

            if config['freq']:
                data_for_predictions_df.sort_index(inplace=True)
                data_for_predictions_df = data_for_predictions_df.resample(config['freq']).sum()
                data_for_predictions_df = data_for_predictions_df.asfreq(config['freq'], fill_value=0)

            else:

                data_for_predictions_df.index.freq = pd.infer_freq(data_for_predictions_df.index)

                if data_for_predictions_df.index.freq is None:
                    user_warning("Datetime index was provided from config, but frequency guess failed. Specify 'freq' in config to resample and have equal sampling.")
                    data_for_predictions_df.reset_index(inplace=True)

        # Make predicted column index 0
            data_for_predictions_df.insert(0, predicted_column_name, data_for_predictions_df.pop(predicted_column_name))

        data_for_predictions_df = data_for_predictions_df.iloc[-config['datalength']:, :]

        if config['other_columns']:

            if config['remove_nans'] == 'any_columns':
                data_for_predictions_df.dropna(how='any', inplace=True, axis=1)

            data_for_predictions_df = data_for_predictions_df.select_dtypes(include='number')
            data_for_predictions = data_for_predictions_df.values.reshape(-1, 1)

        else:
            data_for_predictions = data_for_predictions_df[predicted_column_name].values.reshape(-1, 1)

    ### np.ndarray ###


    elif isinstance(data, np.ndarray):
        data_for_predictions = data

        if not isinstance(config['predicted_column'], int):
            raise TypeError(colorize("'predicted_column' in config is a string and data in numpy array format. Numpy does not allow string assignment"))

        predicted_column_name = 'Predicted column'

        if data_for_predictions.ndim == 1:
            data_for_predictions = data_for_predictions.reshape(-1, 1)

        if data_for_predictions.shape[0] < data_for_predictions.shape[1]:
            data_for_predictions = data_for_predictions.T

        data_for_predictions = data_for_predictions[-config['datalength']:, :]

        # Make predicted column on index 0
        if config['other_columns'] and config['predicted_column'] != 0 and data_for_predictions.shape[1] != 1:
            data_for_predictions[:, [0, config['predicted_column']]] = data_for_predictions[:, [config['predicted_column'], 0]]

        if not config['other_columns']:
            data_for_predictions = data_for_predictions[:, 0].reshape(-1, 1)

        data_for_predictions_df = pd.DataFrame(data_for_predictions)
        data_for_predictions_df.rename(columns={0: predicted_column_name})

    else:
        raise TypeError(colorize("""Input data must be in pd.dataframe, pd.series or numpy array. If you use csv or sql data source, its converted automatically,
                                 but setup csv_full_path. Check config comments fore  more informations..."""))

    data_for_predictions = data_for_predictions.astype(config['dtype'], copy=False)

    return data_for_predictions, data_for_predictions_df, predicted_column_name


def keep_corelated_data(data, predicted_column_index=0, threshold=0.5):
    """Remove columns that are not corelated enough to predicted columns.

    Args:
        data (np.array): Time series data.
        predicted_column_index (int, optional): Column that is predicted. Defaults to 0.
        threshold (float, optional): After correlation matrix is evaluated, all columns that are correlated less than threshold are deleted. Defaults to 0.2.

    Returns:
        np.array: Data with no columns that are not corelated with predicted column.
    """

    # If some row have no variance - RuntimeWarning warning in correlation matrix computing and then in comparing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        corr = np.corrcoef(data)
        corr = np.nan_to_num(corr, 0)

        range_array = np.array(range(corr.shape[0]))
        columns_to_del = range_array[abs(corr[predicted_column_index]) <= threshold]

        data = np.delete(data, columns_to_del, axis=0)

        return data

    # if isinstance(data, pd.DataFrame):
    #     corr = data.corr()
    #     columns_to_del = list(corr[corr[corr.columns[predicted_column_index]] <= abs(0.6)].index)
    #     data.drop(columns=columns_to_del, inplace=True)


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

    data_mean = data[:, predicted_column_index].mean()
    data_std = data[:, predicted_column_index].std()

    range_array = np.array(range(data.shape[0]))
    names_to_del = range_array[abs(data[:, predicted_column_index] - data_mean) > threshold * data_std]
    data = np.delete(data, names_to_del, axis=0)

    return data


def do_difference(data):
    """Transform data into neighbor difference. For example from [1, 2, 4] into [1, 2].

    Args:
        dataset (ndarray): Numpy on or multi dimensional array.

    Returns:
        ndarray: Differenced data.

    Examples:

        >>> data = np.array([1, 3, 5, 2])
        >>> print(remove_outliers(data))
        [2, 2, -3]
    """

    diff = np.diff(data.T).T

    return diff


def inverse_difference(differenced_predictions, last_undiff_value):
    """Transform do_difference transform back.

    Args:
        differenced_predictions (ndarray): One dimensional!! differenced data from do_difference function.
        last_undiff_value (float): First value to computer the rest.

    Returns:
        ndarray: Normal data, not the additive series.

    Examples:

        >>> data = np.array([1, 1, 1, 1])
        >>> print(inverse_difference(data, 1))
        [2, 3, 4, 5]
    """

    assert differenced_predictions.ndim == 1, 'Data input must be one-dimensional.'

    undiff = np.insert(differenced_predictions, 0, last_undiff_value).cumsum()[1:]

    return undiff


def standardize(data, used_scaler='standardize', predicted_column=0):
    """Standardize or normalize data. More standardize methods available.

    Args:
        data (ndarray): Time series data.
        used_scaler (str, optional): '01' and '-11' means scope from to for normalization.
            'robust' use RobustScaler and 'standard' use StandardScaler - mean is 0 and std is 1. Defaults to 'standardize'.

    Returns:
        ndarray: Standardized data.
    """

    scaler = {
        '01': preprocessing.MinMaxScaler(feature_range=(0, 1)),
        '-11': preprocessing.MinMaxScaler(feature_range=(-1, 1)),
        'robust': preprocessing.RobustScaler(),
        'standardize': preprocessing.StandardScaler()
    }[used_scaler]

    normalized = scaler.fit_transform(data)

    final_scaler = scaler.fit(data[:, predicted_column].reshape(-1, 1))

    return normalized, final_scaler


def split(data, predicts=7, predicted_column_index=0):
    """Divide data set on train and test set.

    Args:
        data (pandas.DataFrame, ndarray): Time series data.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        predicted_column_index (int, optional): Index of column that is predicted. Defaults to 0.

    Returns:
        ndarray, ndarray: Train set and test set.

    Examples:

        >>> data = np.array([1, 2, 3, 4])
        >>> train, test = (split(data, predicts=2))
        >>> print(train, test)
        [2, 3]
        [4, 5]
    """
    if isinstance(data, pd.DataFrame):
        train = data.iloc[:-predicts, :]
        test = data.iloc[-predicts:, predicted_column_index]
    else:
        train = data[:-predicts, :]
        test = data[-predicts:, predicted_column_index]

    return train, test


def smooth(data, window=101, polynom_order=2):
    """Smooth data (reduce noise) with Savitzky-Golay filter. For more info on filter check scipy docs.

    Args:
        data (ndarray): Input data.
        window (tuple, optional): Length of sliding window. Must be odd.
        polynom_order - Must be smaller than window.

    Returns:
        ndarray: Cleaned data with less noise.
    """
    for i in range(data.shape[1]):
        data[:, i] = scipy.signal.savgol_filter(data[:, i], window, polynom_order)

    return data


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

            power_transformed = scipy.stats.yeojohnson(data, lmbda=lmbd_arr[j])
            transformed_stdev = np.std(power_transformed)
            if abs(transformed_stdev - fitted_stdev) < lbmda_best_stdv_error:
                lbmda_best_stdv_error = abs(transformed_stdev - fitted_stdev)
                lmbda_best_id = j

        if lmbda_best_id > 0:
            lmbda_low = lmbd_arr[lmbda_best_id - 1]
        if lmbda_best_id < len(lmbd_arr) - 1:
            lmbda_high = lmbd_arr[lmbda_best_id + 1]
        lmbd_arr = np.linspace(lmbda_low, lmbda_high, fragments)

    transformed_results = scipy.stats.yeojohnson(data, lmbda=lmbd_arr[j])

    if mean is not None:
        mean_difference = np.mean(transformed_results) - mean
        transformed_results = transformed_results - mean_difference

    return transformed_results
