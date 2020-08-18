"""Module for data preprocessing. Contain functions for removing outliers,
NaN columns or data smoothing. If you want to see how functions work -
working examples with printed results are in tests - visual.py.

Default output data shape is (n_samples, n_features)

All processing funtions after data_consolidation use numpy.ndarray with ndim == 2,
so reshape(1, -1) if necessary.

"""

import pandas as pd
import numpy as np
import scipy.signal
import scipy.stats
import warnings
from sklearn import preprocessing

import predictit
from predictit.misc import user_warning, colorize


def load_data(config):

    ############# Load CSV data #############
    if config.data_source == 'csv':
        if config.csv_test_data_relative_path:
            try:
                combined_path = config.this_path / 'predictit' / \
                    'test_data' / config.csv_test_data_relative_path
                config.csv_full_path = combined_path.as_posix()
            except Exception:
                raise FileNotFoundError(colorize(
                    "\n\nERROR - Test data load failed - You are using relative path. This continue from folder test_data"
                    "in predictit. If you want to use other folder, keep relative_path empty and use absolute csv_full_path.\n\n"))
        try:
            data = pd.read_csv(config.csv_full_path, sep=config.csv_style['separator'],
                               decimal=config.csv_style['decimal']).iloc[-config.max_imported_length:, :]
        except Exception:
            raise FileNotFoundError(colorize(
                "\n ERROR - Test data load failed - Setup CSV adress and column name in config \n\n"))

    ############# Load SQL data #############
    elif config.data_source == 'sql':
        try:
            data = predictit.database.database_load(server=config.server, database=config.database, freq=config.freq,
                                                    data_limit=config.max_imported_length)

        except Exception:
            raise RuntimeError(colorize("ERROR - Data load from SQL server failed - "
                                        "Setup server, database and predicted column name in config"))

    elif config.data_source == 'test':
        data = predictit.test_data.generate_test_data.gen_random()
        user_warning(("Test data was used. Setup config.py 'data_source'. Check official readme or do"
                      " >>> predictit.configuration.print_config() to see all possible options with comments. "
                      "Data can be also inserted as function parameters, with editing config.py or with CLI."))

    return data


def data_consolidation(data, config):
    """Input data in various formats and shapes and return data in defined shape,
    that other functions rely on.

    Args:
        data (np.ndarray, pd.DataFrame): Input data in well standardized format.

    Raises:
        KeyError, TypeError: If wrong configuration in configuration.py.
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
        data_for_predictions_df = data.copy()

        if isinstance(config.predicted_column, str):

            predicted_column_name = config.predicted_column

            if predicted_column_name not in data_for_predictions_df.columns:
                raise KeyError(colorize(
                    f"Predicted column name - '{config.predicted_column}' not found in data. Change in config - 'predicted_column'. Available columns: {list(data_for_predictions_df.columns)}"))

        else:
            predicted_column_name = data_for_predictions_df.columns[config.predicted_column]

        try:
            int(data_for_predictions_df[predicted_column_name].iloc[0])
        except Exception:
            raise KeyError(colorize(
                f"Predicted column is not number datatype. Setup correct 'predicted_column' in config.py. Available columns: {list(data_for_predictions_df.columns)}"))

        if config.datetime_index not in [None, '']:

            try:
                if isinstance(data_for_predictions_df.index, (pd.core.indexes.datetimes.DatetimeIndex, pd._libs.tslibs.timestamps.Timestamp)):
                    pass
                else:
                    if isinstance(config.datetime_index, str):
                        data_for_predictions_df.set_index(
                            config.datetime_index, drop=True, inplace=True)

                    else:
                        data_for_predictions_df.set_index(
                            data_for_predictions_df.columns[config.datetime_index], drop=True, inplace=True)

                    data_for_predictions_df.index = pd.to_datetime(
                        data_for_predictions_df.index)

            except Exception:
                raise KeyError(colorize(
                    f"Datetime name / index from config - '{config.datetime_index}' not found in data or not datetime format. Change in config - 'datetime_index'. Available columns: {list(data_for_predictions_df.columns)}"))

            if config.freq:
                data_for_predictions_df.sort_index(inplace=True)
                if config.resample_function == 'sum':
                    data_for_predictions_df = data_for_predictions_df.resample(
                        config.freq).sum()
                if config.resample_function == 'mean':
                    data_for_predictions_df = data_for_predictions_df.resample(
                        config.freq).mean()
                data_for_predictions_df = data_for_predictions_df.asfreq(
                    config.freq, fill_value=0)

            else:

                data_for_predictions_df.index.freq = pd.infer_freq(
                    data_for_predictions_df.index)

                if data_for_predictions_df.index.freq is None:
                    user_warning(
                        "Datetime index was provided from config, but frequency guess failed. Specify 'freq' in config to resample and have equal sampling if you want.")
                    data_for_predictions_df.reset_index(inplace=True)

        # Make predicted column index 0
        data_for_predictions_df.insert(
            0, predicted_column_name, data_for_predictions_df.pop(predicted_column_name))

        data_for_predictions_df = data_for_predictions_df.iloc[-config.datalength:, :]

        if config.other_columns and data_for_predictions_df.ndim > 1:

            if config.remove_nans:
                if config.remove_nans == 'any_columns':
                    axis = 1
                if config.remove_nans == 'any_rows':
                    axis = 0

                data_for_predictions_df.dropna(
                    how='any', inplace=True, axis=axis)

            data_for_predictions_df = data_for_predictions_df.select_dtypes(
                include='number')
            data_for_predictions = data_for_predictions_df.values

        else:
            data_for_predictions = data_for_predictions_df[predicted_column_name].values.reshape(
                -1, 1)

    ### np.ndarray ###

    elif isinstance(data, np.ndarray):
        data_for_predictions = data.copy()

        if not isinstance(config.predicted_column, int):
            raise TypeError(colorize(
                "'predicted_column' in config is a string and data in numpy array format. Numpy does not allow string assignment"))

        predicted_column_name = 'Predicted column'

        if data_for_predictions.ndim == 1:
            data_for_predictions = data_for_predictions.reshape(-1, 1)

        if data_for_predictions.shape[0] < data_for_predictions.shape[1]:
            data_for_predictions = data_for_predictions.T

        data_for_predictions = data_for_predictions[-config.datalength:, :]

        # Make predicted column on index 0
        if config.other_columns and config.predicted_column != 0 and data_for_predictions.shape[1] != 1:
            data_for_predictions[:, [0, config.predicted_column]
                                 ] = data_for_predictions[:, [config.predicted_column, 0]]

        if not config.other_columns:
            data_for_predictions = data_for_predictions[:, 0].reshape(-1, 1)

        data_for_predictions_df = pd.DataFrame(data_for_predictions)
        data_for_predictions_df.rename(
            columns={data_for_predictions_df.columns[0]: predicted_column_name}, inplace=True)

    else:
        raise TypeError(colorize("""Input data must be in pd.dataframe, pd.series or numpy array. If you use csv or sql data source, its converted automatically,
                                 but setup csv_full_path. Check config comments fore  more informations..."""))

    try:
        data_for_predictions = data_for_predictions.astype(
            config.dtype, copy=False)
    except Exception:
        user_warning(
            f"Predicted column - {config.predicted_column} is not a number. Check config")

    return data_for_predictions, data_for_predictions_df, predicted_column_name


def keep_corelated_data(data, predicted_column_index=0, threshold=0.5):
    """Remove columns that are not corelated enough to predicted columns.

    Args:
        data (np.array, pd.DataFrame): Time series data.
        predicted_column_index (int, optional): Column that is predicted. Defaults to 0.
        threshold (float, optional): After correlation matrix is evaluated, all columns that are correlated less than threshold are deleted. Defaults to 0.2.

    Returns:
        np.array, pd.DataFrame: Data with no columns that are not corelated with predicted column.
    """

    if isinstance(data, np.ndarray):
        # If some row have no variance - RuntimeWarning warning in correlation matrix computing and then in comparing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            corr = np.corrcoef(data.T)
            corr = np.nan_to_num(corr, 0)

            range_array = np.array(range(corr.shape[0]))
            columns_to_del = range_array[abs(
                corr[predicted_column_index]) <= threshold]

            data = np.delete(data, columns_to_del, axis=0)

    elif isinstance(data, pd.DataFrame):
        corr = data.corr()
        names_to_del = list(
            corr[corr[corr.columns[predicted_column_index]] <= abs(0.6)].index)
        data.drop(columns=names_to_del, inplace=True)

    return data

    # if isinstance(data, pd.DataFrame):
    #     corr = data.corr()
    #     columns_to_del = list(corr[corr[corr.columns[predicted_column_index]] <= abs(0.6)].index)
    #     data.drop(columns=columns_to_del, inplace=True)


def remove_the_outliers(data, predicted_column_index=0, threshold=3):
    """Remove values far from mean - probably errors. If more columns, then onlyu rows that have outlier on predicted column will be deleted.

    Args:
        data (np.array): Time series data. Must have ndim = 2, if univariate, reshape...
        predicted_column_index (int, optional): Column that is predicted. Defaults to 0.
        threshold (int, optional): How many times must be standard deviation from mean to be ignored. Defaults to 3.

    Returns:
        np.array: Cleaned data.

    Examples:

        >>> data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]])
        >>> print(remove_the_outliers(data))
        [1, 3, 5, 2, 3, 4, 5, 3]
    """

    if isinstance(data, np.ndarray):
        data_mean = data[:, predicted_column_index].mean()
        data_std = data[:, predicted_column_index].std()

        range_array = np.array(range(data.shape[0]))
        names_to_del = range_array[abs(
            data[:, predicted_column_index] - data_mean) > threshold * data_std]
        data = np.delete(data, names_to_del, axis=0)

    elif isinstance(data, pd.DataFrame):
        data_mean = data.iloc[:, predicted_column_index].mean()
        data_std = data.iloc[:, predicted_column_index].std()

        data = data[abs(data[data.columns[0]] - data_mean) < threshold * data_std]

    return data


def do_difference(data):
    """Transform data into neighbor difference. For example from [1, 2, 4] into [1, 2].

    Args:
        dataset (ndarray): Numpy on or multi dimensional array.

    Returns:
        ndarray: Differenced data.

    Examples:

        >>> data = np.array([1, 3, 5, 2])
        >>> print(do_difference(data))
        [2, 2, -3]
    """

    if isinstance(data, np.ndarray):
        return np.diff(data, axis=0)
    else:
        return data.diff().iloc[1:]


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

    return np.insert(differenced_predictions, 0, last_undiff_value).cumsum()[1:]


def standardize(data, used_scaler='standardize', predicted_column=0):
    """Standardize or normalize data. More standardize methods available.

    Args:
        data (ndarray): Time series data.
        used_scaler (str, optional): '01' and '-11' means scope from to for normalization.
            'robust' use RobustScaler and 'standard' use StandardScaler - mean is 0 and std is 1. Defaults to 'standardize'.

    Returns:
        ndarray: Standardized data.
    """

    if used_scaler == '01':
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    elif used_scaler == '-11':
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    elif used_scaler == 'robust':
        scaler = preprocessing.RobustScaler()
    elif used_scaler == 'standardize':
        scaler = preprocessing.StandardScaler()

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
        data[:, i] = scipy.signal.savgol_filter(
            data[:, i], window, polynom_order)

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

    for _ in range(iterations):
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


def preprocess_data(data_for_predictions, predicted_column_index=0, multicolumn=0, remove_outliers=False, smoothit=False,
                    correlation_threshold=False, data_transform=False, standardizeit=False):
    if remove_outliers:
        data_for_predictions = remove_the_outliers(
            data_for_predictions, predicted_column_index=predicted_column_index, threshold=remove_outliers)

    if smoothit:
        data_for_predictions = smooth(
            data_for_predictions, smoothit[0], smoothit[1])

    if correlation_threshold and multicolumn:
        data_for_predictions = keep_corelated_data(
            data_for_predictions, threshold=correlation_threshold)

    if data_transform == 'difference':
        last_undiff_value = data_for_predictions[-1, 0]
        for i in range(data_for_predictions.shape[1]):
            data_for_predictions[1:, i] = do_difference(
                data_for_predictions[:, i])
    else:
        last_undiff_value = None

    if standardizeit:
        data_for_predictions, final_scaler = standardize(
            data_for_predictions, used_scaler=standardizeit)
    else:
        final_scaler = None

    return data_for_predictions, last_undiff_value, final_scaler


def preprocess_data_inverse(
        data, final_scaler=None, last_undiff_value=None,
        standardizeit=False, data_transform=False):
    if standardizeit:
        data = final_scaler.inverse_transform(data.reshape(-1, 1)).ravel()

    if data_transform == 'difference':
        data = inverse_difference(data, last_undiff_value)

    return data
