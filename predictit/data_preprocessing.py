"""Module for data preprocessing. Contain functions for removing outliers,
NaN columns or data smoothing. If you want to see how functions work -
working examples with printed results are in tests - visual.py.

Default output data shape is (n_samples, n_features)!

There are many small functions, but there they are called automatically with main preprocess functions.
    - load_data
    - data_consolidation
    - preprocess_data
    - preprocess_data_inverse

In data consolidation, predicted column is moved on index 0!

All processing funtions after data_consolidation use numpy.ndarray with ndim == 2,
so reshape(1, -1) if necessary...

"""

import pandas as pd
import numpy as np
import warnings
from sklearn import preprocessing
from pathlib import Path

import predictit
from predictit.misc import user_warning, user_message


def load_data(config):

    if config.data == 'test':
        data = predictit.test_data.generate_test_data.gen_random()
        print(user_message(("Test data was used. Setup config.py 'data'. Check official readme or do"
                            " >>> predictit.configuration.print_config() to see all possible options with comments. "
                            "Data can be also inserted as function parameters, with editing config.py or with CLI.")))

        return data

    ############# Load SQL data #############
    elif config.data == 'sql':
        try:
            data = predictit.database.database_load(server=config.server, database=config.database, freq=config.freq,
                                                    data_limit=config.max_imported_length)

        except Exception:
            raise RuntimeError(user_message("ERROR - Data load from SQL server failed - "
                                            "Setup server, database and predicted column name in config"))

        return data

    # If data are url with csv, pathlib break the url...
    data_path_original = config.data

    if isinstance(config.data, str):
        config.data = Path(config.data)

    data_type_suffix = config.data.suffix

    if not data_type_suffix:
        raise TypeError(user_message("Data has no suffix (e.g. .csv). Setup correct path format or inuput "
                                     "data for example in dataframe or numpy array",
                                     caption="Data load error"))

    ############# Load CSV data #############
    if data_type_suffix in ('.csv', '.CSV'):
        try:
            if not config.data.is_file():
                not_found = True
            data = pd.read_csv(config.data.as_posix(), sep=config.csv_style['separator'],
                               decimal=config.csv_style['decimal']).iloc[-config.max_imported_length:, :]

        except Exception:

            # Maybe it is URL with csv, then only original path works
            try:
                data = pd.read_csv(data_path_original, sep=config.csv_style['separator'],
                                   decimal=config.csv_style['decimal']).iloc[-config.max_imported_length:, :]

            except Exception:

                # If failed, maybe that's test data in test_data folder
                try:
                    test_data_path = 'test_data' / config.data

                    if not_found and not config.data.is_file():
                        not_found = True

                    data = pd.read_csv(test_data_path.as_posix(), sep=config.csv_style['separator'],
                                       decimal=config.csv_style['decimal']).iloc[-config.max_imported_length:, :]

                except Exception as err:
                    if not_found:
                        raise FileNotFoundError(user_message(
                            "File not found on configured path. If you are using relative path, file must have be in CWD "
                            "(current working directory) or must be inserted in system paths (sys.path.insert(...)). If url, check if page is available."
                            f"\n\n Detailed error: \n\n {err}",
                            caption="File not found error"))
                    else:
                        raise(RuntimeError(user_message("Data load error. File found on path, but not loaded. Check if you use "
                                                        "corrent locales - correct value and decimal separators in config (different in US and EU...)",
                                                        caption="Data load failed")))
    else:
        raise TypeError(user_message(f"Your file format {data_type_suffix} not implemented yet. You can use csv, excel, parquet or txt.", "Wrong (not implemented) format"))

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

    if isinstance(data, np.ndarray):

        if not isinstance(config.predicted_column, int):
            raise TypeError(user_message("'predicted_column' in config is a string and data in numpy array format. Numpy does not allow "
                                         "string assignment", caption="Numpy string assignment not allowed"))

        data = pd.DataFrame(data)
        data.rename(columns={data.columns[config.predicted_column]: 'Predicted column'}, inplace=True)

    if isinstance(data, pd.Series):
        data = pd.Dataframe(data)

    if isinstance(data, pd.DataFrame):

        if data.shape[0] < data.shape[1]:
            print(user_message("Input data must be in shape (n_samples, n_features) that means (rows, columns) Your shape is "
                               f" {data.shape}. It's unusual to have more features than samples. Probably wrong shape.",
                               caption="Data transposed warning!!!"))
            data = data.T

        data_for_predictions_df = data.copy()

        if isinstance(config.predicted_column, str):

            predicted_column_name = config.predicted_column

            if predicted_column_name not in data_for_predictions_df.columns:
                raise KeyError(user_message(
                    f"Predicted column name - '{config.predicted_column}' not found in data. Change in config"
                    f" - 'predicted_column'. Available columns: {list(data_for_predictions_df.columns)}"))

        else:
            predicted_column_name = data_for_predictions_df.columns[config.predicted_column]

        # Make predicted column index 0
        data_for_predictions_df.insert(
            0, predicted_column_name, data_for_predictions_df.pop(predicted_column_name))

        # TODO Some less stupid check - with any()...
        try:
            int(data_for_predictions_df[predicted_column_name].iloc[0])
        except Exception:
            raise KeyError(user_message(
                "Predicted column is not number datatype. Setup correct 'predicted_column' in config.py. "
                f"Available columns: {list(data_for_predictions_df.columns)}",
                caption="Prediction available only on number datatype column.",
                traceback=True))

        if config.datetime_index not in [None, False, '']:

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
                raise KeyError(user_message(
                    f"Datetime name / index from config - '{config.datetime_index}' not found in data or not datetime format. "
                    f"Change in config - 'datetime_index'. Available columns: {list(data_for_predictions_df.columns)}"))

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
                    user_warning("Datetime index was provided from config, but frequency guess failed. "
                                 "Specify 'freq' in config to resample and have equal sampling if you want "
                                 "to have date in plot or if you want to have equal sampling. Otherwise index will "
                                 "be reset because cannot generate date indexes of predicted values.",
                                 caption="Datetime frequency not inferred")
                    data_for_predictions_df.reset_index(inplace=True)

        data_for_predictions_df = data_for_predictions_df.iloc[-config.datalength:, :]

        data_for_predictions_df = data_for_predictions_df.select_dtypes(include='number')

        if config.other_columns and data_for_predictions_df.ndim > 1:

            if config.remove_nans:
                if config.remove_nans == 'any_columns':
                    axis = 1
                    if data_for_predictions_df[predicted_column_name].isnull().values.any():
                        user_message(
                            "Nan in predicted column and remove_nans set to columns, that would remove predicted column. "
                            "Change remove_nans to rows.", "Value error")
                        axis = 0
                elif config.remove_nans == 'any_rows':
                    axis = 0

                else:
                    raise ValueError(user_message("Only possible choice in 'remove_nans' are ['any_columns', 'any_rows']", caption="Config value error"))

                data_for_predictions_df.dropna(
                    how='any', inplace=True, axis=axis)

            data_for_predictions_df = data_for_predictions_df.select_dtypes(
                include='number')
            data_for_predictions = data_for_predictions_df.values

            if data_for_predictions.ndim == 1:
                data_for_predictions = data_for_predictions.reshape(-1, 1)

        else:
            data_for_predictions = data_for_predictions_df[predicted_column_name].values.reshape(
                -1, 1)

    else:
        raise TypeError(user_message(
            "Input data must be in pd.dataframe, pd.series, numpy array or in a path (str or pathlib) with supported formats"
            " - csv, xlsx, txt or parquet. Check config comments for more informations...", "Data format error"))

    return data_for_predictions, data_for_predictions_df, predicted_column_name



def preprocess_data(data_for_predictions, multicolumn=0, remove_outliers=False, smoothit=False,
                    correlation_threshold=False, data_transform=False, standardizeit=False):
    if remove_outliers:
        data_for_predictions = remove_the_outliers(
            data_for_predictions, threshold=remove_outliers)

    if smoothit:
        data_for_predictions = smooth(
            data_for_predictions, smoothit[0], smoothit[1])

    if correlation_threshold and multicolumn:
        data_for_predictions = keep_corelated_data(
            data_for_predictions, threshold=correlation_threshold)

    if data_transform == 'difference':
        if isinstance(data_for_predictions, np.ndarray):
            last_undiff_value = data_for_predictions[-1, 0]
        else:
            last_undiff_value = data_for_predictions.iloc[-1, 0]
        data_for_predictions = do_difference(data_for_predictions)
    else:
        last_undiff_value = None

    if standardizeit:
        data_for_predictions, final_scaler = standardize(
            data_for_predictions, used_scaler=standardizeit)
    else:
        final_scaler = None

    return data_for_predictions, last_undiff_value, final_scaler


def preprocess_data_inverse(data, standardizeit=False, final_scaler=None, data_transform=False, last_undiff_value=None):
    """Undo all data preprocessing to get real data. Not not inverse all the columns, but only predicted one.
    Only predicted column is also returned. Order is reverse than preprocessing. Output is in numpy array.

    Args:
        data (np.ndarray, pd.DataFrame): Preprocessed data
        standardizeit (bool, optional): Whether use inverse standardization and what. Choices [None, 'standardize', '-11', '01', 'robust']. Defaults to False.
        final_scaler (sklearn.preprocessing.__x__scaler, optional): Scaler used in standardization. Defaults to None.
        data_transform (bool, optional): Use data transformation. Choices [False, 'difference]. Defaults to False.
        last_undiff_value (float, optional): Last used value in difference transform. Defaults to None.

    Returns:
        np.ndarray: Inverse preprocessed data
    """

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values

    if standardizeit:
        data = final_scaler.inverse_transform(data.reshape(1, -1)).ravel()

    if data_transform == 'difference':
        data = inverse_difference(data, last_undiff_value)

    return data


### Other functions are called from main upper functions...


def keep_corelated_data(data, threshold=0.5):
    """Remove columns that are not corelated enough to predicted columns. Predicted column is supposed to be 0.

    Args:
        data (np.array, pd.DataFrame): Time series data.
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
                corr[0]) <= threshold]

            data = np.delete(data, columns_to_del, axis=1)

    elif isinstance(data, pd.DataFrame):
        corr = data.corr()
        names_to_del = list(
            corr[abs(corr[corr.columns[0]]) <= threshold].index)
        data.drop(columns=names_to_del, inplace=True)

    return data


def remove_the_outliers(data, threshold=3):
    """Remove values far from mean - probably errors. If more columns, then only rows that have outlier on
    predicted column will be deleted. Predicted column is supposed to be 0.

    Args:
        data (np.array, pd.DataFrame): Time series data. Must have ndim = 2, if univariate, reshape...
        threshold (int, optional): How many times must be standard deviation from mean to be ignored. Defaults to 3.

    Returns:
        np.array: Cleaned data.

    Examples:

        >>> data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]])
        >>> print(remove_the_outliers(data))
        [1, 3, 5, 2, 3, 4, 5, 3]
    """

    if isinstance(data, np.ndarray):
        data_mean = data[:, 0].mean()
        data_std = data[:, 0].std()

        range_array = np.array(range(data.shape[0]))
        names_to_del = range_array[abs(
            data[:, 0] - data_mean) > threshold * data_std]
        data = np.delete(data, names_to_del, axis=0)

    elif isinstance(data, pd.DataFrame):
        data_mean = data.iloc[:, 0].mean()
        data_std = data.iloc[:, 0].std()

        data = data[abs(data[data.columns[0]] - data_mean) < threshold * data_std]

    return data


def do_difference(data):
    """Transform data into neighbor difference. For example from [1, 2, 4] into [1, 2].

    Args:
        dataset (np.ndarray, pd.DataFrame): Numpy on or multi dimensional array.

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
        np.ndarray: Normal data, not the additive series.

    Examples:

        >>> data = np.array([1, 1, 1, 1])
        >>> print(inverse_difference(data, 1))
        [2, 3, 4, 5]
    """

    assert differenced_predictions.ndim == 1, 'Data input must be one-dimensional.'

    return np.insert(differenced_predictions, 0, last_undiff_value).cumsum()[1:]


def standardize(data, used_scaler='standardize'):
    """Standardize or normalize data. More standardize methods available. Predicted column is supposed to be 0.

    Args:
        data (np.ndarray): Time series data.
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

    # First normalized values are calculated, then scler just for predicted value is computed again so no full matrix is necessary for inverse
    if isinstance(data, pd.DataFrame):
        normalized = data.copy()
        normalized.iloc[:, :] = scaler.fit_transform(data.copy().values)
        final_scaler = scaler.fit(data.values[:, 0].reshape(-1, 1))

    else:
        normalized = scaler.fit_transform(data)
        final_scaler = scaler.fit(data[:, 0].reshape(-1, 1))

    return normalized, final_scaler


def split(data, predicts=7):
    """Divide data set on train and test set. Predicted column is supposed to be 0.

    Args:
        data (pandas.DataFrame, ndarray): Time series data.
        predicts (int, optional): Number of predicted values. Defaults to 7.

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
        test = data.iloc[-predicts:, 0]
    else:
        train = data[:-predicts, :]
        test = data[-predicts:, 0]

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
    import scipy.signal

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

    import scipy.stats

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
