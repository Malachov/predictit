""" Test module. Auto pytest that can be started in IDE or with

    >>> python -m pytest

in terminal in tests folder.
"""
#%%

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import inspect
import os
import warnings
import matplotlib

import mydatapreprocessing as mdp
import mylogging

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use('agg')

sys.path.insert(0, Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[0].as_posix())
sys.path.insert(0, Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[1].as_posix())

from visual import visual_test

import predictit
predictit.misc._IS_TESTED = 1

Config = predictit.configuration.Config
Config.plotit = 0
config_unchanged = Config.freeze()

mylogging._COLORIZE = 0

# ANCHOR Tests config
Config.update({
    'return_type': 'best',
    'predicted_column': 0,
    'debug': 1,
    'printit': 0,
    'plotit': 0,
    'show_plot': 0,
    'data': None,
    'datalength': 120,
    'default_n_řžsteps_in': 3,
    'analyzeit': 0,
    'optimization': 0,
    'optimizeit': 0,
    'used_models': [
        "AR (Autoregression)",
        "Conjugate gradient",
        "Sklearn regression",
    ]
})

config_for_tests = Config.freeze()

np.random.seed(2)


def test_1():
    Config.update({
        'data': [pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c']), pd.DataFrame(np.random.randn(80, 3), columns=['e', 'b', 'c'])],
        'predicted_column': "b",
        'remove_nans_threshold': 0.85,
        'remove_nans_or_replace': 'neighbor',
        'add_fft_columns': 0,
        'do_data_extension': False,
        'embedding': 'label',
    })

    Config.plotit = 1
    Config.printit = 1
    Config.plot_type = 'plotly'
    Config.show_plot = 0
    Config.confidence_interval = 0

    Config.multiprocessing = 0
    result = predictit.main.predict(predicts=3, return_type=None)
    assert not np.isnan(np.min(list(result.values())[0]['Test errors']))
    return result


def test_readmes():

    ######
    ### Simple example of using predictit as a python library and function arguments
    ######
    Config.update(config_unchanged)

    predictions_1 = predictit.main.predict(data=np.random.randn(100, 2), predicted_column=1, predicts=3, return_type='best')

    mydata = pd.DataFrame(np.random.randn(100, 2), columns=['a', 'b'])
    predictions_1_positional = predictit.main.predict(mydata, 'b')

    ######
    ### Simple example of using as a python library and editing Config
    ######
    Config.update(config_for_tests)

    # You can edit Config in two ways
    Config.data = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'  # You can use local path on pc as well... "/home/name/mycsv.csv" !
    Config.predicted_column = 'Temp'  # You can use index as well
    Config.datetime_column = 'Date'  # Will be used in result
    Config.freq = "D"  # One day - one value resampling
    Config.resample_function = "mean"  # If more values in one day - use mean (more sources)
    Config.return_type = 'detailed_dictionary'
    Config.debug = 0  # Ignore warnings

    # Or
    Config.update({
        'datalength': 300,  # Used datalength
        'predicts': 9,  # Number of predicted values
        'default_n_steps_in': 8  # Value of recursive inputs in model (do not use too high - slower and worse predictions)
    })

    predictions_2 = predictit.main.predict()

    ######
    ### Example of compare_models function
    ######
    Config.update(config_for_tests)

    my_data_array = np.random.randn(200, 2)  # Define your data here

    # You can compare it on same data in various parts or on different data (check configuration on how to insert dictionary with data names)
    Config.update({
        'data_all': {'First part': (my_data_array[:100], 0), 'Second part': (my_data_array[100:], 1)}
    })

    compared_models = predictit.main.compare_models()

    ######
    ### Example of predict_multiple function
    ######
    Config.update(config_for_tests)

    Config.data = np.random.randn(120, 3)
    Config.predicted_columns = ['*']  # Define list of columns or '*' for predicting all of the numeric columns
    Config.used_models = ['Conjugate gradient', 'Decision tree regression']  # Use just few models to be faster

    multiple_columns_prediction = predictit.main.predict_multiple_columns()

    ######
    ### Example of Config variable optimization ###
    ######
    Config.update(config_for_tests)

    Config.update({
        'data': "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        'predicted_column': 'Temp',
        'return_type': 'all_dataframe',
        'datalength': 120,
        'optimization': 1,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [4, 6, 8],
        'plot_all_optimized_models': 0,
        'print_table': 1,  # Print detailed table
        'used_models': ['AR (Autoregression)', 'Sklearn regression']

    })

    predictions_optimized_config = predictit.main.predict()

    ######
    ### Data preprocessing, plotting and other Functions
    ######

    from predictit.analyze import analyze_column
    from mydatapreprocessing.preprocessing import load_data, data_consolidation, preprocess_data
    from predictit.plots import plot

    data = "https://blockchain.info/unconfirmed-transactions?format=json"

    # Load data from file or URL
    data_loaded = load_data(data, request_datatype_suffix=".json", predicted_table='txs', data_orientation="index")

    # Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
    # only numeric data and resample ifg configured. It return array, dataframe
    data_consolidated = data_consolidation(
        data_loaded, predicted_column="weight", remove_nans_threshold=0.9, remove_nans_or_replace='interpolate')

    # Predicted column is on index 0 after consolidation)
    analyze_column(data_consolidated.iloc[:, 0])

    # Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
    # transformation, so unpack it with _
    data_preprocessed, _, _ = preprocess_data(data_consolidated, remove_outliers=True, smoothit=False,
                                              correlation_threshold=False, data_transform=False, standardizeit='standardize')

    # Plot inserted data (show false just because tests)
    plot(data_preprocessed, show=0)


    ######
    ### Example of using model apart main function
    ######

    data = mdp.generatedata.gen_sin(1000)
    test = data[-7:]
    data = data[: -7]
    data = mdp.preprocessing.data_consolidation(data)
    (X, y), x_input, _ = mdp.inputs.create_inputs(data.values, 'batch', input_type_params={'n_steps_in': 6})  # First tuple, because some models use raw data, e.g. [1, 2, 3...]

    trained_model = predictit.models.sklearn_regression.train((X, y), regressor='bayesianridge')
    predictions_one_model = predictit.models.sklearn_regression.predict(x_input, trained_model, predicts=7)

    predictions_one_model_error = predictit.evaluate_predictions.compare_predicted_to_test(predictions_one_model, test, error_criterion='mape')  # , plot=1

    ######
    ### Example of using library as a pro with deeper editting Config
    ######
    Config.update(config_for_tests)

    Config.update({
        'data': r'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv',  # Full CSV path with suffix
        'predicted_column': 'Temp',  # Column name that we want to predict
        'datalength': 200,
        'predicts': 7,  # Number of predicted values - 7 by default
        'print_number_of_models': 6,  # Visualize 6 best models
        'repeatit': 50,  # Repeat calculation times on shifted data to evaluate error criterion
        'other_columns': 0,  # Whether use other columns or not
        'debug': 1,  # Whether print details and warnings

        # Chose models that will be computed - remove if you want to use all the models
        'used_models': [
            "AR (Autoregression)",
            "ARIMA (Autoregression integrated moving average)",
            "Autoregressive Linear neural unit",
            "Conjugate gradient",
            "Sklearn regression",
            "Bayes ridge regression one step",
            "Decision tree regression",
        ],

        # Define parameters of models

        'models_parameters': {

            "AR (Autoregression)": {'used_model': 'ar', 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
            "ARIMA (Autoregression integrated moving average)": {'used_model': 'arima', 'p': 6, 'd': 0, 'q': 0},

            "Autoregressive Linear neural unit": {'mi_multiple': 1, 'mi_linspace': (1e-5, 1e-4, 3), 'epochs': 10, 'w_predict': 0, 'minormit': 0},
            "Conjugate gradient": {'epochs': 200},

            "Bayes ridge regression": {'regressor': 'bayesianridge', 'n_iter': 300, 'alpha_1': 1.e-6, 'alpha_2': 1.e-6, 'lambda_1': 1.e-6, 'lambda_2': 1.e-6},
        }
    })

    predictions_configured = predictit.main.predict()
    first_multiple_array = multiple_columns_prediction[list(multiple_columns_prediction.keys())[0]]

    condition_1 = not np.isnan(np.min(predictions_1))
    condition_1_a = not np.isnan(np.min(predictions_1_positional))
    condition_2 = not np.isnan(np.min(predictions_2['best']))
    condition_3 = compared_models
    condition_4 = not np.isnan(np.nanmax(first_multiple_array))
    condition_5 = not predictions_optimized_config.dropna().empty
    condition_6 = 0 <= predictions_one_model_error < 1000
    condition_7 = not np.isnan(np.min(predictions_configured))

    assert (condition_1 and condition_1_a and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7)


def test_main_from_config():

    Config.update(config_for_tests)
    Config.update({
        'data': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv',
        'predicted_column': 'Temp',
        'datetime_column': 'Date',
        'freq': 'D',
        'return_type': 'all_dataframe',
        'max_imported_length': 300,
        'plotit': 1,
        'plot_type': 'matplotlib',
        'show_plot': 0,
        'trace_processes_memory': False,
        'print_number_of_models': 10,
        'add_fft_columns': 1,
        'fft_window': 16,
        'do_data_extension': True,
        'add_differences': True,
        'add_second_differences': True,
        'add_multiplications': True,
        'add_rolling_means': True,
        'add_rolling_stds': True,
        'rolling_data_window': 10,
        'add_mean_distances': True,
        'analyzeit': 1,
        'last_row': 0,
        'correlation_threshold': 0.2,
        'optimizeit': 0,
        'standardizeit': '01',
        'multiprocessing': 'process',
        'smoothit': (19, 2),
        'power_transformed': 1,
        'analyze_seasonal_decompose': {'period': 32, 'model': 'additive'},
        'confidence_interval': 1,

        'used_models': [
            "Bayes ridge regression",
            "Conjugate gradient",
        ]
    })

    result = predictit.main.predict()
    assert not result.dropna().empty
    return result


def test_main_optimize_and_args():

    Config.update(config_for_tests)
    Config.update({
        'data': 'test',
        'predicted_column': '',
        'predicts': 6,
        'datetime_column': '',
        'freq': 'M',
        'datalength': 100,
        'default_n_steps_in': 3,
        'data_transform': 'difference',
        'error_criterion': 'rmse',
        'remove_outliers': 1,
        'print_number_of_models': 1,
        'print_table': 2,
        'print_time_table': 1,
        'last_row': 1,
        'correlation_threshold': 0.2,
        'optimizeit': 1,
        'optimizeit_limit': 0.1,
        'optimizeit_details': 3,
        'optimizeit_plot': 1,
        'standardizeit': 0,
        'multiprocessing': 'pool',
        'trace_processes_memory': True,
        'used_models': ["Bayes ridge regression"],
        'models_parameters': {"Bayes ridge regression": {"regressor": 'bayesianridge', "n_iter": 300, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6}},
        'fragments': 4,
        'iterations': 2,
        'models_parameters_limits': {"Bayes ridge regression": {"alpha_1": [0.1e-6, 3e-6], "regressor": ['bayesianridge', 'lasso']}},
    })

    result = predictit.main.predict(data='test', predicted_column=[], repeatit=20)
    assert not np.isnan(result.min())
    return result


def test_config_optimization():

    Config.update(config_for_tests)
    df = pd.DataFrame([range(200), range(1000, 1200)]).T
    df['time'] = pd.date_range('2018-01-01', periods=len(df), freq='H')

    Config.update({
        'data': df,
        'datetime_column': 'time',
        'freq': '',
        'predicted_column': 0,
        'datalength': 100,
        'other_columns': 1,
        'default_other_columns_length': 5,
        'data_transform': None,
        'repeatit': 1,
        'remove_outliers': 0,
        'remove_nans_threshold': 0.8,
        'remove_nans_or_replace': 'mean',
        'optimization': 1,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [3, 6, 9],
        'plot_all_optimized_models': 1,
        'last_row': 0,
        'correlation_threshold': 0.4,
        'standardizeit': 'standardize',
        'predicts': 7,
        'smoothit': 0,
    })

    result = predictit.main.predict()
    assert not np.isnan(np.min(result))
    return result


def test_most_models():

    Config.update(config_for_tests)
    df = pd.DataFrame([range(100), range(1000, 1100)]).T

    Config.update({
        'data': df,
        'predicts': 7,
        'default_n_steps_in': 3,
        'error_criterion': 'mape',

        'used_models': [
            'ARMA', 'ARIMA (Autoregression integrated moving average)', 'autoreg', 'SARIMAX (Seasonal ARIMA)',
            'Autoregressive Linear neural unit', 'Autoregressive Linear neural unit normalized', 'Linear neural unit with weights predict',
            'Sklearn regression', 'Bayes ridge regression', 'Passive aggressive regression', 'Gradient boosting',
            'KNeighbors regression', 'Decision tree regression', 'Hubber regression',
            'Bagging regression', 'Stochastic gradient regression', 'Extreme learning machine', 'Gen Extreme learning machine', 'Extra trees regression', 'Random forest regression',
            'Tensorflow LSTM', 'Tensorflow MLP',
            'Compare with average',
            'Regression', 'Ridge regression'
        ]
    })

    result = predictit.main.predict()

    assert not np.isnan(np.min(result))

    return result


def test_other_models_functions():
    tf_optimizers = predictit.models.tensorflow.get_optimizers_loses_activations()
    sklearn_regressors = predictit.models.sklearn_regression.get_regressors()

    sequentions = mdp.inputs.make_sequences(np.random.randn(100, 1), 5)
    predictit.models.autoreg_LNU.train(sequentions, plot=1)

    assert tf_optimizers and sklearn_regressors


def test_presets():
    Config.update(config_for_tests)
    Config.update({
        'data': "https://blockchain.info/unconfirmed-transactions?format=json",
        'request_datatype_suffix': '.json',
        'predicted_table': 'txs',
        'data_orientation': 'index',
        'predicted_column': 'weight',
        'datalength': 100,
        'use_config_preset': 'fast',
    })

    result = predictit.main.predict()
    assert not np.isnan(np.min(result))

    Config.update({
        'use_config_preset': 'normal'
    })

    result = predictit.main.predict()
    assert not np.isnan(np.min(result))
    return result


def test_main_multiple():

    Config.update(config_for_tests)
    Config.update({
        'data': np.random.randn(120, 3),
        'predicted_columns': [0, 1],
        'error_criterion': 'mse_sklearn',
    })

    result_multiple = predictit.main.predict_multiple_columns()
    first_array = result_multiple[list(result_multiple.keys())[0]]
    assert not np.isnan(np.min(first_array))
    return result_multiple


def test_main_multiple_all_columns():

    Config.update(config_for_tests)
    Config.update({
        'use_config_preset': 'fast',
        'datetime_column': 'Date',
        'predicted_columns': '*',
        'freqs': ['D', 'M'],
        'data': pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv").iloc[:120, :3],
        'remove_nans_threshold': 0.9,
        'remove_nans_or_replace': 2,
        'optimization': 0,
    })

    result_multiple = predictit.main.predict_multiple_columns()
    first_array = result_multiple[list(result_multiple.keys())[0]]
    assert not np.isnan(np.min(first_array))
    return result_multiple


def test_compare_models():
    Config.update(config_for_tests)
    data_all = None

    result = predictit.main.compare_models(data_all=data_all)

    assert result


def test_compare_models_list():
    Config.update(config_for_tests)
    dummy_data = np.random.randn(300)
    data_all = [dummy_data[:100], dummy_data[100: 200], dummy_data[200:]]

    result = predictit.main.compare_models(data_all=data_all)

    assert result


def test_compare_models_with_optimization():
    Config.update(config_for_tests)
    Config.update({
        'data_all': None,  # Means default sin, random, sign
        'optimization': 1,
        'optimization_variable': 'data_transform',
        'optimization_values': [0, 'difference'],
    })

    result = predictit.main.compare_models()

    assert result


def test_visual():
    visual_test(print_analyze=1, print_preprocessing=1, print_data_flow=1, print_postprocessing=1)


# def test_GUI():
#     predictit.gui_start.run_gui()


# For deeper debug, uncomment problematic test
if __name__ == "__main__":
    # result = test_1()
    # result_readmes = test_readmes()
    # result1 = test_main_from_config()
    # result_2 = test_main_optimize_and_args()
    # result_3 = test_config_optimization()
    # result_4 = test_presets()
    # result_multiple = test_main_multiple()
    # test_main_multiple_all_columns = test_main_multiple_all_columns()
    # test_compare_models = test_compare_models()
    # test_compare_models_list = test_compare_models_list()
    # test_compare_models_with_optimization = test_compare_models_with_optimization()
    # test_GUI()
    # test_preprocessing()

    ## Custom use case test...

    pass
