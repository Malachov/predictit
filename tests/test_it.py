""" Test module. Auto pytest that can be started in IDE or with

    >>> python -m pytest

in terminal in tests folder.
"""

#%%


import matplotlib
matplotlib.use('agg')

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import inspect
import os

sys.path.insert(0, Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[1].as_posix())
import predictit

config = predictit.configuration.config
predictit.misc._COLORIZE = 0

config_unchanged = config.freeze()

config.update({
    'return_type': 'best',
    'predicted_column': 0,
    'debug': 1,
    'printit': 0,
    'plotit': 0,
    'show_plot': 1,
    'data': None,
    'datalength': 500,
    'analyzeit': 0,
    'optimization': 0,
    'optimizeit': 0,
    'used_models': [
        "AR (Autoregression)",
        "Conjugate gradient",
        "Sklearn regression",
    ]
})

config_original = config.freeze()


def test_own_data():
    config.plotit = 1
    config.printit = 1
    config.plot_type = 'plotly'
    config.show_plot = 0
    config.other_columns = 0
    config.multiprocessing = 0
    result = predictit.main.predict(data=np.random.randn(5, 100), predicts=3, return_type='detailed_dictionary')
    assert not np.isnan(np.min(result['best']))
    return result


def test_main_from_config():

    config.update(config_original)
    config.update({
        'data_source': 'csv',
        'csv_full_path': 'https://datahub.io/core/global-temp/r/monthly.csv',
        'predicted_column': 'Mean',
        'datetime_index': 'Date',
        'freq': 'M',
        'return_type': 'all_dataframe',
        'max_imported_length': 1000,
        'plotit': 1,
        'plot_type': 'matplotlib',
        'show_plot': 0,
        'optimization': 1,
        'optimization_variable': 'data_transform',
        'optimization_values': [0, 'difference'],
        'print_number_of_models': 10,
        'analyzeit': 1,
        'last_row': 0,
        'correlation_threshold': 0.2,
        'optimizeit': 0,
        'standardizeit': '01',
        'multiprocessing': 'process',
        'smoothit': (19, 2),
        'power_transformed': 1,
        'analyze_seasonal_decompose': {'period': 32, 'model': 'additive'},

        'used_models': [
            "Bayes ridge regression",
        ]
    })

    result = predictit.main.predict()
    assert not result.dropna().empty
    return result


def test_main_optimize_and_args():

    config.update(config_original)
    config.update({
        'data_source': 'test',
        'predicted_column': '',
        'predicts': 3,
        'datetime_index': '',
        'freq': 'M',
        'datalength': 1000,
        'default_n_steps_in': 5,
        'data_transform': 'difference',
        'error_criterion': 'rmse',
        'remove_outliers': 1,
        'print_number_of_models': 1,
        'last_row': 1,
        'correlation_threshold': 0.2,
        'optimizeit': 1,
        'optimizeit_limit': 0.1,
        'optimizeit_details': 3,
        'optimizeit_plot': 0,
        'standardizeit': 0,
        'multiprocessing': 'pool',
        'used_models': ["Bayes ridge regression"],
        'models_parameters': {"Bayes ridge regression": {"regressor": 'bayesianridge', "n_iter": 300, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6}},
        'fragments': 4,
        'iterations': 2,
        'models_parameters_limits': {"Bayes ridge regression": {"alpha_1": [0.1e-6, 3e-6], "alpha_2": [0.1e-6, 3e-6]}},
    })

    result = predictit.main.predict(data_source='test', predicted_column=[], repeatit=20)
    assert not np.isnan(result.min())
    return result


def test_config_optimization():

    config.update(config_original)
    df = pd.DataFrame([range(200), range(1200, 1200)]).T
    df['time'] = pd.date_range('2018-01-01', periods=len(df), freq='H')

    config.update({
        'data': df,
        'datetime_index': 'time',
        'freq': '',
        'predicted_column': 0,
        'datalength': 300,
        'default_other_columns_length': 5,
        'data_transform': None,
        'repeatit': 1,
        'remove_outliers': 0,
        'optimization': 1,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [12, 20, 40],
        'last_row': 0,
        'correlation_threshold': 0.4,
        'standardizeit': 'standardize',
        'predicts': 7,
        'smoothit': 0,
        'default_n_steps_in': 10,

    })

    result = predictit.main.predict()
    assert not np.isnan(np.min(result))
    return result


def test_most_models():

    config.update(config_original)
    df = pd.DataFrame([range(200), range(1200, 1200)]).T

    config.update({
        'data': df,
        'predicts': 7,
        'default_n_steps_in': 10,

        'used_models': [
            'ARMA', 'ARIMA (Autoregression integrated moving average)', 'autoreg', 'SARIMAX (Seasonal ARIMA)',
            'Autoregressive Linear neural unit', 'Autoregressive Linear neural unit normalized', 'Linear neural unit with weigths predict',
            'Sklearn regression', 'Bayes ridge regression', 'Passive aggressive regression', 'Gradient boosting',
            'KNeighbors regression', 'Decision tree regression', 'Hubber regression',
            'Bagging regression', 'Stochastic gradient regression', 'Extreme learning machine', 'Gen Extreme learning machine', 'Extra trees regression', 'Random forest regression',
            'tensorflow_lstm', 'tensorflow_mlp',
            'Compare with average'
        ]
    })

    result = predictit.main.predict()
    assert not np.isnan(np.min(result))
    return result


def test_presets():
    config.update(config_original)
    config.update({
        'data_source': 'csv',
        'csv_full_path': "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        'predicted_column': 'Temp',
        'datalength': 500,
        'use_config_preset': 'fast',
    })

    result = predictit.main.predict()
    assert result[0]

    config.update({
        'use_config_preset': 'normal'
    })

    result = predictit.main.predict()
    assert not np.isnan(np.nanmax(result))
    return result


def test_main_multiple():

    config.update(config_original)
    config.update({
        'data_source': 'csv',
        'freqs': ['D', 'M'],
        'error_criterion': 'mse_sklearn',
        'csv_full_path': 'https://www.stats.govt.nz/assets/Uploads/Effects-of-COVID-19-on-trade/Effects-of-COVID-19-on-trade-1-February-1-July-2020-provisional/Download-data/Effects-of-COVID-19-on-trade-1-February-1-July-2020-provisional.csv',
        'predicted_columns': ['Cumulative', 'Value'],
    })

    result_multiple = predictit.main.predict_multiple_columns()
    first_array = result_multiple[list(result_multiple.keys())[0]]
    assert not np.isnan(np.nanmax(first_array))
    return result_multiple


def test_main_multiple_all_columns():

    config.update(config_original)
    config.update({
        'use_config_preset': 'fast',
        'data_source': 'csv',
        'datetime_index': 'Date',
        'freqs': ['D'],
        'csv_full_path': 'https://www.stats.govt.nz/assets/Uploads/Effects-of-COVID-19-on-trade/Effects-of-COVID-19-on-trade-1-February-1-July-2020-provisional/Download-data/Effects-of-COVID-19-on-trade-1-February-1-July-2020-provisional.csv',
        'predicted_columns': '*',
        'optimization': 0,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [12, 20, 40],
    })

    result_multiple = predictit.main.predict_multiple_columns()
    first_array = result_multiple[list(result_multiple.keys())[0]]
    assert not np.isnan(np.nanmax(first_array))
    return result_multiple


def test_compare_models():
    config.update(config_original)
    data_all = {'sin': [predictit.test_data.generate_test_data.gen_sin(), 0], 'Sign': [predictit.test_data.generate_test_data.gen_sign(), 0], 'Random data': [predictit.test_data.generate_test_data.gen_random(), 0]}

    result = predictit.main.compare_models(data_all=data_all)

    assert result


def test_compare_models_with_optimization():
    config.update(config_original)
    config.update({
        'data_all': {'sin': [predictit.test_data.generate_test_data.gen_sin(), 0],
                     'Sign': [predictit.test_data.generate_test_data.gen_sign(), 0],
                     'Random data': [predictit.test_data.generate_test_data.gen_random(), 0]},
        'optimization': 0,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [12, 20, 40]
    })

    result = predictit.main.compare_models()

    assert result


# def test_GUI():
#     predictit.gui_start.run_gui()


# For deeper debug, uncomment problematic test
if __name__ == "__main__":
    # result = test_own_data()
    # result1 = test_main_from_config()
    # result_2 = test_main_optimize_and_args()
    # result_3 = test_config_optimization()
    # result_4 = test_presets()
    # result_multiple = test_main_multiple()
    # test_main_multiple_all_columns()
    # test_compare_models()
    # test_compare_models_with_optimization()
    # test_GUI()

    ## Custom use case test...

    # config.update(config_unchanged)
    # config.update({
    #     'data_source': 'csv',
    #     'csv_full_path': "/home/dan/ownCloud/Github/Data engineering/test_data/pokus.csv",
    #     'predicted_column': 2,
    #     'plotit': 0,
    #     'error_criterion': 'dtw',
    #     "show_plot": 0,
    #     "save_plot": 0,
    #     'printit': 0,
    #     'return_type': 'detailed_dictionary',
    #     'plot_number_of_models': 40
    # })

    # results = predictit.main.predict()
    # a = 8 # just for debugger not to stop

    pass
