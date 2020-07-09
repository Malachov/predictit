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

sys.path.insert(0, Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[1].as_posix())
import predictit

config = predictit.configuration.config
predictit.misc._COLORIZE = 0

config.update({
    'return_type': 'best',
    'predicted_column': 0,
    'debug': 1,
    'printit': 0,
    'plot': 0,
    'show_plot': 1,
    'data': None,
    'datalength': 500,
    'analyzeit': 0,
    'optimization': 0,
    'optimizeit': 0,
    'used_models': {
        "AR (Autoregression)": predictit.models.statsmodels_autoregressive,
        "Conjugate gradient": predictit.models.conjugate_gradient,
        # "Extreme learning machine": predictit.models.sklearn_regression,
        "Sklearn regression": predictit.models.sklearn_regression,
        # "Compare with average": predictit.models.compare_with_average
    }
})

config_original = config.freeze()



def test_own_data():
    config.other_columns = 0
    config.multiprocessing = 0
    config.return_type = 'all'
    result = predictit.main.predict(data=np.random.randn(5, 100), predicts=3)
    assert result[0][0][0]
    return result


def test_main_from_config():

    config.update(config_original)
    config.update({
        'data_source': 'csv',
        'csv_full_path': 'https://datahub.io/core/global-temp/r/monthly.csv',
        'predicted_column': 'Mean',
        'datetime_index': 'Date',
        'freq': 'D',
        'return_type': 'results_dataframe',
        'optimization': 1,
        'optimization_variable': 'data_transform',
        'optimization_values': [0, 'difference'],
        'print_number_of_models': 10,

        'last_row': 0,
        'correlation_threshold': 0.2,
        'optimizeit': 0,
        'standardizeit': '01',
        'multiprocessing': 'process',
        'smoothit': (19, 2),

        'used_models': {
            "Bayes ridge regression": predictit.models.sklearn_regression,
        }
    })

    result = predictit.main.predict()
    assert result.iloc[0][0]
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
        'used_models': {"Bayes ridge regression": predictit.models.sklearn_regression},
        'models_parameters': {"Bayes ridge regression": {"regressor": 'bayesianridge', "n_iter": 300, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6}},
        'fragments': 4,
        'iterations': 2,
        'models_parameters_limits': {"Bayes ridge regression": {"alpha_1": [0.1e-6, 3e-6], "alpha_2": [0.1e-6, 3e-6]}},
    })

    result = predictit.main.predict(data_source='test', predicted_column=[], repeatit=20)
    assert result[0]
    return result


def test_main_dataframe():

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
        'mode': 'predict',
        'repeatit': 1,
        'error_criterion': 'dtw',
        'remove_outliers': 0,
        'optimization': 1,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [12, 20, 40],
        'last_row': 0,
        'correlation_threshold': 0.4,
        'standardizeit': 'standardize',
        'predicts': 7,
        'smoothit': 0,
        'default_n_steps_in': 40,
        'power_transformed': 2,

        'used_models': {
            "AR (Autoregression)": predictit.models.statsmodels_autoregressive,
            "Autoregressive Linear neural unit": predictit.models.autoreg_LNU,
            "Compare with average": predictit.models.compare_with_average
        }
    })

    result = predictit.main.predict()
    assert result[0]
    return result


def test_presets():
    config.update(config_original)
    config.update({
        'data_source': 'csv',
        'csv_full_path': "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        'predicted_column': 'Temp',
        'datalength': 500,
        'use_config_preset': 'fast'
    })

    result = predictit.main.predict()
    assert result[0]

    config.update({
        'use_config_preset': 'normal'
    })

    result = predictit.main.predict()
    assert (result[0] and result[0] is not np.nan)

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

    assert(list(result_multiple.values())[0][0] and list(result_multiple.values())[0][0] is not np.nan)
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
    assert(list(result_multiple.values())[0][0] and list(result_multiple.values())[0][0] is not np.nan)
    return result_multiple


def test_compare_models():
    config.update(config_original)
    data_all = {'sin': [predictit.test_data.generate_test_data.gen_sin(), 0], 'Sign': [predictit.test_data.generate_test_data.gen_sign(), 0], 'Random data': [predictit.test_data.generate_test_data.gen_random(), 0]}
    try:
        predictit.main.compare_models(data_all=data_all)
        ok = 'fine'
    except Exception:
        ok = 'nok'
    assert ok == 'fine'


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

    try:
        predictit.main.compare_models()
        ok = 'fine'
    except Exception:
        ok = 'nok'
    assert ok == 'fine'



if __name__ == "__main__":
    # result = test_own_data()
    # print("\n\ntest_main_from_config\n")
    # result1 = test_main_from_config()
    # print("\n\ntest_main_optimize_and_args\n")get_master
    # result_2 = test_main_optimize_and_args()
    # print("\n\ntest_main_dataframe\n")
    # result_3 = test_main_dataframe()
    # print("\n\ntest_fast_preset\n")
    # result_4 = test_presets()
    # print("\n\ntest_main_multiple\n")
    # result_multiple = test_main_multiple()
    # print("\n\ntest_main_multiple\n")
    # test_main_multiple_all_columns()
    # print("\n\ntest_compare_models_with_optimization\n")
    # test_compare_models()
    # print("\n\ntest_compare_models_with_optimization\n")
    # test_compare_models_with_optimization()

    pass
