""" Test module. Auto pytest that can be started ide.
Also in jupyter cell in VS Code and debug

"""

#%%
import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, pathlib.Path(__file__).resolve().parents[1].as_posix())

try:
    __IPYTHON__
    from IPython import get_ipython
    ipython = get_ipython()
    magic_load_ex = '%load_ext autoreload'
    magic_autoreload = '%autoreload 2'

    ipython.magic(magic_load_ex)
    ipython.magic(magic_autoreload)

except Exception:
    pass

import predictit
from predictit.config import config

config.update({
    'return_type': 'best',
    'debug': 1,
    'plot': 0,
    'show_plot': 0,
    'data': None,
    'used_models': {
        "AR (Autoregression)": predictit.models.statsmodels_autoregressive,
        "Conjugate gradient": predictit.models.conjugate_gradient,
        "Extreme learning machine": predictit.models.sklearn_regression,
        "Sklearn regression": predictit.models.sklearn_regression,
        "Compare with average": predictit.models.compare_with_average
    }
})

config_original = config.copy()


def test_own_data():
    config.update({
        'other_columns': 0})
    result = predictit.main.predict(np.random.randn(5, 100), predicts=3, plot=config["show_plot"])
    assert result[0]
    return result


def test_main_from_config():

    config.update(config_original.copy())
    config.update({
        'data_source': 'csv',
        'csv_test_data_relative_path': '5000 Sales Records.csv',
        'datetime_index': 5,
        'freq': 'M',
        'predicted_column': 'Units Sold',
        'datalength': 100,
        'data_transform': None,
        'repeatit': 1,
        'other_columns': 1,
        'default_other_columns_length': None,
        'lengths': 3,
        'error_criterion': 'mape',
        'remove_outliers': 0,
        'print_number_of_models': 10,
        'last_row': 0,
        'correlation_threshold': 0.2,
        'optimizeit': 0,
        'standardize': '01',

        'used_models': {
            "Hubber regression": predictit.models.sklearn_regression,
        }
    })

    result_1 = predictit.main.predict()
    assert result_1[0]
    return result_1


def test_main_optimize_and_args():

    if 1:
        config.update(config_original.copy())
        config.update({
            'data_source': 'test',
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
            'optimizeit_details': 1,
            'standardize': 0,
            'used_models': {"Bayes ridge regression": predictit.models.sklearn_regression},
            'models_parameters': {"Bayes ridge regression": {"regressor": 'bayesianridge', "n_iter": 300, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6}},
            'fragments': 4,
            'iterations': 2,
            'models_parameters_limits': {"Bayes ridge regression": {"alpha_1": [0.1e-6, 3e-6], "alpha_2": [0.1e-6, 3e-6], 'lambda_1': [0.1e-6, 3e-6]}},
        })

    result_2 = predictit.main.predict(debug=config["debug"], data_source='test', predicted_column=[], repeatit=20, lengths=0)
    assert result_2[0]
    return result_2


def test_main_dataframe():

    if 1:
        config.update(config_original.copy())
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
            'lengths': 3,
            'error_criterion': 'dtw',
            'remove_outliers': 0,
            'last_row': 0,
            'correlation_threshold': 0.4,
            'standardize': 'standardize',
            'predicts': 7,
            'default_n_steps_in': 40,
            'power_transformed': 2,

            'used_models': {
                "AR (Autoregression)": predictit.models.statsmodels_autoregressive,
                "Autoregressive Linear neural unit": predictit.models.autoreg_LNU,
                "Compare with average": predictit.models.compare_with_average
            }
        })

    result_1 = predictit.main.predict()
    assert result_1[0]
    return result_1


def test_main_multiple():

    if 1:
        config.update(config_original.copy())
        config.update({
            'data_source': 'csv',
            'datetime_index': 5,
            'freqs': ['D', 'M'],
            'csv_test_data_relative_path': '5000 Sales Records.csv',
            'predicted_columns': ['Units Sold', 'Total Profit'],
            'optimizeit': 0,
            'repeatit': 0,
            'lengths': 0
        })

    result_multiple = predictit.main.predict_multiple_columns()
    assert result_multiple[0][0][0]
    return result_multiple


def test_compare_models():
    config.update(config_original.copy())
    data_length = 1000
    data_all = {'sin': predictit.test_data.generate_test_data.gen_sin(data_length), 'Sign': predictit.test_data.generate_test_data.gen_sign(data_length), 'Random data': predictit.test_data.generate_test_data.gen_random(data_length)}
    try:
        predictit.main.compare_models(data_all)
        ok = 'fine'
    except Exception:
        ok = 'nok'
    assert ok == 'fine'


if __name__ == "__main__":
    # result = test_own_data()
    # print("\n\ntest_main_from_config\n")
    # result_1 = test_main_from_config()
    # print("\n\ntest_main_optimize_and_args\n")
    # result_2 = test_main_optimize_and_args()
    # print("\n\ntest_main_dataframe\n")
    # result_3 = test_main_dataframe()
    # print("\n\ntest_main_multiple\n")
    # result_multiple = test_main_multiple()
    # print("\n\ntest_main_multiple\n")
    test_compare_models()

    pass
