""" Test module. Auto pytest that can be started ide.
Also in jupyter cell in VS Code and debug

"""

#%%
import sys
import pathlib
import numpy as np
import pandas as pd

script_dir = pathlib.Path(__file__).resolve()
lib_path_str = script_dir.parents[1].as_posix()
sys.path.insert(0, lib_path_str)

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

config["return_type"] = 'best'
config["debug"] = 1
config["plot"] = 0
config["show_plot"] = 0
config["data"] = None
if 'tensorflow_mlp' in config['used_models']:
    del config['used_models']['tensorflow_mlp']
if 'tensorflow_lstm' in config['used_models']:
    del config['used_models']['tensorflow_lstm']
config_original = config.copy()


def test_own_data():
    result = predictit.main.predict(np.random.randn(1, 100), predicts=3, plot=config["show_plot"])
    assert result[0]
    return result


def test_main_from_config():

    if 1:
        config.update(config_original.copy())
        config["data_source"] = 'csv'
        config["csv_test_data_relative_path"] = '5000 Sales Records.csv'
        config["datetime_index"] = 5
        config["freq"] = 'M'
        config["predicted_column"] = 'Units Sold'
        config["datalength"] = 100
        config["data_transform"] = None
        config["repeatit"] = 1
        config["other_columns_length"] = None
        config["lengths"] = 3
        config["criterion"] = 'mape'
        config["remove_outliers"] = 0
        config["compareit"] = 10
        config["last_row"] = 0
        config["correlation_threshold"] = 0.2
        config["optimizeit"] = 0
        config["standardize"] = '01'

        config["used_models"] = {
            "AR (Autoregression)": predictit.models.statsmodels_autoregressive,
            "ARMA": predictit.models.statsmodels_autoregressive,
            "ARIMA (Autoregression integrated moving average)": predictit.models.statsmodels_autoregressive,

            "Autoregressive Linear neural unit": predictit.models.autoreg_LNU,
            "Linear neural unit with weigths predict": predictit.models.autoreg_LNU,
            "Conjugate gradient": predictit.models.conjugate_gradient,

            "Extreme learning machine": predictit.models.sklearn_regression,
            "Gen Extreme learning machine": predictit.models.sklearn_regression,

            "Sklearn regression": predictit.models.sklearn_regression,
            "Bayes ridge regression": predictit.models.sklearn_regression,
            "Hubber regression": predictit.models.sklearn_regression,

            "Compare with average": predictit.models.compare_with_average
        }

    result_1 = predictit.main.predict()
    assert result_1[0]
    return result_1


def test_main_optimize_and_args():

    if 1:
        config.update(config_original.copy())
        config["data_source"] = 'test'
        config["predicts"] = 3
        config["datetime_index"] = ''
        config["freq"] = 'M'
        config["datalength"] = 1000
        config["default_n_steps_in"] = 5
        config["data_transform"] = 'difference'
        config["criterion"] = 'rmse'
        config["remove_outliers"] = 1
        config["compareit"] = 10
        config["last_row"] = 1
        config["correlation_threshold"] = 0.2
        config["optimizeit"] = 1
        config["optimizeit_limit"] = 0.1
        config["optimizeit_details"] = 1
        config["standardize"] = 0
        config["used_models"] = {"Bayes ridge regression": predictit.models.sklearn_regression}
        config["models_parameters"] = {"Bayes ridge regression": {"regressor": 'bayesianridge', "n_iter": 300, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6}}
        config["fragments"] = 4
        config["iterations"] = 2
        config["models_parameters_limits"] = {"Bayes ridge regression": {"alpha_1": [0.1e-6, 3e-6], "alpha_2": [0.1e-6, 3e-6], 'lambda_1': [0.1e-6, 3e-6]}}

    result_2 = predictit.main.predict(debug=config["debug"], data_source='test', predicted_column=[], repeatit=20, lengths=0)
    assert result_2[0]
    return result_2


def test_main_dataframe():

    if 1:
        config.update(config_original.copy())
        df = pd.DataFrame([range(200), range(1200, 1200)]).T
        df['time'] = pd.date_range('2018-01-01', periods=len(df), freq='H')
        config["data"] = df
        config["datetime_index"] = 'time'
        config["freq"] = ''
        config["predicted_column"] = 0
        config["datalength"] = 300
        config['other_columns_length'] = 5
        config["data_transform"] = None
        config["repeatit"] = 1
        config["lengths"] = 3
        config["criterion"] = 'mape'
        config["remove_outliers"] = 0
        config["compareit"] = 10
        config["last_row"] = 0
        config["correlation_threshold"] = 0.4
        config["standardize"] = 'standardize'
        config["predicts"] = 20
        config["default_n_steps_in"] = 40
        config["power_transformed"] = 2

        config["used_models"] = {
            "AR (Autoregression)": predictit.models.statsmodels_autoregressive,

            "Autoregressive Linear neural unit": predictit.models.autoreg_LNU,

            "Compare with average": predictit.models.compare_with_average
        }

    result_1 = predictit.main.predict()
    assert result_1[0]
    return result_1


def test_main_multiple():

    if 1:
        config.update(config_original.copy())
        config["data_source"] = 'csv'
        config["datetime_index"] = 5
        config["freqs"] = ['D', 'M']
        config["csv_test_data_relative_path"] = '5000 Sales Records.csv'
        config["predicted_columns"] = ['Units Sold', 'Total Profit']
        config["optimizeit"] = 0
        config["repeatit"] = 0
        config["lengths"] = 0

    result_multiple = predictit.main.predict_multiple_columns()
    assert result_multiple[0][0][0]
    return result_multiple


def test_compare_models():
    config.update(config_original.copy())
    data_length = 1000
    config["optimizeit"] = 0
    data_all = {'sin': predictit.test_data.generate_test_data.gen_sin(data_length), 'Sign': predictit.test_data.generate_test_data.gen_sign(data_length), 'Random data': predictit.test_data.generate_test_data.gen_random(data_length)}
    predictit.main.compare_models(data_all)


if __name__ == "__main__":
    # result = test_own_data()
    # print("\n\ntest_main_from_config\n")
    # result_1 = test_main_from_config()
    # print("\n\ntest_main_optimize_and_args\n")
    # result_2 = test_main_optimize_and_args()
    # print("\n\ntest_main_dataframe\n")
    result_2 = test_main_dataframe()
    # print("\n\ntest_main_multiple\n")
    # result_multiple = test_main_multiple()
    # print("\n\ntest_main_multiple\n")
    # test_compare_models()

    pass
