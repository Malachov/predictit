""" Test module. Auto pytest that can be started ide.
Also in jupyter cell in VS Code and debug

"""

#%%
import sys
import pathlib
import numpy as np
import importlib

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
from predictit import config


def test_own_data():
    result = predictit.main.predict(np.random.randn(1, 100), predicts=3, plot=0)
    assert result[0]
    return result


def test_main_from_config():

    if 1:
        config.plot = 0
        config.data_source = 'csv'
        config.csv_test_data_relative_path = '5000 Sales Records.csv'
        config.date_index = 5
        config.freq = ''
        config.predicted_column = 'Units Sold'
        config.datalength = 100
        config.data_transform = None
        config.repeatit = 1
        config.other_columns = 1
        config.lengths = 3
        config.criterion = 'mape'
        config.remove_outliers = 0
        config.compareit = 10
        config.last_row = 0
        config.correlation_threshold = 0.2
        config.optimizeit = 0
        config.optimizeit_final = 0
        config.piclkeit = 1
        config.standardizeit = 0
        config.used_models = {

                    "AR (Autoregression)": predictit.models.ar,
                    "ARMA": predictit.models.arma,
                    "ARIMA (Autoregression integrated moving average)": predictit.models.arima,
                    "SARIMAX (Seasonal ARIMA)": predictit.models.sarima,

                    "Autoregressive Linear neural unit": predictit.models.autoreg_LNU,
                    "Linear neural unit with weigths predict": predictit.models.autoreg_LNU_withwpred,
                    "Conjugate gradient": predictit.models.cg,

                    "Extreme learning machine": predictit.models.regression,
                    "Gen Extreme learning machine": predictit.models.regression,

                    "Sklearn regression": predictit.models.regression,
                    "Bayes ridge regression": predictit.models.regression,
                    "Hubber regression": predictit.models.regression,

                    "Compare with average": predictit.models.compare_with_average

        }

    result_1 = predictit.main.predict()
    assert result_1[0]
    return result_1


def test_main_from_config_and_args():

    if 1:
        config.plot = 0
        config.debug = 1
        config.date_index = ''
        config.freq = 'M'
        config.datalength = 1000
        config.data_transform = 'difference'
        config.criterion = 'rmse'
        config.remove_outliers = 1
        config.compareit = 10
        config.last_row = 1
        config.correlation_threshold = 0.2
        config.optimizeit = 1
        config.optimizeit_final = 0
        config.optimizeit_limit = 0.1
        config.optimizeit_details = 1
        config.optimizeit_final_limit = 0.01
        config.piclkeit = 0
        config.standardizeit = 1

        config.used_models = {
            "Bayes ridge regression": predictit.models.regression,
        }

        n_steps_in = 3
        output_shape = 'batch'  # Batch or one

        config.models_parameters = {
                "Bayes ridge regression": {"n_steps_in": n_steps_in, "regressor": 'bayesianridge', "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "n_iter": 300, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6},
        }

        fragments = 4
        iterations = 2

        # This is for final optimization of the best model, not for all models
        config.fragments_final = fragments
        config.iterations_final = iterations

        # Threshold values
        # If you need integers, type just number, if you need float, type dot (e.g. 2.0)

        # This boundaries repeat across models
        steps = [2, 200]

        config.models_parameters_limits = {

                "Bayes ridge regression": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "alpha_1": [0.1e-6, 3e-6], "alpha_2": [0.1e-6, 3e-6]},

        }

    result_2 = predictit.main.predict(debug=1, data_source='test', predicted_column=[], repeatit=0, lengths=3)
    assert result_2[0]
    return result_2


def test_main_multiple():

    if 1:
        config.plot = 0
        config.debug = 1
        config.data_source = 'csv'
        config.date_index = 5
        config.freqs = ['D', 'M']
        config.csv_test_data_relative_path = '5000 Sales Records.csv'
        config.predicted_columns = ['Units Sold', 'Total Profit']
        config.optimizeit = 0
        config.optimizeit_final = 0
        config.repeatit = 0
        config.lengths = 0

    result_multiple = predictit.main.predict_multiple_columns()
    assert result_multiple[0][0][0]
    return result_multiple


def test_compare_models():
    importlib.reload(config)
    config.plot = 0
    config.debug = 1
    data_length = 1000
    config.from_pickled = 0
    config.optimizeit = 0
    config.optimizeit_final = 0
    data_all = {'sin': predictit.test_data.generate_test_data.gen_sin(data_length), 'Sign': predictit.test_data.generate_test_data.gen_sign(data_length), 'Random data': predictit.test_data.generate_test_data.gen_random(data_length)}
    predictit.main.compare_models(data_all)


if __name__ == "__main__":
    result = test_own_data()
    result_1 = test_main_from_config()
    result_2 = test_main_from_config_and_args()
    #result_multiple = test_main_multiple()
    #test_compare_models()
