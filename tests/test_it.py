""" Test module. Auto pytest that can be started ide.
Also in jupyter cell in VS Code and debug"""

#%%

import sys
import pathlib

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
from predictit.models.sklearn_regression import get_regressors


def test_main_from_config_1():

    if 1:
        config.plot = 0
        config.data_source = 'csv'
        config.csv_from_test_data_name = '5000 Sales Records.csv'
        config.date_index = 5
        config.freqs = ''
        config.predicted_columns = ['Units Sold', 'Total Profit']
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

                    "Extreme learning machine": predictit.models.elm,
                    "Gen Extreme learning machine": predictit.models.elm_gen,

                    #"LSTM": predictit.models.lstm,
                    #"Bidirectional LSTM": predictit.models.lstm_bidirectional,
                    #"LSTM batch": predictit.models.lstm_batch,

                    "Sklearn regression": predictit.models.sklearn_regression,

                    "Bayes ridge regression": predictit.models.regression,
                    "Hubber regression": predictit.models.regression,
                    "Regression": predictit.models.regression,

                    "Compare with average": predictit.models.compare_with_average

        }

    result_1 = predictit.main.predict()
    assert result_1[0]
    return result_1


def test_main_from_config_2():

    if 1:
        config.plot = 0
        config.data_source = 'test'
        config.date_index = ''
        config.freqs = 'M'
        config.predicted_columns = []
        config.datalength = 100
        config.data_transform = 'difference'
        config.repeatit = 0
        config.other_columns = 1
        config.lengths = 0
        config.criterion = 'rmse'
        config.remove_outliers = 1
        config.compareit = 10
        config.last_row = 1
        config.correlation_threshold = 0.2
        config.optimizeit = 0
        config.optimizeit_final = 1
        config.optimizeit_limit = 0.001
        config.optimizeit_final_limit = 0.001
        config.piclkeit = 0
        config.standardizeit = 1

        config.used_models = {
                    "AR (Autoregression)": predictit.models.ar,
                    "ARMA": predictit.models.arma,
                    "ARIMA (Autoregression integrated moving average)": predictit.models.arima,
                    "SARIMAX (Seasonal ARIMA)": predictit.models.sarima,

                    "Autoregressive Linear neural unit": predictit.models.autoreg_LNU,
                    "Linear neural unit with weigths predict": predictit.models.autoreg_LNU_withwpred,
                    "Conjugate gradient": predictit.models.cg,

                    "Extreme learning machine": predictit.models.elm,
                    "Gen Extreme learning machine": predictit.models.elm_gen,

                    #"LSTM": predictit.models.lstm,
                    #"Bidirectional LSTM": predictit.models.lstm_bidirectional,
                    #"LSTM batch": predictit.models.lstm_batch,

                    "Bayes ridge regression": predictit.models.regression,
                    "Hubber regression": predictit.models.regression,
                    "Sklearn regression": predictit.models.regression,

                    "Compare with average": predictit.models.compare_with_average
        }

        n_steps_in = 3
        output_shape = 'batch'  # Batch or one
        saveit = 0

        config.models_parameters = {

                #TODO
                "ETS": {"plot": 0},


                "AR (Autoregression)": {"plot": 0, 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
                "ARMA": {"plot": 0, "p": 3, "q": 0, 'method': 'mle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs', 'forecast_type': 'in_sample'},
                "ARIMA (Autoregression integrated moving average)": {"p": 3, "d": 0, "q": 0, "plot": 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm', 'forecast_type': 'out_of_sample'},
                "SARIMAX (Seasonal ARIMA)": {"plot": 0, "p": 4, "d": 0, "q": 0, "pp": 1, "dd": 0, "qq": 0, "season": 12, "method": "lbfgs", "trend": 'nc', "enforce_invertibility": False, "enforce_stationarity": False, "verbose": 0},


            # "ARIMA (Autoregression integrated moving average)": {"p": [1, maxorder], "d": [0,1], "q": order, 'method': ['css-mle', 'mle', 'css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},


                "Autoregressive Linear neural unit": {"plot": 0, "lags": n_steps_in, "mi": 1, "minormit": 0, "tlumenimi": 1},
                "Linear neural unit with weigths predict": {"plot": 0, "lags": n_steps_in, "mi": 1, "minormit": 0, "tlumenimi": 1},
                "Conjugate gradient": {"n_steps_in": 30, "epochs": 5, "constant": 1, "other_columns_lenght": None},

                "Extreme learning machine": {"n_steps_in": 20, "output_shape": 'one_step', "other_columns_lenght": None, "constant": None, "n_hidden": 20, "alpha": 0.3, "rbf_width": 0, "activation_func": 'selu'},
                "Gen Extreme learning machine": {"n_steps_in": 20, "output_shape": 'one_step', "other_columns_lenght": None, "constant": None, "alpha": 0.5},

                # TODO finish lstm
                "LSTM": {"n_steps_in": 50, "save": saveit, "already_trained": 0, "epochs": 70, "units": 50, "optimizer": 'adam', "loss": 'mse', "verbose": 1, "activation": 'relu', "timedistributed": 0, "metrics": ['mape']},
                "LSTM batch": {"n_steps_in": n_steps_in, "n_features": 1, "epochs": 70, "units": 50, "optimizer": 'adam', "loss": 'mse', "verbose": 1, 'dropout': 0, "activation": 'relu'},
                "Bidirectional LSTM": {"n_steps_in": n_steps_in, "epochs": 70, "units": 50, "optimizer": 'adam', "loss": 'mse', "verbose": 0},

                "Bayes ridge regression": {"n_steps_in": n_steps_in, "regressor": 'bayesianridge', "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "n_iter": 300, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6},
                "Hubber regression": {"regressor": 'huber', "n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "epsilon": 1.35, "alpha": 0.0001},
                "Sklearn regression": {"regressor": 'linear', "n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alpha": 0.0001, "n_iter": 100, "epsilon": 1.35, "alphas": [0.1, 0.5, 1], "gcv_mode": 'auto', "solver": 'auto'}
        }

        fragments = 4
        iterations = 2

        # This is for final optimalisation of the best model, not for all models
        config.fragments_final = 2 * fragments
        config.iterations_final = 2 * iterations

        # Threshold values
        # If you need integers, type just number, if you need float, type dot (e.g. 2.0)

        # This boundaries repeat across models
        steps = [2, 200]
        alpha = [0.0, 1.0]
        epochs = [2, 100]
        units = [1, 100]
        order = [0, 5]

        maxorder = 6

        config.models_parameters_limits = {
                "AR (Autoregression)": {"ic": ['aic', 'bic', 'hqic', 't-stat'], "trend": ['c', 'nc'], "solver": ['bfgs', 'newton', 'nm', 'cg']},

                "ARMA": {"p": [1, maxorder], "q": order, 'method': ['css-mle', 'mle', 'css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'], 'forecast_type': ['in_sample', 'out_of_sample']},
                #"ARIMA (Autoregression integrated moving average)": {"p": [1, maxorder], "d": [0,1], "q": order, 'method': ['css-mle', 'mle', 'css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},
                "ARIMA (Autoregression integrated moving average)": {"p": [1, maxorder], "d": [0, 1], "q": order, 'method': ['css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},
                "SARIMAX (Seasonal ARIMA)": {"p": [1, maxorder], "d": order, "q": order, "pp": order, "dd": order, "qq": order, "season": order, "method": ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'], "trend": ['n', 'c', 't', 'ct'], "enforce_invertibility": [True, False], "enforce_stationarity": [True, False], 'forecast_type': ['in_sample', 'out_of_sample']},

                "Autoregressive Linear neural unit": {"lags": steps, "mi": [1.0, 10.0], "minormit": [0, 1], "tlumenimi": [0.0, 100.0]},
                "Linear neural unit with weigths predict": {"lags": steps, "mi": [1.0, 10.0], "minormit": [0, 1], "tlumenimi": [0.0, 100.0]},
                "Conjugate gradient": {"n_steps_in": steps, "epochs": epochs, "constant": [None, 1], "other_columns_lenght": [None, steps[1]]},

                "Extreme learning machine": {"n_steps_in": steps, "n_hidden": [2, 300], "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "alpha": alpha, "rbf_width": [0.0, 10.0], "activation_func": predictit.models.activations},
                "Gen Extreme learning machine": {"n_steps_in": steps, "alpha": alpha, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]]},

                "Bayes Ridge Regression": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "alpha_1": [0.1e-6, 3e-6], "alpha_2": [0.1e-6, 3e-6], "lambda_1": [0.1e-6, 3e-6], "lambda_2": [0.1e-7, 3e-6]},
                "Hubber regression": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "epsilon": [1.01, 5.0], "alpha": alpha},
                "Sklearn regression": {"n_steps_in": steps, "regressor": [get_regressors], "output_shape": ['batch', 'one_step'], "other_columns_lenght": [None, steps[1]], "constant": [None, 1], "alpha": alpha, "n_iter": [100, 500], "epsilon": [1.01, 5.0], "alphas": [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]], "gcv_mode": ['auto', 'svd', 'eigen'], "solver": ['auto', 'svd', 'eigen']}
        }

    result_2 = predictit.main.predict()
    assert result_2[0]
    return result_2


def test_main_multiple():

    if 1:
        config.plot = 0
        config.data_source = 'csv'
        config.date_index = 5
        config.freqs = ['D', 'M']
        config.csv_from_test_data_name = '5000 Sales Records.csv'
        config.predicted_columns = ['Units Sold', 'Total Profit']


    result_multiple = predictit.main.predict_multiple_columns()
    assert result_multiple[0][0][0]
    return result_multiple


if __name__ == "__main__":
    #result_1 = test_main_from_config_1()
    #result_2 = test_main_from_config_2()
    result_multiple = test_main_multiple()
