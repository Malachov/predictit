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
matplotlib.use('agg')

sys.path.insert(0, Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[1].as_posix())

warnings.filterwarnings('ignore', message=r"[\s\S]*Matplotlib is currently using agg, which is a non-GUI backend*")

import predictit
config = predictit.configuration.config

predictit.misc._COLORIZE = 0


config.plotit = 0

config_unchanged = config.freeze()

config.update({
    'return_type': 'best',
    'predicted_column': 0,
    'debug': 1,
    'printit': 0,
    'plotit': 0,
    'show_plot': 0,
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

np.random.seed(1)


def test_own_data():
    config.plotit = 1
    config.printit = 1
    config.plot_type = 'plotly'
    config.show_plot = 0

    config.other_columns = 0
    config.multiprocessing = 0
    result = predictit.main.predict(data=np.random.randn(100, 5), predicts=3, return_type='detailed_dictionary')
    assert not np.isnan(np.min(result['best']))
    return result


def test_readmes():

    ### Example 1 ###
    config.update(config_unchanged)

    predictions_1 = predictit.main.predict(data=np.random.randn(100, 2), predicted_column=1, predicts=3, return_type='best')

    ### Example 2 ###
    config.update(config_unchanged)

    # You can edit config in two ways
    config.data = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'  # You can use local path on pc as well... "/home/name/mycsv.csv" !
    config.predicted_column = 'Temp'  # You can use index as well
    config.datetime_index = 'Date'  # Will be used in result
    config.freq = "D"  # One day - one value
    config.resample_function = "mean"  # If more values in one day - use mean (more sources)
    config.return_type = 'detailed_dictionary'
    config.debug = 0  # Ignore warnings

    # Or
    config.update({
        'predicts': 12,  # Number of predicted values
        'default_n_steps_in': 12  # Value of recursive inputs in model (do not use too high - slower and worse predictions)
    })

    predictions_2 = predictit.main.predict()

    ### Example 3 ###
    config.update(config_unchanged)

    my_data_array = np.random.randn(2000, 4)  # Define your data here

    # You can compare it on same data in various parts or on different data (check configuration on how to insert dictionary with data names)
    config.update({
        'data_all': (my_data_array[-2000:], my_data_array[-1500:], my_data_array[-1000:])
    })

    compared_models = predictit.main.compare_models()

    ### Example 4 ###
    config.update(config_unchanged)

    config.data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")

    # Define list of columns or '*' for predicting all of the columns
    config.predicted_columns = ['*']

    multiple_columns_prediction = predictit.main.predict_multiple_columns()

    ### Example 5 ###
    config.update(config_unchanged)

    config.update({
        'data': "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        'predicted_column': 'Temp',
        'return_type': 'all_dataframe',
        'optimization': 1,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [4, 8, 10],
        'plot_all_optimized_models': 1,
        'print_table': 2  # Print detailed table
    })

    predictions_optimized_config = predictit.main.predict()

    ### Example 6 ###
    config.update(config_unchanged)

    config.update({
        'data': r'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv',  # Full CSV path with suffix
        'predicted_column': 'Temp',  # Column name that we want to predict

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
            "ARIMA (Autoregression integrated moving average)": {'used_model': 'arima', 'p': 6, 'd': 0, 'q': 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm'},

            "Autoregressive Linear neural unit": {'mi_multiple': 1, 'mi_linspace': (1e-5, 1e-4, 3), 'epochs': 10, 'w_predict': 0, 'minormit': 0},
            "Conjugate gradient": {'epochs': 200},

            "Bayes ridge regression": {'regressor': 'bayesianridge', 'n_iter': 300, 'alpha_1': 1.e-6, 'alpha_2': 1.e-6, 'lambda_1': 1.e-6, 'lambda_2': 1.e-6},
        }
    })

    predictions_configured = predictit.main.predict()


    first_multiple_array = multiple_columns_prediction[list(multiple_columns_prediction.keys())[0]]

    condition_1 = not np.isnan(np.min(predictions_1))
    condition_2 = not np.isnan(np.min(predictions_2['best']))
    condition_3 = compared_models
    condition_4 = not np.isnan(np.nanmax(first_multiple_array))
    condition_5 = not predictions_optimized_config.dropna().empty
    condition_6 = not np.isnan(np.min(predictions_configured))

    assert (condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6)


def test_main_from_config():

    config.update(config_original)
    config.update({
        'data': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv',
        'predicted_column': 'Temp',
        'datetime_index': 'Date',
        'freq': 'D',
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
        'data': 'test',
        'predicted_column': '',
        'predicts': 6,
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
        'optimizeit_plot': 1,
        'standardizeit': 0,
        'multiprocessing': 'pool',
        'used_models': ["Bayes ridge regression"],
        'models_parameters': {"Bayes ridge regression": {"regressor": 'bayesianridge', "n_iter": 300, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6}},
        'fragments': 4,
        'iterations': 2,
        'models_parameters_limits': {"Bayes ridge regression": {"alpha_1": [0.1e-6, 3e-6], "regressor": ['bayesianridge', 'lasso'], "n_iter": [50, 100]}},
    })

    result = predictit.main.predict(data='test', predicted_column=[], repeatit=20)
    assert not np.isnan(result.min())
    return result


def test_config_optimization():

    config.update(config_original)
    df = pd.DataFrame([range(200), range(1000, 1200)]).T
    df['time'] = pd.date_range('2018-01-01', periods=len(df), freq='H')

    config.update({
        'data': df,
        'datetime_index': 'time',
        'freq': '',
        'predicted_column': 0,
        'datalength': 300,
        'other_columns': 1,
        'default_other_columns_length': 5,
        'data_transform': None,
        'repeatit': 1,
        'remove_outliers': 0,
        'remove_nans': 'any_rows',
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
    df = pd.DataFrame([range(200), range(1000, 1200)]).T

    config.update({
        'data': df,
        'predicts': 7,
        'default_n_steps_in': 10,
        'error_criterion': 'mape',


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
        'data': "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        'predicted_column': 'Temp',
        'datalength': 500,
        'use_config_preset': 'fast',
    })

    result = predictit.main.predict()
    assert not np.isnan(np.min(result))

    config.update({
        'use_config_preset': 'normal'
    })

    result = predictit.main.predict()
    assert not np.isnan(np.min(result))
    return result


def test_main_multiple():

    config.update(config_original)
    config.update({
        'data': np.random.randn(300, 3),
        'predicted_columns': [0, 1],
        'error_criterion': 'mse_sklearn',
    })

    result_multiple = predictit.main.predict_multiple_columns()
    first_array = result_multiple[list(result_multiple.keys())[0]]
    assert not np.isnan(np.min(first_array))
    return result_multiple


def test_main_multiple_all_columns():

    config.update(config_original)
    config.update({
        'use_config_preset': 'fast',
        'datetime_index': 'Date',
        'freqs': ['D'],
        'data': 'https://www.stats.govt.nz/assets/Uploads/Effects-of-COVID-19-on-trade/Effects-of-COVID-19-on-trade-1-February-1-July-2020-provisional/Download-data/Effects-of-COVID-19-on-trade-1-February-1-July-2020-provisional.csv',
        'predicted_columns': '*',
        'remove_nans': 'any_columns',
        'optimization': 0,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [12, 20, 40],
    })

    result_multiple = predictit.main.predict_multiple_columns()
    first_array = result_multiple[list(result_multiple.keys())[0]]
    assert not np.isnan(np.min(first_array))
    return result_multiple


def test_compare_models():
    config.update(config_original)
    data_all = None

    result = predictit.main.compare_models(data_all=data_all)

    assert result


def test_compare_models_list():
    config.update(config_original)
    dummy_data = np.random.randn(500)
    data_all = [dummy_data[:200], dummy_data[200: 400], dummy_data[300: ]]

    result = predictit.main.compare_models(data_all=data_all)

    assert result


def test_compare_models_with_optimization():
    config.update(config_original)
    config.update({
        'data_all': {'sin': [predictit.test_data.generate_test_data.gen_sin(), 0],
                     'Sign': [predictit.test_data.generate_test_data.gen_sign(), 0],
                     'Random data': [predictit.test_data.generate_test_data.gen_random(), 0]},
        'optimization': 1,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [4, 6, 8]
    })

    result = predictit.main.compare_models()

    assert result


##################
### Unit tests ###
##################

def test_preprocessing():

    ### Column with nan should be removed, row with outlier big value should be removed.
    ### Preprocessing and inverse will be made and than just  compare with good results

    test_df = pd.DataFrame(np.array([range(5), range(20, 25), range(25, 30), np.random.randn(5)]).T, columns=["First", "Predicted", "Ignored", "Ignored 2"])
    test_df.iloc[2, 1] = 500
    test_df.iloc[2, 2] = np.nan

    config = predictit.configuration.config
    config.predicted_column = 1
    config.other_columns = 1
    config.datetime_index = False
    config.remove_nans = "any_columns"

    data_df, df_df, _ = predictit.data_preprocessing.data_consolidation(test_df, config)

    # Predicted column moved to index 0, but for test reason test, use different one
    processed_df, last_undiff_value_df, final_scaler_df = predictit.data_preprocessing.preprocess_data(
        df_df, multicolumn=1, remove_outliers=1, correlation_threshold=0.6, data_transform='difference', standardizeit='standardize')

    inverse_processed_df = predictit.data_preprocessing.preprocess_data_inverse(
        processed_df['Predicted'].iloc[1:], final_scaler=final_scaler_df, last_undiff_value=test_df['Predicted'][0],
        standardizeit='standardize', data_transform='difference')

    processed_df_2, last_undiff_value_df_2, final_scaler_df_2 = predictit.data_preprocessing.preprocess_data(
        data_df, multicolumn=1, remove_outliers=1, correlation_threshold=0.6, data_transform='difference',
        standardizeit='standardize')

    inverse_processed_df_2 = predictit.data_preprocessing.preprocess_data_inverse(
        processed_df_2[1:, 0], final_scaler=final_scaler_df_2, last_undiff_value=test_df['Predicted'][0],
        standardizeit='standardize', data_transform='difference')

    correct_preprocessing = np.array([[-0.707107, -0.707107],
                                      [1.414214, 1.414214],
                                      [-0.707107, -0.707107]])

    check_1 = np.allclose(processed_df.values, correct_preprocessing)
    check_2 = np.allclose(processed_df_2, correct_preprocessing)

    correct_inveerse_preprocessing = np.array([22., 23.])

    check_3 = np.allclose(inverse_processed_df, correct_inveerse_preprocessing)
    check_4 = np.allclose(inverse_processed_df_2, correct_inveerse_preprocessing)

    assert check_1 and check_2 and check_3 and check_4


# def test_GUI():
#     predictit.gui_start.run_gui()


# For deeper debug, uncomment problematic test
if __name__ == "__main__":
    # result = test_own_data()
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
    test_preprocessing()

    ## Custom use case test...

    # # You can edit config in two ways
    # config.data = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'  # You can use local path on pc as well... "/home/name/mycsv.csv" !
    # config.predicted_column = 'Temp'  # You can use index as well
    # config.datetime_index = 'Date'  # Will be used in result

    # # Or
    # config.update({
    #     'predicts': 14,  # Number of predicted values
    #     'default_n_steps_in': 15  # Value of recursive inputs in model (do not use too high - slower and worse predictions)
    # })

    # predictions = predictit.main.predict()
    pass
