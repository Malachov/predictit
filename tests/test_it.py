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


sys.path.insert(0, Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[0].as_posix())

from visual import visual_test

sys.path.insert(0, Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[1].as_posix())

import predictit

Config = predictit.configuration.Config
Config.plotit = 0
config_unchanged = Config.freeze()

mylogging._COLORIZE = 0
warnings.filterwarnings('ignore', message=r"[\s\S]*Matplotlib is currently using agg, which is a non-GUI backend*")
matplotlib.use('agg')


Config.update({
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

config_original = Config.freeze()

np.random.seed(2)


def test_1():
    Config.update({
        'data': "https://blockchain.info/unconfirmed-transactions?format=json",
        'data_orientation': "index",
        'request_datatype_suffix': ".json",
        'predicted_table': 'txs',
        'predicted_column': "weight",
        'datalength': 200,
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
    assert not np.isnan(np.min(list(result.values())[0]['tests_results']))
    return result


def test_readmes():

    ### Example 1 ###
    Config.update(config_unchanged)

    predictions_1 = predictit.main.predict(data=np.random.randn(100, 2), predicted_column=1, predicts=3, return_type='best')

    mydata = pd.DataFrame(np.random.randn(100, 2), columns=['a', 'b'])
    predictions_1_positional = predictit.main.predict(mydata, 'b')

    ### Example 2 ###
    Config.update(config_unchanged)

    # You can edit Config in two ways
    Config.data = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'  # You can use local path on pc as well... "/home/name/mycsv.csv" !
    Config.predicted_column = 'Temp'  # You can use index as well
    Config.datetime_column = 'Date'  # Will be used in result
    Config.freq = "D"  # One day - one value
    Config.resample_function = "mean"  # If more values in one day - use mean (more sources)
    Config.return_type = 'detailed_dictionary'
    Config.debug = 0  # Ignore warnings

    # Or
    Config.update({
        'predicts': 12,  # Number of predicted values
        'default_n_steps_in': 12  # Value of recursive inputs in model (do not use too high - slower and worse predictions)
    })

    predictions_2 = predictit.main.predict()

    ### Example 3 ###
    Config.update(config_unchanged)

    my_data_array = np.random.randn(500, 3)  # Define your data here

    # You can compare it on same data in various parts or on different data (check configuration on how to insert dictionary with data names)
    Config.update({
        'data_all': {'First part': (my_data_array[-500:], 0), 'Second part': (my_data_array[-200:], 2)}
    })

    compared_models = predictit.main.compare_models()

    ### Example 4 ###
    Config.update(config_unchanged)

    Config.data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")

    # Define list of columns or '*' for predicting all of the columns
    Config.predicted_columns = ['*']

    multiple_columns_prediction = predictit.main.predict_multiple_columns()

    ### Example 5 ###
    Config.update(config_unchanged)

    Config.update({
        'data': "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        'predicted_column': 'Temp',
        'return_type': 'all_dataframe',
        'optimization': 1,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [4, 5, 6],
        'plot_all_optimized_models': 1,
        'print_table': 1,  # Print detailed table
        'used_models': ['AR (Autoregression)', 'Conjugate gradient', 'Sklearn regression']

    })

    predictions_optimized_config = predictit.main.predict()

    ### Example 6 ###

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

    ### Example 7 ###
    Config.update(config_unchanged)

    Config.update({
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
    condition_6 = not np.isnan(np.min(predictions_configured))

    assert (condition_1 and condition_1_a and condition_2 and condition_3 and condition_4 and condition_5 and condition_6)


def test_main_from_config():

    Config.update(config_original)
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

    Config.update(config_original)
    Config.update({
        'data': 'test',
        'predicted_column': '',
        'predicts': 6,
        'datetime_column': '',
        'freq': 'M',
        'datalength': 1000,
        'default_n_steps_in': 5,
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
        'models_parameters_limits': {"Bayes ridge regression": {"alpha_1": [0.1e-6, 3e-6], "regressor": ['bayesianridge', 'lasso'], "n_iter": [50, 100]}},
    })

    result = predictit.main.predict(data='test', predicted_column=[], repeatit=20)
    assert not np.isnan(result.min())
    return result


def test_config_optimization():

    Config.update(config_original)
    df = pd.DataFrame([range(200), range(1000, 1200)]).T
    df['time'] = pd.date_range('2018-01-01', periods=len(df), freq='H')

    Config.update({
        'data': df,
        'datetime_column': 'time',
        'freq': '',
        'predicted_column': 0,
        'datalength': 300,
        'other_columns': 1,
        'default_other_columns_length': 5,
        'data_transform': None,
        'repeatit': 1,
        'remove_outliers': 0,
        'remove_nans_threshold': 0.8,
        'remove_nans_or_replace': 'mean',
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

    Config.update(config_original)
    df = pd.DataFrame([range(200), range(1000, 1200)]).T

    Config.update({
        'data': df,
        'predicts': 7,
        'default_n_steps_in': 10,
        'error_criterion': 'mape',


        'used_models': [
            'ARMA', 'ARIMA (Autoregression integrated moving average)', 'autoreg', 'SARIMAX (Seasonal ARIMA)',
            'Autoregressive Linear neural unit', 'Autoregressive Linear neural unit normalized', 'Linear neural unit with weights predict',
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


def test_other_models_functions():
    tf_optimizers = predictit.models.tensorflow.get_optimizers_loses_activations()
    sklearn_regressors = predictit.models.sklearn_regression.get_regressors()

    sequentions = mdp.inputs.make_sequences(np.random.randn(100, 1), 5)
    predictit.models.autoreg_LNU.train(sequentions, plot=1)

    assert tf_optimizers and sklearn_regressors


def test_presets():
    Config.update(config_original)
    Config.update({
        'data': "https://blockchain.info/unconfirmed-transactions?format=json",
        'request_datatype_suffix': '.json',
        'predicted_table': 'txs',
        'data_orientation': 'index',
        'predicted_column': 'weight',
        'datalength': 500,
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

    Config.update(config_original)
    Config.update({
        'data': np.random.randn(300, 3),
        'predicted_columns': [0, 1],
        'error_criterion': 'mse_sklearn',
    })

    result_multiple = predictit.main.predict_multiple_columns()
    first_array = result_multiple[list(result_multiple.keys())[0]]
    assert not np.isnan(np.min(first_array))
    return result_multiple


def test_main_multiple_all_columns():

    Config.update(config_original)
    Config.update({
        'use_config_preset': 'fast',
        'datetime_column': 'Date',
        'freqs': ['D'],
        'data': 'https://www.stats.govt.nz/assets/Uploads/Effects-of-COVID-19-on-trade/Effects-of-COVID-19-on-trade-1-February-1-July-2020-provisional/Download-data/Effects-of-COVID-19-on-trade-1-February-1-July-2020-provisional.csv',
        'predicted_columns': '*',
        'remove_nans_threshold': 0.9,
        'remove_nans_or_replace': 2,
        'optimization': 0,
        'optimization_variable': 'default_n_steps_in',
        'optimization_values': [12, 20, 40],
    })

    result_multiple = predictit.main.predict_multiple_columns()
    first_array = result_multiple[list(result_multiple.keys())[0]]
    assert not np.isnan(np.min(first_array))
    return result_multiple


def test_compare_models():
    Config.update(config_original)
    data_all = None

    result = predictit.main.compare_models(data_all=data_all)

    assert result


def test_compare_models_list():
    Config.update(config_original)
    dummy_data = np.random.randn(500)
    data_all = [dummy_data[:200], dummy_data[200: 400], dummy_data[300:]]

    result = predictit.main.compare_models(data_all=data_all)

    assert result


def test_compare_models_with_optimization():
    Config.update(config_original)
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

    # You can edit Config in two ways

    # Or
    # Config.update(config_unchanged)
    # Config.update({
    #     "data": '/home/dan/Desktop/archive/Jan_2019_ontime.csv',
    #     "predicted_column": 'DISTANCE',
    #     "datalength": 100000,

    #     # 'pool' or 'process' or 0
    #     "multiprocessing": 0
    # })

    # predictions = predictit.main.predict()

    pass


#%%

# import categorical_embedder as ce
# from sklearn.model_selection import train_test_split

# # df = pd.DataFrame([['Hodne', 20, 'efs', 3], ['stredne', 10, 'efs', 3], ['malp', 2, 'ef', 3], ['Hodne', 20, 'ef', 3], ['stredne', 10, 'ef', 3], ['malp', 2, 'ef', 3],['Hodne', 20, 'ef', 3], ['stredne', 10, 'ef', 3], ['malp', 2, 'ef', 3]])
# # X = df.iloc[:, 0:2]
# # y = df.iloc[:, 3:4]



# import categorical_embedder as ce
# from sklearn.model_selection import train_test_split
# df = pd.read_csv('tests/data.csv')
# X = df.drop(['employee_id', 'is_promoted'], axis=1)
# y = df['is_promoted']
# embedding_info = ce.get_embedding_info(X)
# X_encoded, encoders = ce.get_label_encoded_data(X)
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y)
# embeddings = ce.get_embeddings(X_train, y_train, categorical_embedding_info=embedding_info, 
#                             is_classification=True, epochs=100,batch_size=256)

# # embedding_info = ce.get_embedding_info(X)
# # X_encoded,encoders = ce.get_label_encoded_data(X)

# # embeddings = ce.get_embeddings(X, y, categorical_embedding_info=embedding_info, 
# #                             is_classification=True, epochs=100, batch_size=256)
# # embeddings_df = ce.get_embeddings_in_dataframe(embeddings, encoders)

