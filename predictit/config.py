''' All config for main file. Most of values are boolean 1 or 0. You can setup data source, models to use,
whether you want find optimal parameters etc. Some setting can be inserted as function parameters, then it has higher priority.
All values are commented and easy to understand.

If you downloaded the script, edit, save and run function from main, if you use library, import config and edit values...
Examples:

    >>> import predictit
    >>> from predictit.config import config

    >>> config.update({
    >>>     'data_source': 'csv',
    >>>     'csv_full_path': 'my/path/to.csv',
    >>>     'plot': 1,
    >>>     'debug': 1,
    >>>     'used_models': {
    >>>            'AR (Autoregression)': predictit.models.ar,
    >>>            'Autoregressive Linear neural unit': predictit.models.autoreg_LNU,
    >>>            'Sklearn regression': predictit.models.regression,
    >>>     }
    >>> }

    >>> predictions = predictit.main.predict()

To see all the possible values, use

    >>> predictit.config.print_config()

'''

import predictit


config = {

    # Edit presets as you need but beware - it will overwrite normal config. Default presets edit for example
    # number of used models, number of lengths, n_steps_in.

    'used_function': 'predict',  # If running main.py script, execute function. Choices: 'predict', 'predict_multiple' or 'compare_models'. If import as library, ignore - just use function.

    'use_config_preset': 'none',  # 'fast', 'normal' or 'none'. Edit some selected config, other remains the same, check config_presets.py file.

    # Input settings

    'data': None,  # Use numpy array, or pandas dataframe. This will overwrite data_source. If you use csv, set up to None. Data shape is (n_samples, n_feature)
                   # - that means rows are samples and columns are features.
    'data_all': None,  # Just for compare_models function. Dictionary of data names and list of it's values and predicted columns or list of data parts.
    # data_all examples: {'data_1': [my_dataframe, 'column_name']} or [my_data[-2000:], my_data[-1000:]] and 'predicted_column' from config as column name.

    'data_source': 'test',  # Data source. ('csv', 'txt', 'sql' or 'test') - If sql, you have to edit the query to fit your database.

    'csv_full_path': r'',  # Full CSV path with suffix.
    'csv_test_data_relative_path': '',  # CSV name with suffix in test_data folder (5000 Sales Records.csv or daily-minimum-temperatures.csv)
    # !!! Turn it off if not testing to not break csv full path !!! Test data are gitignored... use your own.

    'predicted_column': '',  # Name of predicted column (for dataframe data) or it's index - string or int.
    'predicted_columns': [],  # For predict_multiple function only! List of names of predicted columns or it's indexes. If 'data' is dataframe or numpy, then you can use '*' to predict all the columns.

    'freq': 0,  # Interval for predictions 'M' - months, 'D' - Days, 'H' - Hours.
    'freqs': [],  # For predict_multiple function only! List of intervals of predictions 'M' - months, 'D' - Days, 'H' - Hours. Default use [].

    'datetime_index': '',  # Index of dataframe datetime column or it's name if it has datetime column. If there already is index in input data, it will be used automatically. Data are sorted by time.

    'datalength': 0,  # The length of the data used for prediction (after resampling). If 0, than full length.
    'max_imported_length': 0,  # Max length of imported samples (before resampling). If 0, than full length.

    # Output settings

    'predicts': 7,  # Number of predicted values - 7 by default.

    'mode': 'predict',  # If 'validate', put apart last few ('predicts' + 'validation_gap') values and evaluate on test data that was not in train set. Do not setup - use compare_models function, it will use it automattically.
    'validation_gap': 10,  # If 'mode' == 'validate', then describe how many samples are between train and test set. The bigger gap is, the bigger knowledge generalization is necesssary.
    'return_type': 'best',  # 'best', 'all', 'dict', 'models_error_criterion', 'results_dataframe' or 'detailed_results_dictionary'. 'best' return array of predictions, 'all' return more models
                            # results sorted by how efficient they are. 'results_dataframe' return results and models names in columns. 'detailed_results_dictionary' is used for GUI
                            # and return results as best result, all results, string div with plot and more. 'models_error_criterion' returns MAPE,
                            # RMSE (based on config) or dynamic warping criterion of all models in array.
    'debug': 1,  # Debug - If 1, print all results and all the warnings and errors on the way, if 2, stop on first warning, if -1 do not print anything, just return result.

    'plot': 1,  # If 1, plot interactive graph.
    'plot_type': 'plotly',  # 'plotly' (interactive) or matplotlib.
    'show_plot': 1,  # Whether display plot or not. If in jupyter dislplay in jupyter, else in default browser.
    'save_plot': 0,  # (bool) - Html if plotly type, png if matplotlibtype.
    'save_plot_path': '',  # Path where to save the plot (String), if empty string or 0 - saved to desktop.
    'plot_name': 'Predictions',
    'plot_history_length': 100,  # How many historic values will be plotted
    'plot_number_of_models': 12,  # confiNumber of plotted models. If none or 0, than all models will be evaluated. 1 if only the best one.

    'print': 1,  # Turn on and off all printed details at once.
    'print_table': 1,  # Whether print table with models errors and time to compute.
    'print_number_of_models': None,  # How many models will be evaluated and printed in results table. If none or 0, than all models will be evaluated. 1 if only the best one.
    'print_time_table': 1,  # Whether print table with models errors and time to compute.
    'print_result': 1,  # Whether print best model results and model details.
    'print_detailed_result': 0,  # Whether print detailed results.

    # Prediction parameters settings

    'multiprocessing': 'process',  # 'pool' or 'process' or 0
    'default_n_steps_in': 30,  # How many lagged values are in vector input to model.
    'other_columns': 1,  # If use other columns. Bool.
    'default_other_columns_length': 2,  # Other columns vector length used for predictions. If None, lengths same as predicted columnd. If 0, other columns are not used for prediction.
    'remove_nans': 'any_columns',  # 'any_columns' will remove all the columns where is at least one nan column.
    'dtype': 'float32',  # Main dtype used in prediction. 'float32' or 'float64'.
    'repeatit': 50,  # How many times is computation repeated for error criterion evaluation.
    'lengths': 0,  # If 1, compute on various length of data (1, 1/2, 1/4...). Automatically choose the best length. If 0, use only full length.

    'data_transform': 0,  # 'difference' or 0 - Transform the data on differences between two neighbor values.
    'analyzeit': 0,  # If 1, analyze original data, if 2 analyze preprocessed data, if 3, then both. Statistical distribution, autocorrelation, seasonal decomposition etc.
    'analyze_seasonal_decompose': {'period': 365, 'model': 'additive'},  # Parameters for seasonal decompose in analyze. Find if there are periodically repeating patterns in data.
    'standardize': 'standardize',  # One from None, 'standardize', '-11', '01', 'robust'.

    'optimizeit': 0,  # Find optimal parameters of models.
    'optimizeit_details': 2,  # 1 print best parameters of models, 2 print every new best parameters achieved, 3 prints all results.
    'optimizeit_limit': 10,  # How many seconds can take one model optimization.

    'smooth': (11, 2),  # Smoothing data with Savitzky-Golay filter. First argument is window and second is polynomial order. If 0, not smoothing.
    'power_transformed': 0,  # Whether transform results to have same standard deviation and mean as train data. 0 no power transform, 1 on output and 2 on output and before evaluating.
    'remove_outliers': 0,  # Remove extraordinary values. Value is threshold for ignored values. Value means how many times standard deviation from the average threshold is far (have to be > 3, else delete most of data). If predicting anomalies, use 0.
    'error_criterion': 'mape',  # 'mape' or 'rmse' or 'dtw' (dynamic time warping).
    'last_row': 0,  # If 0, erase last row (for example in day frequency, day is not complete yet).
    'correlation_threshold': 0.5,  # If evaluating from more collumns, use only collumns that are correlated. From 0 (All columns included) to 1 (only column itself).Z
    'confidence_interval': 0.6,  # Area of confidence in result plot (grey area where we suppose values) - Bigger value, narrower area - maximum 1. If 0 - not plotted.
    'already_trained': 0,  # Computationaly hard models (LSTM) load from disk.
    #'plotallmodels': 0,  # Plot all models (recommended only in interactive jupyter window).


    #################################
    ### Models, that will be used ###
    #################################
    # Verbose documentation is in models module (__init__.py) and it's files.
    # Ctrl and Click on import name in main, or right click and go to definition.

    # Two asterisks means bulk edit values

    # Extra trees regression and tensorflow are turned off by default, because too timeconsuming.
    # If editting or adding new models, name of the models have to be the same as in models module.
    # If using presets - overwriten.

    # Comment out models you don't want to use.
    # Do not comment out input_types, models_input or models_parameters.
    'used_models': {

        **{model_name: predictit.models.statsmodels_autoregressive for model_name in [
            'AR (Autoregression)', 'ARMA', 'ARIMA (Autoregression integrated moving average)']},  # 'SARIMAX (Seasonal ARIMA)'

        **{model_name: predictit.models.autoreg_LNU for model_name in [
            'Autoregressive Linear neural unit', 'Autoregressive Linear neural unit normalized']},  # , 'Linear neural unit with weigths predict'

        'Conjugate gradient': predictit.models.conjugate_gradient,

        # 'tensorflow_lstm': predictit.models.tensorflow,
        # 'tensorflow_mlp': predictit.models.tensorflow,

        **{model_name: predictit.models.sklearn_regression for model_name in [
            'Sklearn regression', 'Bayes ridge regression', 'Passive aggressive regression', 'Gradient boosting',
            'KNeighbors regression', 'Decision tree regression',
            # 'Bagging regression', 'Hubber regression', 'Stochastic gradient regression', 'Extreme learning machine', 'Gen Extreme learning machine',  'Extra trees regression', 'Random forest regression'
        ]},

        'Compare with average': predictit.models.compare_with_average
    },
}


# Config is made by parts, because some values are reused...
def update_references_1():
    config.update({
        'input_types': {

            # data
            # data_one_column

            'one_step_constant': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': 1, 'default_other_columns_length': config['default_other_columns_length'], 'constant': 1},
            'one_step': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': 1, 'default_other_columns_length': config['default_other_columns_length'], 'constant': 0},
            'batch_constant': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': config['predicts'], 'default_other_columns_length': config['default_other_columns_length'], 'constant': 1},
            'batch': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': config['predicts'], 'default_other_columns_length': config['default_other_columns_length'], 'constant': 0},
            'one_in_one_out_constant': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': 1, 'constant': 1},
            'one_in_one_out': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': 1, 'constant': 0},
            'not_serialized': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': config['predicts'], 'constant': 0, 'serialize_columns': 0},
        },
    })


config.update({
    # Don not forget inputs data and data_one_column, not only input types...
    'models_input': {

        **{model_name: 'data_one_column' for model_name in [
            'AR (Autoregression)', 'ARMA', 'ARIMA (Autoregression integrated moving average)', 'SARIMAX (Seasonal ARIMA)']},

        **{model_name: 'one_in_one_out_constant' for model_name in [
            'Autoregressive Linear neural unit', 'Autoregressive Linear neural unit normalized', 'Linear neural unit with weigths predict', 'Conjugate gradient']},

        **{model_name: 'batch' for model_name in [
            'Sklearn regression', 'Bayes ridge regression', 'Hubber regression', 'Extra trees regression', 'Decision tree regression', 'KNeighbors regression', 'Random forest regression',
            'Bagging regression', 'Stochastic gradient regression', 'Passive aggressive regression', 'Extreme learning machine', 'Gen Extreme learning machine', 'Gradient boosting']},

        'tensorflow_lstm': 'not_serialized',
        'tensorflow_mlp': 'one_step',

        'Compare with average': 'data_one_column'
    },

    # If using presets - overwriten.
    # If commented - default parameters will be used.
    'models_parameters': {

        'AR (Autoregression)': {'model': 'ar', 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        'ARMA': {'model': 'arma', 'p': 3, 'q': 0, 'method': 'mle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        'ARIMA (Autoregression integrated moving average)': {'model': 'arima', 'p': 3, 'd': 0, 'q': 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm'},
        # 'SARIMAX (Seasonal ARIMA)': {'p': 4, 'd': 0, 'q': 0, 'pp': 1, 'dd': 0, 'qq': 0, 'season': 12, 'method': 'lbfgs', 'trend': 'nc', 'enforce_invertibility': False, 'enforce_stationarity': False},

        'Autoregressive Linear neural unit': {'mi_multiple': 1, 'mi_linspace': (1e-5, 1e-4, 3), 'epochs': 10, 'w_predict': 0, 'minormit': 0},
        'Autoregressive Linear neural unit normalized': {'mi': 1, 'mi_multiple': 0, 'epochs': 10, 'w_predict': 0, 'minormit': 1},
        'Linear neural unit with weigths predict': {'mi_multiple': 1, 'mi_linspace': (1e-5, 1e-4, 3), 'epochs': 10, 'w_predict': 1, 'minormit': 0},
        'Conjugate gradient': {'epochs': 200},

        'tensorflow_lstm': {'layers': 'default', 'epochs': 200, 'already_trained': 0, 'save': 1, 'saved_model_path_string': 'stored_models', 'optimizer': 'adam', 'loss': 'mse', 'verbose': 0, 'metrics': 'accuracy', 'timedistributed': 0},
        'tensorflow_mlp': {'layers': [['dense', {'units': 32, 'activation': 'relu'}],
                                      ['dropout', {'rate': 0.1}],
                                      ['dense', {'units': 7, 'activation': 'relu'}]],
                           'epochs': 100, 'already_trained': 0, 'save': 1, 'saved_model_path_string': 'stored_models', 'optimizer': 'adam', 'loss': 'mse', 'verbose': 0, 'metrics': 'accuracy', 'timedistributed': 0},

        'Sklearn regression': {'regressor': 'linear', 'alpha': 0.0001, 'n_iter': 100, 'epsilon': 1.35, 'alphas': [0.1, 0.5, 1], 'gcv_mode': 'auto', 'solver': 'auto', 'alpha_1': 1.e-6,
                               'alpha_2': 1.e-6, 'lambda_1': 1.e-6, 'lambda_2': 1.e-6, 'n_hidden': 20, 'rbf_width': 0, 'activation_func': 'selu'},
        'Bayes ridge regression': {'regressor': 'bayesianridge', 'n_iter': 300, 'alpha_1': 1.e-6, 'alpha_2': 1.e-6, 'lambda_1': 1.e-6, 'lambda_2': 1.e-6},
        'Hubber regression': {'regressor': 'huber', 'epsilon': 1.35, 'alpha': 0.0001},
        'Extra trees regression': {'regressor': 'Extra trees'},
        'Decision tree regression': {'regressor': 'Decision tree'},
        'KNeighbors regression': {'regressor': 'KNeighbors'},
        'Random forest regression': {'regressor': 'Random forest'},
        'Bagging regression': {'regressor': 'Bagging'},
        'Stochastic gradient regression': {'regressor': 'Stochastic gradient'},

        'Extreme learning machine': {'regressor': 'elm', 'n_hidden': 50, 'alpha': 0.3, 'rbf_width': 0, 'activation_func': 'tanh'},
        'Gen Extreme learning machine': {'regressor': 'elm_gen', 'alpha': 0.5}
    },

    ########################################
    ### Models parameters optimizations ###
    ########################################
    # Find best parameters for prediction.
    # Example how it works. Function predict_with_my_model(param1, param2).
    # If param1 limits are [0, 10], and 'fragments': 5, it will be evaluated for [0, 2, 4, 6, 8, 10].
    # Then it finds best value and make new interval that is again divided in 5 parts...
    # This is done as many times as iteration value is.
    'fragments': 4,
    'iterations': 2,

    # Threshold values.
    # If you need integers, type just number, if you need float, type dot (e.g. 2.0).

    # This boundaries repeat across models.
    'alpha': [0.0, 1.0],
    'epochs': [1, 300],
    'units': [1, 100],
    'order': [0, 20],

    'maxorder': 20,

    'regressors': predictit.models.sklearn_regression.get_regressors()
})


# !! Every parameters here have to be in models_parameters, or error.
# Some models can be very computationaly hard - use optimizeit_limit or already_trained!
# If models here are commented, they are not optimized !
# You can optmimize as much parameters as you want - for example just one (much faster).
def update_references_2():
    config.update({
        'models_parameters_limits': {
            'AR (Autoregression)': {'ic': ['aic', 'bic', 'hqic', 't-stat'], 'trend': ['c', 'nc']},

            'ARMA': {'p': [1, config['maxorder']], 'q': config['order'], 'trend': ['c', 'nc']},
            # 'ARIMA (Autoregression integrated moving average)': {'p': [1, config['maxorder']], 'd': [0, 2], 'q': [0, 4], 'trend': ['c', 'nc']},
            # 'SARIMAX (Seasonal ARIMA)': {'p': [1, config['maxorder']], 'd': config['order'], 'q': config['order'], 'pp': config['order'], 'dd': config['order'], 'qq': config['order'],
            # 'season': config['order'], 'method': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'], 'trend': ['n', 'c', 't', 'ct'], 'enforce_invertibility': [True, False], 'enforce_stationarity': [True, False], 'forecast_type': ['in_sample', 'out_of_sample']},

            # 'Autoregressive Linear neural unit': {'mi': [1e-8, 10.0], 'minormit': [0, 1], 'damping': [0.0, 100.0]},
            # 'Linear neural unit with weigths predict': {'mi': [1e-8, 10.0], 'minormit': [0, 1], 'damping': [0.0, 100.0]},
            # 'Conjugate gradient': {'epochs': config['epochs']},

            ### 'tensorflow_lstm': {'loses':["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "logcosh",
            ### "kullback_leibler_divergence"], 'activations':['softmax', 'elu', 'selu', 'softplus', 'tanh', 'sigmoid', 'exponential', 'linear']},
            ### 'tensorflow_mlp': {'loses':["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "logcosh",
            ### "kullback_leibler_divergence"], 'activations':['softmax', 'elu', 'selu', 'softplus', 'tanh', 'sigmoid', 'exponential', 'linear']},

            # 'Sklearn regression': {'regressor': config['regressors']},  # 'alpha': config['alpha'], 'n_iter': [100, 500], 'epsilon': [1.01, 5.0], 'alphas': [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]], 'gcv_mode': ['auto', 'svd', 'eigen'], 'solver': ['auto', 'svd', 'eigen']},
            'Extra trees': {'n_estimators': [1., 500.]},
            'Bayes ridge regression': {'alpha_1': [0.1e-6, 3e-6], 'alpha_2': [0.1e-6, 3e-6], 'lambda_1': [0.1e-6, 3e-6], 'lambda_2': [0.1e-7, 3e-6]},
            'Hubber regression': {'epsilon': [1.01, 5.0], 'alpha': config['alpha']},

            # 'Extreme learning machine': {'n_hidden': [2, 300], 'alpha': config['alpha'], 'rbf_width': [0.0, 10.0], 'activation_func': ['tanh', 'sine', 'tribas', 'inv_tribase', 'sigmoid', 'hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric']},
            'Gen Extreme learning machine': {'alpha': config['alpha']}

        },
    })


config.update({
    'ignored_warnings': ["AR coefficients are not stationary.",
                         "numpy.ufunc size changed, may indicate binary incompatibility",
                         "invalid value encountered in sqrt",
                         "encountered in double_scalars",
                         "Inverting hessian failed",
                         "can't resolve package from",
                         "unclosed file <_io.TextIOWrapper name",
                         "Mean of empty slice",
                         "Check mle_retvals",
                         "overflow encountered in multiply",
                         "The default value of n_estimators will change",
                         "statsmodels.tsa.AR has been deprecated "],


    # If SQL, sql credentials
    'server': '.',
    'database': 'FK',  # Database name
    'database_deploy': 0,  # Whether save the predictions to database
})








###############
### Presets ###
###############

# Edit if you want, but it's not necessary from here - Mostly for GUI.

###!!! overwrite defined settings !!!###
presets = {
    'fast': {
        'optimizeit': 0, 'default_n_steps_in': config['default_n_steps_in'] + 3, 'repeatit': 20, 'lengths': 0, 'datalength': 1000,
        'other_columns': 0, 'data_transform': 0, 'remove_outliers': 0, 'analyzeit': 0, 'standardize': None,

        # If editting or adding new models, name of the models have to be the same as in models module
        'used_models': {
            'AR (Autoregression)': predictit.models.statsmodels_autoregressive,
            'Conjugate gradient': predictit.models.conjugate_gradient,
            'Sklearn regression': predictit.models.sklearn_regression,
            'Compare with average': predictit.models.compare_with_average
        },
    }
}


presets['normal'] = {
    'optimizeit': 0, 'default_n_steps_in': 15, 'repeatit': 50, 'lengths': 0, 'datalength': 3000,
    'other_columns': 1, 'remove_outliers': 0, 'analyzeit': 0, 'standardize': None,

    'used_models': {

        **{model_name: predictit.models.statsmodels_autoregressive for model_name in [
            'AR (Autoregression)', 'ARIMA (Autoregression integrated moving average)']},  # 'SARIMAX (Seasonal ARIMA)'

        'Autoregressive Linear neural unit': predictit.models.autoreg_LNU,
        # 'Linear neural unit with weigths predict': predictit.models.autoreg_LNU,
        'Conjugate gradient': predictit.models.conjugate_gradient,

        # # 'tensorflow_lstm': predictit.models.tensorflow,
        # # 'tensorflow_mlp': predictit.models.tensorflow,

        **{model_name: predictit.models.sklearn_regression for model_name in [
            'Sklearn regression', 'Bayes ridge regression', 'Hubber regression', 'Decision tree regression',
            'KNeighbors regression', 'Random forest regression', 'Bagging regression',
            'Passive aggressive regression', 'Extreme learning machine', 'Gen Extreme learning machine', 'Gradient boosting']},

        'Compare with average': predictit.models.compare_with_average
    },
}


update_references_1()
update_references_2()

config_check_set = set(config.keys())


def print_config():
    import pygments

    from pathlib import Path
    with open(Path(__file__).resolve(), "r") as f:
        print(pygments.highlight("\n".join(f.readlines()), pygments.lexers.python.PythonLexer(), pygments.formatters.Terminal256Formatter(style='friendly')))
