''' All config for main file. Most of values are boolean 1 or 0. You can setup data source, models to use,
whether you want find optimal parameters etc. Some setting can be inserted as function parameters, then it has higher priority.
All values are commented and easy to understand.

Examples:

    >>> config = {
    >>>     'data_source': 'csv'  # 'csv' or 'sql' or 'test'
    >>>     'plot': 1  # If 1, plot interactive graph
    >>>     'debug': 1  # Debug - print all results and all the errors on the way
    >>>     'used_models': {
    >>>            'AR (Autoregression)': models.ar,
    >>>            'Autoregressive Linear neural unit': models.autoreg_LNU,
    >>>            'Sklearn regression': models.regression,
    >>>             }
    >>> }

'''

from predictit import models
from predictit.models.sklearn_regression import get_regressors




import numpy as np
da = np.array([range(300), range(1000, 1300)])

config = {


'use_config_preset': 'none',  # 'fast', 'normal', 'optimize' or 'none'. Edit some selected config, other remains the same, check config_presets.py file.
                              # Edit presets as you need but beware - it will overwrite normal config. Default presets edim mainly number of used models, number of 


'used_function': 'predict',  # If running main.py script, execute function. Choices: 'predict', 'predict_multiple' or 'compare_models'. If import as library, ignored.
'data': da,  # Use numpy array, or pandas dataframe. This will overwrite data_source. If you use csv, set up to None.
'data_all': None,  # Just for compare_models function. Dictionary of data names and it's values
'data_source': 'test',  # Data source. ('csv' or 'sql' or 'test')

'csv_full_path': r'C:\Users\truton\ownCloud\Github\predictit_library\predictit\test_data\5000 Sales Records.csv',  # Full CSV path with suffix
'csv_test_data_relative_path': r'5000 Sales Records.csv',  # CSV name with suffix in test_data (5000 Sales Records.csv or daily-minimum-temperatures.csv) !! Turn off to not break csv full path !!

# TODO if empty, then 0
'predicted_column': 0,  # Name of predicted column (for dataframe data) or it's index - string or int
'predicted_columns': [],  # For predict_multiple function only! List of names of predicted columns or it's indexes

'return_type': 'dict',  # 'best', 'all', 'dict', or 'model_criterion'. Best return array of predictions, 'all' return more models (config.compareit) results sorted by how efficient they are. 'dict' return results as best result, all results, string div with plot and more. 'models_criterion' returns MAPE or RMSE (based on config) of all models in array.
'debug': 1,  # Debug - If 1, print all results and all the warnings and errors on the wignoreday, if 2, stop on all warnings.

'plot': 1,  # If 1, plot interactive graph
'plot_type': 'plotly',  # 'plotly' (interactive) or matplotlib(faster)
'show_plot': 1,
'save_plot': 0,  # (bool) - Html if plotly type, png if matplotlibtype
'save_plot_path': '',  # Path where to save the plot (String), if empty string or 0 - saved to desktop
'plot_name': 'Predictions',

'print_table': 1,  # Whether print table with models errors and time to compute
'print_time_table': 1,  # Whether print table with models errors and time to compute

'print_result': 1,  # Whether print best model resuls and model details

'freq': 0,  # Interval for predictions 'M' - months, 'D' - Days, 'H' - Hours.
'freqs': [],  # For predict_multiple function only! List of intervals of predictions 'M' - months, 'D' - Days, 'H' - Hours. Default use []

'date_index': 0,  # Index of dataframe or it's name. Can be empty, then index 0 or new if no date column.
'predicts': 7,  # Number of predicted values - 7 by default


### Next section to standardize is in updated if using presets
'optimizeit': 0,  # Find optimal parameters of models
'default_n_steps_in': 10,  # How many lagged values are in vector input to model
'other_columns_length': None,
'dtype': 'float32',
'repeatit': 4,  # How many times is computation repeated
'lengths': 1,  # Compute on various length of data (1, 1/2, 1/4...). Automatically choose the best length. If 0, use only full length.
'datalength': 1000,  # The length of the data used for prediction

'data_transform': 0,  # 'difference' or 0 - Transform the data on differences between two neighbor values,
'analyzeit': 0,  # Analyze input data - Statistical distribution, autocorrelation, seasonal decomposition etc.
'standardize': None,  # One from None, 'standardize', '-11', '01', 'robust'


'power_transformed': 0,  # Whether transform results to have same standard deviation and mean as train data. 0 no power transform, 1 on output and 2 on output and before evaluating.
'other_columns': 1,  # If 0, only predicted column will be used for making predictions.
'remove_outliers': 0,  # Remove extraordinary values. Value is threshold for ignored values. Value means how many times standard deviation from the average threshold is far
'criterion': 'mape',  # 'mape' or 'rmse' or 'dwt' (dynamic time warp)
'compareit': 10,  # How many models will be displayed in final plot. 0 if only the best one.
'last_row': 0,  # If 0, erase last non-empty row
'correlation_threshold': 0.5,  # If evaluating from more collumns, use only collumns that are correlated. From 0 (All columns included) to 1 (only column itself)
'optimizeit_details': 2,  # 1 print best parameters of models, 2 print every new best parameters achieved, 3 prints all results
'optimizeit_limit': 1,  # How many seconds can take one model optimization
'optimizeit_final': 0,  # If optimize paratmeters of the best model
'optimizeit_final_limit': 1,  # If optimize paratmeters of the best model
'confidence': 0.8,  # Area of confidence in result plot (grey area where we suppose values) - Bigger value, narrower area - maximum 1
'already_trained': 0,  # Computationaly hard models (LSTM) load from disk
#'plotallmodels': 0,  # Plot all models (recommended only in interactive jupyter window)


#################################
### Models, that will be used ###
#################################
# Verbose documentation is in models module (__init__.py) and it's files
# Ctrl and Click on import name in main, or right click and go to definition


# If editting or adding new models, name of the models have to be the same as in models module
# If using presets - overwriten

# Comment out models you don't want to use.
# !!! Do not comment out input_types, models_input or models_parameters or models_parameters_limits !!!
'used_models': {

            # 'AR (Autoregression)': models.statsmodels_autoregressive,
            # 'ARMA': models.statsmodels_autoregressive,
            # 'ARIMA (Autoregression integrated moving average)': models.statsmodels_autoregressive,
            # # # # 'SARIMAX (Seasonal ARIMA)': models.sarima,

            # 'Autoregressive Linear neural unit': models.autoreg_LNU,
            # 'Linear neural unit with weigths predict': models.autoreg_LNU,
            # 'Conjugate gradient': models.conjugate_gradient,

            #  'tensorflow': models.tensorflow,
            'tensorflow1': models.tensorflow,
            # 'tensorflow2': models.tensorflow,
            # 'tensorflow3': models.tensorflow,
            # 'tensorflow4': models.tensorflow,
            # 'tensorflow5': models.tensorflow,
            # 'tensorflow6': models.tensorflow,
            # 'tensorflow7': models.tensorflow,
            # 'tensorflow8': models.tensorflow,
            # 'tensorflow9': models.tensorflow,
            # 'tensorflow10': models.tensorflow,
            # 'tensorflow11': models.tensorflow,

            # 'Sklearn regression': models.sklearn_regression,
            # 'Bayes ridge regression': models.sklearn_regression,
            # 'Hubber regression': models.sklearn_regression,

            # 'Extreme learning machine': models.sklearn_regression,
            # 'Gen Extreme learning machine': models.sklearn_regression,

            # 'Compare with average': models.compare_with_average

},

## If you want to test one module and don't want to comment and uncomment one after one
# 'used_models': {
#     'AR (Autoregression)': models.ar,
# }

}

# Config is made by parts, because some values are reused...
config.update({
'input_types': {
    'one_step_constant': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': 1, 'other_columns_length': config['other_columns_length'], 'constant': 1},
    'one_step': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': 1, 'other_columns_length': config['other_columns_length'], 'constant': 0},
    'batch_constant': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': config['predicts'], 'other_columns_length': config['other_columns_length'], 'constant': 1},
    'batch': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': config['predicts'], 'other_columns_length': config['other_columns_length'], 'constant': 0},
    'one_in_one_out_constant': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': 1, 'constant': 1},
    'one_in_one_out': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': 1, 'constant': 0},
    'not_serialized': {'n_steps_in': config['default_n_steps_in'], 'n_steps_out': config['predicts'], 'constant': 0, 'serialize_columns': 0},
},

# Don not forget inputs data and data_one_column, not only input types...
'models_input': {

            'AR (Autoregression)': 'data_one_column',
            'ARMA': 'data_one_column',
            'ARIMA (Autoregression integrated moving average)': 'data_one_column',
            'SARIMAX (Seasonal ARIMA)': 'data_one_column',

            'Autoregressive Linear neural unit': 'one_step_constant',
            'Linear neural unit with weigths predict': 'one_step_constant',
            'Conjugate gradient': 'one_in_one_out_constant',

            'tensorflow': 'one_step',

            'tensorflow1': 'one_step',
            'tensorflow2': 'one_step',
            'tensorflow3': 'one_step',
            'tensorflow4': 'one_step',
            'tensorflow5': 'one_step',
            'tensorflow6': 'one_step',
            'tensorflow7': 'one_step',
            'tensorflow8': 'one_step',
            'tensorflow9': 'one_step',
            'tensorflow10': 'one_step',
            'tensorflow11': 'one_step',






            'Sklearn regression': 'batch',
            'Bayes ridge regression': 'one_step',
            'Hubber regression': 'one_step',


            'Extreme learning machine': 'one_step',
            'Gen Extreme learning machine': 'one_step_constant',

            'Compare with average': 'data'
},

# If using presets - overwriten
# If commented - default parameters will be used
'models_parameters': {

        'AR (Autoregression)': {'model': 'ar', 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        'ARMA': {'model': 'arma', 'p': 3, 'q': 0, 'method': 'mle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        'ARIMA (Autoregression integrated moving average)': {'model': 'arima', 'p': 3, 'd': 0, 'q': 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm'},
        'SARIMAX (Seasonal ARIMA)': {'p': 4, 'd': 0, 'q': 0, 'pp': 1, 'dd': 0, 'qq': 0, 'season': 12, 'method': 'lbfgs', 'trend': 'nc', 'enforce_invertibility': False, 'enforce_stationarity': False},

        'Autoregressive Linear neural unit': {'plot': 0, 'mi': 1, 'mi_multiple': 1, 'epochs': 20, 'w_predict': 0, 'minormit': 1, 'damping': 1},
        'Linear neural unit with weigths predict': {'plot': 0, 'mi': 1, 'minormit': 0, 'damping': 1},
        'Conjugate gradient': {'epochs': 5},

        # 'tensorflow': {'layers': 'default', 'epochs': 20, 'already_trained': 0, 'save': 1, 'saved_model_path_string': 'stored_models', 'optimizer': 'adam', 'loss': 'mse', 'verbose': 0, 'metrics': 'accuracy', 'timedistributed': 0},

        'tensorflow1': {'layers': [['lstm', {'units': 32, 'activation': 'relu'}]]},
        'tensorflow2': {'layers': [['lstm', {'units': 7, 'activation': 'relu', 'return_sequences':1}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}],
                                    ['lstm', {'units': 32, 'activation': 'relu'}]]},
        'tensorflow3': {'layers': [['lstm', {'units': 32, 'activation': 'relu', 'return_sequences':1}],
                                    ['dropout', {'rate': 0.1}],
                                    ['lstm', {'units': 7, 'activation': 'relu'}]]},
        'tensorflow4': {'layers': [['lstm', {'units': 7, 'activation': 'relu', 'return_sequences':1}],
                                    ['lstm', {'units': 32, 'activation': 'relu', 'return_sequences':1}],
                                    ['lstm', {'units': 7, 'activation': 'relu'}]]},
        'tensorflow5': {'layers': [['lstm', {'units': 64, 'activation': 'relu', 'return_sequences':1}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}],
                                    ['lstm', {'units': 32, 'activation': 'relu', 'return_sequences':1}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}],
                                    ['lstm', {'units': 256, 'activation': 'relu', 'return_sequences':1}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}],
                                    ['lstm', {'units': 256, 'activation': 'relu', 'return_sequences':1}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}],
                                    ['lstm', {'units': 64, 'activation': 'relu'}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}]]},
        'tensorflow6': {'layers': [['lstm', {'units': 64, 'activation': 'relu', 'return_sequences':1}],
                                    ['lstm', {'units': 32, 'activation': 'relu', 'return_sequences':1}],
                                    ['lstm', {'units': 256, 'activation': 'relu', 'return_sequences':1}],
                                    ['lstm', {'units': 256, 'activation': 'relu', 'return_sequences':1}],
                                    ['lstm', {'units': 64, 'activation': 'relu'}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}]]},
        'tensorflow7': {'layers': [['dense', {'units': 32, 'activation': 'relu'}],
                                    ['dropout', {'rate': 0.1}],
                                    ['dense', {'units': 7, 'activation': 'relu'}]]},
        'tensorflow8': {'layers': [['dense', {'units': 64, 'activation': 'relu'}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}],
                                    ['dense', {'units': 32, 'activation': 'relu'}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}],
                                    ['dense', {'units': 256, 'activation': 'relu'}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}],
                                    ['dense', {'units': 256, 'activation': 'relu'}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}],
                                    ['dense', {'units': 64, 'activation': 'relu'}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}]]},
        'tensorflow9': {'layers': [['lstm', {'units': 7, 'activation': 'relu'}],
                                    ['batchnormalization'],
                                    ['dropout', {'rate': 0.1}],
                                    ['dense', {'units': 32, 'activation': 'relu'}]]},
        'tensorflow10': {'layers': [['dense', {'units': 32, 'activation': 'relu'}],
                                    ['dropout', {'rate': 0.1}],
                                    ['lstm', {'units': 7, 'activation': 'relu'}]]},
        'tensorflow11': {'layers': [['dense', {'units': 7, 'activation': 'relu'}],
                                    ['lstm', {'units': 32, 'activation': 'relu', 'return_sequences':1}],
                                    ['lstm', {'units': 7, 'activation': 'relu'}]]},




        'Sklearn regression': {'regressor': 'linear', 'alpha': 0.0001, 'n_iter': 100, 'epsilon': 1.35, 'alphas': [0.1, 0.5, 1], 'gcv_mode': 'auto', 'solver': 'auto', 'alpha_1': 1.e-6, 'alpha_2': 1.e-6, 'lambda_1': 1.e-6, 'lambda_2': 1.e-6, 'n_hidden': 20, 'rbf_width': 0, 'activation_func': 'selu'},
        'Bayes ridge regression': {'regressor': 'bayesianridge', 'n_iter': 300, 'alpha_1': 1.e-6, 'alpha_2': 1.e-6, 'lambda_1': 1.e-6, 'lambda_2': 1.e-6},
        'Hubber regression': {'regressor': 'huber', 'epsilon': 1.35, 'alpha': 0.0001},

        'Extreme learning machine': {'regressor': 'elm', 'n_hidden': 50, 'alpha': 0.3, 'rbf_width': 0, 'activation_func': 'tanh'},
        'Gen Extreme learning machine': {'regressor': 'elm_gen', 'alpha': 0.5}
        },

########################################
### Models parameters optimizations ###
########################################
# Find best parameters for prediction
# Example how it works. Function predict_with_my_model(param1, param2)
# If param1 limits are [0, 10], and 'fragments': 5, it will be evaluated for [0, 2, 4, 6, 8, 10]
# Then it finds best value and make new interval that is again divided in 5 parts...
# This is done as many times as iteration value is
'fragments': 4,
'iterations': 2,

# This is for final optimization of the best model, not for all models
'fragments_final': 8,
'iterations_final': 3,

# Threshold values
# If you need integers, type just number, if you need float, type dot (e.g. 2.0)

# This boundaries repeat across models
'alpha': [0.0, 1.0],
'epochs': [2, 100],
'units': [1, 100],
'order': [0, 5],

'maxorder': 6,

'regressors': get_regressors()
})

# !! Every parameters here have to be in models_parameters, or error
# Some models can be very computationaly hard - use optimizeit_limit or already_trained!
# If models here are commented, they are not optimized !
# You can optmimize as much parameters as you want - for example just one (much faster)
config.update({
'models_parameters_limits': {
        'AR (Autoregression)': {'ic': ['aic', 'bic', 'hqic', 't-stat'], 'trend': ['c', 'nc'], 'solver': ['bfgs', 'newton', 'nm', 'cg']},

        # 'ARMA': {'p': [1, config['maxorder']], 'q': config['order'], 'method': ['css-mle', 'mle', 'css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'], 'forecast_type': ['in_sample', 'out_of_sample']},
        # #'ARIMA (Autoregression integrated moving average)': {'p': [1, config['maxorder']], 'd': [0,1], 'q': config['order'], 'method': ['css-mle', 'mle', 'css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},
        # 'ARIMA (Autoregression integrated moving average)': {'p': [1, config['maxorder']], 'd': [0, 1], 'q': config['order'], 'method': ['css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},
        # 'SARIMAX (Seasonal ARIMA)': {'p': [1, config['maxorder']], 'd': config['order'], 'q': config['order'], 'pp': config['order'], 'dd': config['order'], 'qq': config['order'], 'season': config['order'], 'method': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'], 'trend': ['n', 'c', 't', 'ct'], 'enforce_invertibility': [True, False], 'enforce_stationarity': [True, False], 'forecast_type': ['in_sample', 'out_of_sample']},

        # 'Autoregressive Linear neural unit': {'lags': [2, 200], 'mi': [1.0, 10.0], 'minormit': [0, 1], 'damping': [0.0, 100.0]},
        # 'Linear neural unit with weigths predict': {'lags': [2, 200], 'mi': [1.0, 10.0], 'minormit': [0, 1], 'damping': [0.0, 100.0]},
        # 'Conjugate gradient': {'epochs': config['epochs']},

        ### 'tensorflow': {loses = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "logcosh", "kullback_leibler_divergence"], activations = ['softmax', 'elu', 'selu', 'softplus', 'tanh', 'sigmoid', 'exponential', 'linear']}


        'Sklearn regression': {'regressor': config['regressors'], 'alpha': config['alpha'], 'n_iter': [100, 500], 'epsilon': [1.01, 5.0], 'alphas': [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]], 'gcv_mode': ['auto', 'svd', 'eigen'], 'solver': ['auto', 'svd', 'eigen']},
        # 'Bayes ridge regression': {'constant': [None, 1], 'alpha_1': [0.1e-6, 3e-6], 'alpha_2': [0.1e-6, 3e-6], 'lambda_1': [0.1e-6, 3e-6], 'lambda_2': [0.1e-7, 3e-6]},
        # 'Hubber regression': {'constant': [None, 1], 'epsilon': [1.01, 5.0], 'alpha': config['alpha']},

        # 'Extreme learning machine': {'n_hidden': [2, 300], 'alpha': config['alpha'], 'rbf_width': [0.0, 10.0], 'activation_func': ['tanh', 'sine', 'tribas', 'inv_tribase', 'sigmoid', 'hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric']},
        # 'Gen Extreme learning machine': {'alpha': config['alpha']}

},

'ignored_warnings': [ "AR coefficients are not stationary.",
            "numpy.ufunc size changed, may indicate binary incompatibility",
            "invalid value encountered in sqrt",
            "HessianInversionWarning",
            "can't resolve package from",
            "unclosed file <_io.TextIOWrapper name"],


# If SQL, sql credentials
'server': '.',
'database': 'FK',  # Database name
'database_deploy': 0,  # Whether save the predictions to database
})








###############
### Presets ###
###############

# Edit if you want, but it's not necessary from here - Mostly for GUI

###!!! overwrite defined settings !!!###
presets = {
    'fast': {

    'optimizeit': 0,  # Find optimal parameters of models
    'default_n_steps_in': 6,  # How many lagged values are in vector input to model
    'repeatit': 1,  # How many times is computation repeated
    'lengths': 0,  # Compute on various length of data (1, 1/2, 1/4...). Automatically choose the best length. If 0, use only full length.
    'datalength': 1000,  # The length of the data used for prediction
    'other_columns': 0,  # If 0, only predicted column will be used for making predictions.
    'data_transform': 0,  # 'difference' or 0 - Transform the data on differences between two neighbor values,
    'remove_outliers': 0,  # Remove extraordinary values. Value is threshold for ignored values. Value means how many times standard deviation from the average threshold is far
    'analyzeit': 0,  # Analyze input data - Statistical distribution, autocorrelation, seasonal decomposition etc.
    'standardize': None,  # One from 'standardize', '-11', '01', 'robust'
    'compareit': 10,  # How many models will be displayed in final plot. 0 if only the best one.

    # If editting or adding new models, name of the models have to be the same as in models module
    'used_models': {

                'AR (Autoregression)': models.statsmodels_autoregressive,
                'Conjugate gradient': models.conjugate_gradient,
                'Sklearn regression': models.sklearn_regression,
                'Compare with average': models.compare_with_average

    },

    'n_steps_in': 4,  # How many lagged values enter the model in one step
    'output_shape': 'batch'

    }
}

presets['fast']['models_parameters'] = {
        'AR (Autoregression)': {'plot': 0, 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        'Conjugate gradient': {'n_steps_in': presets['fast']['default_n_steps_in'], 'epochs': 5, 'constant': 1, 'other_columns_lenght': None},
        'Sklearn regression': {'regressor': 'linear', 'output_shape': presets['fast']['output_shape'], 'constant': None, 'alpha': 0.0001, 'n_iter': 100, 'epsilon': 1.35, 'alphas': [0.1, 0.5, 1], 'gcv_mode': 'auto', 'solver': 'auto'},
}


presets['normal'] = {

    'optimizeit': 0,  # Find optimal parameters of models
    'default_n_steps_in': 12,  # How many lagged values are in vector input to model
    'repeatit': 3,  # How many times is computation repeated
    'lengths': 1,  # Compute on various length of data (1, 1/2, 1/4...). Automatically choose the best length. If 0, use only full length.
    'datalength': 3000,  # The length of the data used for prediction
    'other_columns': 1,  # If 0, only predicted column will be used for making predictions.
    'remove_outliers': 0,  # Remove extraordinary values. Value is threshold for ignored values. Value means how many times standard deviation from the average threshold is far
    'analyzeit': 0,  # Analyze input data - Statistical distribution, autocorrelation, seasonal decomposition etc.
    'standardize': None,  # One from 'standardize', '-11', '01', 'robust'
    'output_shape': 'batch',


    # If editting or adding new models, name of the models have to be the same as in models module
    'used_models': {

                'AR (Autoregression)': models.statsmodels_autoregressive,
                # 'ARIMA (Autoregression integrated moving average)': models.statsmodels_autoregressive,
                # 'Autoregressive Linear neural unit': models.autoreg_LNU,
                # 'Conjugate gradient': models.conjugate_gradient,
                # 'Sklearn regression': models.sklearn_regression,
                # 'Hubber regression': models.sklearn_regression,
                # 'Extreme learning machine': models.sklearn_regression,
                # 'Compare with average': models.compare_with_average

    },
    'n_steps_in': 16,  # How many lagged values enter the model in one step
    }

presets['normal']['models_parameters'] = {
        'AR (Autoregression)': {'plot': 0, 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        # 'ARIMA (Autoregression integrated moving average)': {'p': 3, 'd': 0, 'q': 0, 'plot': 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm', 'forecast_type': 'out_of_sample'},
        # 'Autoregressive Linear neural unit': {'plot': 0, 'lags': presets['normal']['default_n_steps_in'], 'mi': 1, 'minormit': 0, 'damping': 1},
        # 'Conjugate gradient': {'n_steps_in': presets['normal']['default_n_steps_in'], 'epochs': 5, 'constant': 1, 'other_columns_lenght': None},
        # 'Sklearn regression': {'regressor': 'linear', 'alpha': 0.0001, 'n_iter': 100, 'epsilon': 1.35, 'alphas': [0.1, 0.5, 1], 'gcv_mode': 'auto', 'solver': 'auto'},
        # 'Hubber regression': {'regressor': 'huber', 'epsilon': 1.35, 'alpha': 0.0001},
        # 'Extreme learning machine': {'regressor': 'elm', 'n_hidden': 20, 'alpha': 0.3, 'rbf_width': 0, 'activation_func': 'tanh'},
}


presets['optimize'] = {

    'optimizeit': 1,  # Find optimal parameters of models
    'default_n_steps_in': 10,  # How many lagged values are in vector input to model
    'repeatit': 5,  # How many times is computation repeated
    'lengths': 1,  # Compute on various length of data (1, 1/2, 1/4...). Automatically choose the best length. If 0, use only full length.
    'datalength': 3000,  # The length of the data used for prediction
    'other_columns': 1,  # If 0, only predicted column will be used for making predictions.
    'output_shape': 'batch',
    # If editting or adding new models, name of the models have to be the same as in models module
    'used_models': {

                'AR (Autoregression)': models.statsmodels_autoregressive,
                # 'ARIMA (Autoregression integrated moving average)': models.statsmodels_autoregressive,
                # 'Autoregressive Linear neural unit': models.autoreg_LNU,
                # 'Conjugate gradient': models.conjugate_gradient,
                # 'Sklearn regression': models.sklearn_regression,
                # 'Hubber regression': models.sklearn_regression,
                # 'Extreme learning machine': models.sklearn_regression,
                # 'Compare with average': models.compare_with_average

    },

    'n_steps_in': 16,  # How many lagged values enter the model in one step

    }


presets['optimize']['models_parameters'] = {
        'AR (Autoregression)': {'plot': 0, 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        'ARIMA (Autoregression integrated moving average)': {'p': 3, 'd': 0, 'q': 0, 'plot': 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm', 'forecast_type': 'out_of_sample'},
        'Autoregressive Linear neural unit': {'plot': 0, 'lags': presets['optimize']['default_n_steps_in'], 'mi': 1, 'minormit': 0, 'damping': 1},
        'Conjugate gradient': {'n_steps_in': presets['optimize']['default_n_steps_in'], 'epochs': 5, 'constant': 1, 'other_columns_lenght': None},
        'Sklearn regression': {'regressor': 'linear', 'alpha': 0.0001, 'n_iter': 100, 'epsilon': 1.35, 'alphas': [0.1, 0.5, 1], 'gcv_mode': 'auto', 'solver': 'auto'},
        'Hubber regression': {'regressor': 'huber', 'epsilon': 1.35, 'alpha': 0.0001},
        'Extreme learning machine': {'regressor': 'elm', 'n_hidden': 20, 'alpha': 0.3, 'rbf_width': 0, 'activation_func': 'tanh'},
}
