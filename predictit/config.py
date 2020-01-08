""" All config for main file. Most of values are boolean 1 or 0. You can setup here data source, models to use,
whether you want find optimal parameters etc. Some setting can be inserted as function parameters, then it has higher priority.
All values are commented and easy to understand.

"""

from . import models
from .models.sklearn_regression import get_regressors

# Data source type
data_source = 'csv'  # 'csv' or 'sql' or 'test'

# CSV adress (first row is column names - first column is date column)

# CSV in test_data are gitignored
csv_from_test_data_name = 'daily-minimum-temperatures.csv'  # '5000 Sales Records.csv'  # Csv located in module folder. Override next full adress if not 0 or empty string
csv_adress = r'test_data\daily-minimum-temperatures.csv'  # Full CSV adress with suffix

# If SQL, sql credentials
server = '.'
database = 'FK'  # Database name

freqs = []  # Interval for predictions 'M' - months, 'D' - Days, 'H' - Hours

plot = 1  # If 1, plot interactive graph
save_plot = 0
save_plot_adress = ''  # Path where to save the plot (String), if empty string or 0 - saved to desktop

# TODO if empty, then 0
predicted_columns = 1  # Name of predicted column or it's index - string or int or list of columns

date_index = 0  # Index of dataframe or it's name. Can be empty, then index 0 or new if no date column.
predicts = 7  # Number of predicted values - 7 by default
result_type = 'best'  # If 'all', return all results not only the best one
return_all = 0  # If 1, return more models (config.compareit) results sorted by how efficient they are. If 0, return only the best one
datalength = 100  # The length of the data used for prediction

data_transform = None  # 'difference' or None - Transform the data on differences between two neighbor values

# Evaulation is repeated couple of times on shifted data for ensuring results (remove accidentaly good results)
debug = 1  # Debug - print all results and all the errors on the way
repeatit = 2  # How many times is computation repeated
other_columns = 0  # If 0, only predicted column will be used for making predictions.
lengths = 0  # Compute on various length of data (1, 1/2, 1/4...). Automatically choose the best length. If 0, use only full length.
confidence = 0.8  # Area of confidence in result plot (grey area where we suppose values) - Bigger value, narrower area - maximum 1
remove_outliers = 0  # Remove extraordinary values. Value is threshold for ignored values. Value means how many times standard deviation from the average threshold is far
standardize = 'standardize'  # one from 'standardize', '-11', '01', 'robust'
criterion = 'mape'  # 'mape' or 'rmse'
compareit = 10  # How many models will evaluated (and returned in 'result_type' = 'all') and also compared in final plot. 0 if only the best one.
last_row = 0  # If 0, erase last non-empty row
analyzeit = 0  # Analyze input data - Statistical distribution, autocorrelation, seasonal decomposition etc.
correlation_threshold = 0.5  # If evaluating from more collumns, use only collumns that are correlated. From 0 (All columns included) to 1 (only column itself)
optimizeit = 0  # Find optimal parameters of models
optimizeit_details = 2  # 1 print best parameters of models, 2 print every new best parameters achieved, 3 prints all results
optimizeit_limit = 0.01  # How many seconds can take one model optimization
optimizeit_final = 0  # If optimize paratmeters of the best model
optimizeit_final_limit = 0.1  # If optimize paratmeters of the best model
plotallmodels = 0  # Plot all models (recommended only in interactive jupyter window)
already_trained = 0  # Computationaly hard models (LSTM) load from disk
#tensorflowit = 0  # Whether use computationally hard models (slow even just on library load)

piclkeit = 1  # Wheter save serialized !test! data on disk, that can speed up loading
from_pickled = 1

database_deploy = 0  # Whether save the predictions to database
# TODO
#standardizeit = 0  # Standardize data from -1 to 1
#normalizeit = 0  # Normalizuje data to standard deviation 1 and average 0

#################################
### Models, that will be used ###
#################################
# Verbose documentation is in models module (__init__.py) and it's files
# Ctrl and Click on import name in main, or right click and go to definition


# If editting or adding new models, name of the models have to be the same as in models module
used_models = {

            "AR (Autoregression)": models.ar,
            "ARMA": models.arma,
            "ARIMA (Autoregression integrated moving average)": models.arima,
            "SARIMAX (Seasonal ARIMA)": models.sarima,

            "Autoregressive Linear neural unit": models.autoreg_LNU,
            "Linear neural unit with weigths predict": models.autoreg_LNU_withwpred,
            "Conjugate gradient": models.cg,


            #"LSTM": models.lstm,
            #"Bidirectional LSTM": models.lstm_bidirectional,
            #"LSTM batch": models.lstm_batch,


            "Sklearn regression": models.regression,
            "Bayes ridge regression": models.regression,
            "Hubber regression": models.regression,

            "Extreme learning machine": models.regression,
            "Gen Extreme learning machine": models.regression,

            "Compare with average": models.compare_with_average

}


## If you want to test one module and don't want to comment and uncomment one after one
# used_models = {
#     "AR (Autoregression)": models.ar,
#     "ARMA": models.arma,
#         "Compare with average": predictit.models.compare_with_average
#     "SARIMAX (Seasonal ARIMA)": models.sarima,
# }


# How many lags will be used (It is order of arima and also  number of neurons in some neural nets)
n_steps_in = 10
output_shape = 'one_step'  # 'batch' or 'one_step'
saveit = 0  # Save computationaly hard models (LSTM) on disk

models_parameters = {

        #TODO
        "ETS": {"plot": 0},


        "AR (Autoregression)": {"plot": 0, 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        "ARMA": {"plot": 0, "p": 3, "q": 0, 'method': 'mle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs', 'forecast_type': 'in_sample'},
        "ARIMA (Autoregression integrated moving average)": {"p": 3, "d": 0, "q": 0, "plot": 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm', 'forecast_type': 'out_of_sample'},
        "SARIMAX (Seasonal ARIMA)": {"plot": 0, "p": 4, "d": 0, "q": 0, "pp": 1, "dd": 0, "qq": 0, "season": 12, "method": "lbfgs", "trend": 'nc', "enforce_invertibility": False, "enforce_stationarity": False, "verbose": 0},


       # "ARIMA (Autoregression integrated moving average)": {"p": [1, maxorder], "d": [0,1], "q": order, 'method': ['css-mle', 'mle', 'css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},


        "Autoregressive Linear neural unit": {"plot": 0, "lags": n_steps_in, "mi": 1, "minormit": 0, "tlumenimi": 1},
        "Linear neural unit with weigths predict": {"plot": 0, "lags": n_steps_in, "mi": 1, "minormit": 0, "tlumenimi": 1},
        "Conjugate gradient": {"n_steps_in": n_steps_in, "epochs": 5, "constant": 1, "other_columns_lenght": None},


        # TODO finish lstm
        "LSTM": {"n_steps_in": n_steps_in, "save": saveit, "already_trained": 0, "epochs": 70, "units": 50, "optimizer": 'adam', "loss": 'mse', "verbose": 1, "activation": 'relu', "timedistributed": 0, "metrics": ['mape']},
        "LSTM batch": {"n_steps_in": n_steps_in, "n_features": 1, "epochs": 70, "units": 50, "optimizer": 'adam', "loss": 'mse', "verbose": 1, 'dropout': 0, "activation": 'relu'},
        "Bidirectional LSTM": {"n_steps_in": n_steps_in, "epochs": 70, "units": 50, "optimizer": 'adam', "loss": 'mse', "verbose": 0},

        "Sklearn regression": {"regressor": 'linear', "n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alpha": 0.0001, "n_iter": 100, "epsilon": 1.35, "alphas": [0.1, 0.5, 1], "gcv_mode": 'auto', "solver": 'auto'},
        "Bayes ridge regression": {"n_steps_in": n_steps_in, "regressor": 'bayesianridge', "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "n_iter": 300, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6},
        "Hubber regression": {"regressor": 'huber', "n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "epsilon": 1.35, "alpha": 0.0001},

        "Extreme learning machine": {"regressor": 'elm', "n_steps_in": n_steps_in, "output_shape": 'one_step', "other_columns_lenght": None, "constant": None, "n_hidden": 20, "alpha": 0.3, "rbf_width": 0, "activation_func": 'tanh'},
        "Gen Extreme learning machine": {"regressor": 'elm_gen', "n_steps_in": n_steps_in, "output_shape": 'one_step', "other_columns_lenght": None, "constant": None, "alpha": 0.5}
}

########################################
### Models parameters optimalisation ###
########################################
# Find best parameters for prediction
# Example how it works. Function predict_with_my_model(param1, param2)
# If param1 limits are [0, 10], and fragments = 5, it will be evaluated for [0, 2, 4, 6, 8, 10]
# Then it finds best value and make new interval that is again divided in 5 parts...
# This is done as many times as iteration value is
fragments = 4
iterations = 2

# This is for final optimalisation of the best model, not for all models
fragments_final = 2 * fragments
iterations_final = 2 * iterations

# Threshold values
# If you need integers, type just number, if you need float, type dot (e.g. 2.0)

# This boundaries repeat across models
steps = [2, 200]
alpha = [0.0, 1.0]
epochs = [2, 100]
units = [1, 100]
order = [0, 5]

maxorder = 6

regressors = get_regressors()

# !! Every parameters here have to be in models_parameters, or error
# Some models can be very computationaly hard - use optimizeit_limit or already_trained!
models_parameters_limits = {
        "AR (Autoregression)": {"ic": ['aic', 'bic', 'hqic', 't-stat'], "trend": ['c', 'nc'], "solver": ['bfgs', 'newton', 'nm', 'cg']},

        "ARMA": {"p": [1, maxorder], "q": order, 'method': ['css-mle', 'mle', 'css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'], 'forecast_type': ['in_sample', 'out_of_sample']},
        #"ARIMA (Autoregression integrated moving average)": {"p": [1, maxorder], "d": [0,1], "q": order, 'method': ['css-mle', 'mle', 'css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},
        "ARIMA (Autoregression integrated moving average)": {"p": [1, maxorder], "d": [0, 1], "q": order, 'method': ['css'], 'trend': ['c', 'nc'], 'solver': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg'], 'forecast_type': ['in_sample', 'out_of_sample']},
        "SARIMAX (Seasonal ARIMA)": {"p": [1, maxorder], "d": order, "q": order, "pp": order, "dd": order, "qq": order, "season": order, "method": ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'], "trend": ['n', 'c', 't', 'ct'], "enforce_invertibility": [True, False], "enforce_stationarity": [True, False], 'forecast_type': ['in_sample', 'out_of_sample']},

        "Autoregressive Linear neural unit": {"lags": steps, "mi": [1.0, 10.0], "minormit": [0, 1], "tlumenimi": [0.0, 100.0]},
        "Linear neural unit with weigths predict": {"lags": steps, "mi": [1.0, 10.0], "minormit": [0, 1], "tlumenimi": [0.0, 100.0]},
        "Conjugate gradient": {"n_steps_in": steps, "epochs": epochs, "constant": [None, 1], "other_columns_lenght": [None, steps[1]]},


        "Sklearn regression": {"n_steps_in": steps, "regressor": regressors, "output_shape": ['batch', 'one_step'], "other_columns_lenght": [None, steps[1]], "constant": [None, 1], "alpha": alpha, "n_iter": [100, 500], "epsilon": [1.01, 5.0], "alphas": [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]], "gcv_mode": ['auto', 'svd', 'eigen'], "solver": ['auto', 'svd', 'eigen']},
        "Bayes Ridge Regression": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "alpha_1": [0.1e-6, 3e-6], "alpha_2": [0.1e-6, 3e-6], "lambda_1": [0.1e-6, 3e-6], "lambda_2": [0.1e-7, 3e-6]},
        "Hubber regression": {"n_steps_in": steps, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "epsilon": [1.01, 5.0], "alpha": alpha},

        "Extreme learning machine": {"n_steps_in": steps, "n_hidden": [2, 300], "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]], "alpha": alpha, "rbf_width": [0.0, 10.0], "activation_func": ['tanh', 'sine', 'tribas', 'inv_tribase', 'sigmoid', 'hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric']},
        "Gen Extreme learning machine": {"n_steps_in": steps, "alpha": alpha, "output_shape": ['batch', 'one_step'], "constant": [None, 1], "other_columns_lenght": [None, steps[1]]}

}
