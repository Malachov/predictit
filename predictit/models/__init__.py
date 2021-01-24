# -*- coding: utf-8 -*-
""" Subpackage include models from libraries such as Tensorflow, Statsmodels, Sklearn and some own simple models.
Various models are included autoregressive model, ARIMA, LSTM, various regressions such as linear or ridge
and finally some neural units and conjugate gradient. You can use compare_models function to compare models on test data.
You can see results of all models in interactive plot.

2 main functions in all models

    - train (create model - slow to compute)
    - predict (use model - fast)

Some models contain some other functions, mostly for optimization reasons.

All particular models functions are described in docstrings in it's modules.

Models are usually used from `predictit.main`, but it can be use separately as well.

Examples:

    >>> data = mdp.generatedata.gen_sin(1000)
    >>> test = data[-7:]
    >>> data = data[: -7]
    >>> data = mdp.preprocessing.data_consolidation(data)
    >>> (X, y), x_input, _ = mdp.inputs.create_inputs(data.values, 'batch', input_type_params={'n_steps_in': 6})  # First tuple, because some models use raw data, e.g. [1, 2, 3...]

    >>> trained_model = predictit.models.sklearn_regression.train((X, y), regressor='bayesianridge')
    >>> predictions_one_model = predictit.models.sklearn_regression.predict(x_input, trained_model, predicts=7)

    >>> predictions_one_model_error = predictit.evaluate_predictions.compare_predicted_to_test(predictions_one_model, test, error_criterion='mape')  # , plot=1

"""

from . import models_functions

# Custom models
from . import autoreg_LNU
from . import conjugate_gradient
from . import regression
from . import levenberg_marquardt


# Statsmodels
from . import statsmodels_autoregressive

# Scikit
from . import sklearn_regression

# For comparing results with just an average values
from . import compare_with_average

from . import tensorflow

__all__ = ['models_functions', 'autoreg_LNU', 'conjugate_gradient', 'regression', 'levenberg_marquardt',
           'statsmodels_autoregressive', 'sklearn_regression', 'compare_with_average', 'tensorflow']


models_assignment = {

    **{model_name: statsmodels_autoregressive for model_name in [
        'AR (Autoregression)', 'ARMA', 'ARIMA (Autoregression integrated moving average)', 'autoreg',
        'SARIMAX (Seasonal ARIMA)'
    ]},

    **{model_name: autoreg_LNU for model_name in [
        'Autoregressive Linear neural unit', 'Autoregressive Linear neural unit normalized', 'Linear neural unit with weights predict']},

    **{model_name: regression for model_name in [
        'Regression', 'Ridge regression']},

    'Levenberg-Marquardt': levenberg_marquardt,

    'Conjugate gradient': conjugate_gradient,

    'Tensorflow LSTM': tensorflow,
    'Tensorflow MLP': tensorflow,

    **{model_name: sklearn_regression for model_name in [
        'Sklearn regression', 'Bayes ridge regression', 'Passive aggressive regression', 'Gradient boosting',
        'KNeighbors regression', 'Decision tree regression', 'Hubber regression',
        'Bagging regression', 'Stochastic gradient regression', 'Extreme learning machine', 'Gen Extreme learning machine', 'Extra trees regression', 'Random forest regression',

        'Sklearn regression one column one step', 'Bayes ridge regression one column one step', 'Decision tree regression one column one step',
        'Hubber regression one column one step',
    ]},

    'Compare with average': compare_with_average
}
