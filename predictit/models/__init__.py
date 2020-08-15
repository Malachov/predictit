# -*- coding: utf-8 -*-
""" Subpackage include models from libraries such as Tensorflow, Statsmodels, Sklearn and some own simple models.
Various models are included autoregressive model, ARIMA, LSTM, various regressions such as linear or ridge
and finally some neural units and conjugate gradient. You can use compare_models function to compare models on test data.
You can see results of all models in interactive plot.

All models are described in docstrings in it's modules.
"""

# Moje
from . import autoreg_LNU
from . import conjugate_gradient

# Statsmodels
from . import statsmodels_autoregressive

# Scikit
from . import sklearn_regression

# For comparing results with just an average values
from . import compare_with_average

from . import tensorflow

__all__ = ['autoreg_LNU', 'conjugate_gradient', 'statsmodels_autoregressive', 'sklearn_regression',
           'compare_with_average', 'tensorflow']


models_assignment = {

    **{model_name: statsmodels_autoregressive for model_name in [
        'AR (Autoregression)', 'ARMA', 'ARIMA (Autoregression integrated moving average)', 'autoreg',
        'SARIMAX (Seasonal ARIMA)'
    ]},

    **{model_name: autoreg_LNU for model_name in [
        'Autoregressive Linear neural unit', 'Autoregressive Linear neural unit normalized', 'Linear neural unit with weigths predict']},

    'Conjugate gradient': conjugate_gradient,

    'tensorflow_lstm': tensorflow,
    'tensorflow_mlp': tensorflow,

    **{model_name: sklearn_regression for model_name in [
        'Sklearn regression', 'Bayes ridge regression', 'Passive aggressive regression', 'Gradient boosting',
        'KNeighbors regression', 'Decision tree regression', 'Hubber regression',
        'Bagging regression', 'Stochastic gradient regression', 'Extreme learning machine', 'Gen Extreme learning machine', 'Extra trees regression', 'Random forest regression',

        'Sklearn regression one step', 'Bayes ridge regression one step', 'Decision tree regression one step',
        'Hubber regression one step',
    ]},

    'Compare with average': compare_with_average
}
