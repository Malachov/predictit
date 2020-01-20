# -*- coding: utf-8 -*-
""" Subpackage include models from libraries such as Tensorflow, Statsmodels, Sklearn and some own simple models.
Various models are included autoregressive model, ARIMA, LSTM, various regressions such as linear or ridge
and finally some neural units and conjugate gradient. You can use compare_models function to compare models on test data.
You can see results of all models in interactive plot.

All models are described in docstrings in it's modules.
"""

# Moje
from .autoreg_LNU_withwpred import autoreg_LNU_withwpred
from .autoreg_LNU import autoreg_LNU
from .cg import cg

# Statsmodels
from .sm_ar import ar
from .sm_arma import arma
from .sm_arima import arima
from .sm_sarima import sarima

# Scikit
from .sklearn_regression import regression

# For comparing results with just an average values
from .compare_with_average import compare_with_average

# Tensorflow, Keras
tensorflowit = 0

if tensorflowit:
    from .tf_lstm_batch import lstm_batch
    from .tf_lstm_bidirectional import lstm_bidirectional
    from .tf_lstm_stacked_batch import lstm_stacked_batch
    from .tf_lstm_stacked import lstm_stacked
    from .tf_lstm import lstm
    from .tf_mlp_batch import mlp_batch
    from .tf_mlp import mlp


if tensorflowit:
    from keras import optimizers

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    lstm_optimizers = [sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam]

if tensorflowit == 0:
    lstm_optimizers = [0, 1]

loses = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "logcosh", "kullback_leibler_divergence"]
activations = ['softmax', 'elu', 'selu', 'softplus', 'tanh', 'sigmoid', 'exponential', 'linear']
