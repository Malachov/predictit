# -*- coding: utf-8 -*-
"""Všechny modely se všemi jejich parametry

'AR (Autoregression)': ar
-------------------

    Parametry (defaultně v závorce):
        predicts - počet predikovaných hodnot
        plot - pokud 1, tak vykreslí grafy

'Linear neural unit': autoregLNU
-----------------

    Parametry (defaultně v závorce):
        data - data která vstupují do modelu - list, array, nebo dataframe
        predicts(7) - počet predikovaných hodnot
        lags(100) - počet předchozích kroků, podle kterých bude model počítat
        plot(0) - pokud 1, tak vykreslí grafy
        mi(0.1) - Učící krok - bude vybrán nejlepší z desetinásobků, takže ladit od 0 do 1
        random(0) - pokud 1, váhy budou inicializovány náhodně (pro linear zbytečné)
        seed(0) - Seed náhodných inicializací - náhodná čísla pro konkrétní seed budou vždy stejná

'Linear neural unit with weigt predictions': autoregLNUwithwpred
---------------------

    Narozdíl od LNU, zde jsou predikovány i hodnoty jednotlivých vah
    Parametry (defaultně v závorce):
        data - data která vstupují do modelu - list, array, nebo dataframe
        predicts(7) - počet predikovaných hodnot
        lags(100) - počet předchozích kroků, podle kterých bude model počítat
        plot(0) - pokud 1, tak vykreslí grafy
        mi(0.1) - Učící krok - bude vybrán nejlepší z desetinásobků, takže ladit od 0 do 1
        random(0) - pokud 1, váhy budou inicializovány náhodně (pro linear zbytečné)
        seed(0) - Seed náhodných inicializací - náhodná čísla pro konkrétní seed budou vždy stejná
"""

# Moje
from predictit.models.autoreg_LNU_withwpred import autoreg_LNU_withwpred
from predictit.models.autoreg_LNU import autoreg_LNU
from predictit.models.cg import cg

# Statsmodels
from predictit.models.sm_ar import ar
from predictit.models.sm_arma import arma
from predictit.models.sm_arima import arima
from predictit.models.sm_sarima import sarima

# Tensorflow, Keras
tensorflowit = 0

if tensorflowit:
    from predictit.models.tf_lstm_batch import lstm_batch
    from predictit.models.tf_lstm_bidirectional import lstm_bidirectional
    from predictit.models.tf_lstm_stacked_batch import lstm_stacked_batch
    from predictit.models.tf_lstm_stacked import lstm_stacked
    from predictit.models.tf_lstm import lstm
    from predictit.models.tf_mlp_batch import mlp_batch
    from predictit.models.tf_mlp import mlp

# Scikit
from predictit.models.sklearn_universal import sklearn_universal

# Scikit
from predictit.models.regression_bayes_ridge import regression_bayes_ridge
from predictit.models.regression_hubber import regression_hubber
from predictit.models.regression_lasso import regression_lasso
from predictit.models.regression_linear import regression_linear
from predictit.models.regression_ridge import regression_ridge
from predictit.models.regression_ridge_CV import regression_ridge_CV

from predictit.models.elm import elm
from predictit.models.elm_gen import elm_gen

# Slouží pro porovnání výsledků predikcí s pouhým průměrem
from predictit.models.compare_with_average import compare_with_average


import sklearn
from importlib import import_module

regressors=[]
for module in sklearn.__all__:
    try:
        module = import_module(f'sklearn.{module}')
        regressors.extend([getattr(module,cls) for cls in module.__all__  if 'Regress' in cls ])
    except:
        pass
regressors.append(sklearn.svm.SVR)
default_regressor = sklearn.linear_model.BayesianRidge

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