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
