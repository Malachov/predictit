# -*- coding: utf-8 -*-

"""
Library/framework for making predictions. Choose best of 20 models
(ARIMA, regressions, LSTM...). Preprocess data and chose optimal parameters
of predictions. Output is plotly HTML interactive graph or database deploying. For tutorials and example view readme.

For more read README.md
"""

from . import analyze
from . import best_params
from . import models
from . import config
from . import test_data
from . import database
from . import evaluate_predictions
from . import main
from . import confidence_interval
from . import pickle_test_data

__version__ = "0.4"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"
