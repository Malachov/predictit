# -*- coding: utf-8 -*-

"""
Library/framework for making predictions. Choose best of 20 models
(ARIMA, regressions, LSTM...). Preprocess data and chose optimal parameters
of predictions. Output is plotly HTML interactive graph or database deploying.
For tutorials and examples view readme (or tests).
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

__version__ = "0.62"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"
