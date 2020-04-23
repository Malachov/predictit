# -*- coding: utf-8 -*-

"""
Library/framework for making predictions. Choose best of more than models
(AR, ARIMA, many regressions, nerual nets like MLP or LSTM...). Preprocess data and chose optimal parameters
of predictions. Output is plotly HTML interactive plot, numpy array of results or database deploying.

For tutorials and examples view readme (or tests).
"""

from predictit import analyze
from predictit import best_params
from predictit import models
from predictit import config
from predictit import data_preprocessing
from predictit import database
from predictit import define_inputs
from predictit import evaluate_predictions
from predictit import misc
from predictit import plot
from predictit import test_data

from predictit import main


__version__ = "1.25"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = ['analyze', 'best_params', 'config', 'data_preprocessing', 'database', 'define_inputs', 'evaluate_predictions', 'main', 'misc', 'models', 'plot', 'test_data']
