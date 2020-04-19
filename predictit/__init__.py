# -*- coding: utf-8 -*-

"""
Library/framework for making predictions. Choose best of 20 models
(ARIMA, regressions, LSTM...). Preprocess data and chose optimal parameters
of predictions. Output is plotly HTML interactive graph or database deploying.
For tutorials and examples view readme (or tests).
"""

from predictit import analyze
from predictit import best_params
from predictit import config
from predictit import models
from predictit import data_preprocessing
from predictit import database
from predictit import define_inputs
from predictit import evaluate_predictions
from predictit import main
from predictit import misc
from predictit import plot
from predictit import test_data


__version__ = "1.21"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = ['analyze', 'best_params', 'config', 'data_preprocessing', 'database', 'define_inputs', 'evaluate_predictions', 'main', 'misc', 'models', 'plot', 'test_data']
