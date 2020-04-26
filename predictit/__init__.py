# -*- coding: utf-8 -*-

"""
Library/framework for making predictions. Choose best of more than models
(AR, ARIMA, many regressions, nerual nets like MLP or LSTM...). Preprocess data and chose optimal parameters
of predictions. Output is plotly HTML interactive plot, numpy array of results or database deploying.

How to use it:

1) As downloded framework.

    Setup config['py'] (mainly data source), save and run main.py

    Examples (in terminal):

        >>> python main.py

2) Import from library

    Examples:

        >>> import predictit
        >>> import numpy as np
        >>> config = predictit.config.config

        >>> config.update({
        >>>     'data_source': 'test',
        >>>     'csv_full_path': r'path/to/csv',
        >>>     'predicted_column': 'CO2 (%)'
        >>>     'datetime_index': 'Time',
        >>>     'freq': '2S',
        >>> })

        >>> predictions = predictit.main.predict()

    You can setup config also via function arguments.

        >>> predictions = predictit.main.predict(np.random.randn(1, 100), predicts=3, plot=1)

3) Via CLI

    Examples (in terminal):

        >>> python main.py --function predict --data_source 'csv' --csv_full_path 'test_data/daily-minimum-temperatures.csv' --predicted_column 1

For more informations and some use cases check predictit readme or official docs. at

    https://predictit.readthedocs.io

Feel free to contribute at

    https://github.com/Malachov/predictit

More function examples is in tests/test_it.py for some more working examples.

If you want to know how some parts works, check tests/visual where are displayed used function results on simple example.
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


__version__ = "1.31"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = ['analyze', 'best_params', 'config', 'data_preprocessing', 'database', 'define_inputs', 'evaluate_predictions', 'main', 'misc', 'models', 'plot', 'test_data']
