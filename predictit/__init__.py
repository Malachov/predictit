# -*- coding: utf-8 -*- 

"""
Library/framework for making predictions. Choose best of 20 models 
(ARIMA, regressions, LSTM...). Preprocess data and chose optimal parameters 
of predictions. Output is plotly HTML interactive graph or database deploying. For tutorials and example view readme.

For more read README.md
"""

from . import config
from . import analyze
from . import best_params
from . import data_test
from . import models
from . import database
from . import test_pre
from . import main
from . import main_function
from . import data_test
from . import confidence_interval

__version__ = "0.22"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"
