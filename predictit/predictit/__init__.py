"""Library/framework for making predictions. Choose best of 20 models (ARIMA, regressions, LSTM...). Preprocess data and chose optimal parameters of predictions. Output is plotly HTML interactive graph or database deploying"""
import numpy as np
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import time
import pickle
from pathlib import Path
import os
import plotly.offline as py
import plotly.graph_objs as go
import cufflinks as cf
import sklearn
import pandas as pd
import warnings
from sklearn import preprocessing

from . import data_prep
from . import models
from . import analyze
from . import best_params
from . import confidence_interval
from . import config
from . import database
from . import test_pre
from . import main
from . import main_function
from . import data_test
