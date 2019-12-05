from pandas import Series
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from predictit.data_prep import do_difference, inverse_difference

def bounds(data, predicts=7, confidence=0.1, p=1, d=0, q=0):

    """Function to find confidence interval of prediction for graph
    ======
    Output:
    ------
        ARIMA predictions {arr}, Confidency{f}, Bounds{arr}

    Arguments:
    ------
        data {arr, list} -- Data for prediction
        predicts{int} -- Number of predictions
        confidence{f (0,1)} -- Confidency area
        p{} -- Order of ARIMA model
        d{} -- Order of ARIMA model
        q{} -- Order of ARIMA model
    """
    if len(data) <= 10:
        return

    last_value = data[-1]
    #data = do_difference(data)
    order = (p, d, q)

    model = sm.tsa.ARIMA(data, order=order)

    model_fit = model.fit(disp=0)
    predictions = model_fit.forecast(steps=predicts, alpha=confidence)

    bounds = predictions[2].T
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    #lower_bound = inverse_difference(bounds[0], last_value)
    #upper_bound = inverse_difference(bounds[1], last_value)

    return lower_bound, upper_bound

