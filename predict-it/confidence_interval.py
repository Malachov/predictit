from pandas import Series
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def bounds(data, predicts=7, confidence=0.1, p=3, d=0, q=1):

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

    order = (p, d, q)

    model = sm.tsa.ARIMA(data, order=order)

    model_fit = model.fit(disp=0)
    predictions = model_fit.forecast(steps=predicts, alpha=confidence)

    return predictions[0], predictions[1], predictions[2]


