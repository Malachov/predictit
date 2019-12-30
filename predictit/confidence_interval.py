import pandas as pd
import statsmodels.api as sm
import numpy as np


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

    order = (p, d, q)

    try:

        model = sm.tsa.ARIMA(data, order=order)
        model_fit = model.fit(disp=0)
        predictions = model_fit.forecast(steps=predicts, alpha=confidence)

        bounds = predictions[2].T
        lower_bound = bounds[0]
        upper_bound = bounds[1]

    except Exception:

        import predictit

        last_value = data[-1]
        data = predictit.data_prep.do_difference(data)

        model = sm.tsa.ARIMA(data, order=order)
        model_fit = model.fit(disp=0)
        predictions = model_fit.forecast(steps=predicts, alpha=confidence)

        lower_bound = predictit.data_prep.inverse_difference(bounds[0], last_value)
        upper_bound = predictit.data_prep.inverse_difference(bounds[1], last_value)

    return lower_bound, upper_bound
