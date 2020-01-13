""" Module to find area with some probability where predictions will be. It's used mainly for plotting where it create grey area where we can expect values."""

import pandas as pd
import statsmodels.api as sm
import numpy as np


def bounds(data, predicts=7, confidence=0.1, p=1, d=0, q=0):
    """Function to find confidence interval of prediction for graph.

    Args:
        data (np.ndarray): Time series data
        predicts (int, optional): [description]. Defaults to 7.
        confidence (float, optional): [description]. Defaults to 0.1.
        p (int, optional): 1st order of ARIMA. Defaults to 1.
        d (int, optional): 2nd order of ARIMA. Defaults to 0.
        q (int, optional): 3rd order of ARIMA. Defaults to 0.

    Returns:
        list, list: Lower bound, upper bound.

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

        try:
            import data_prep

            last_value = data[-1]
            data = data_prep.do_difference(data)

            model = sm.tsa.ARIMA(data, order=order)
            model_fit = model.fit(disp=0)
            predictions = model_fit.forecast(steps=predicts, alpha=confidence)

            lower_bound = data_prep.inverse_difference(bounds[0], last_value)
            upper_bound = data_prep.inverse_difference(bounds[1], last_value)

        except Exception:
            return None

    return lower_bound, upper_bound
