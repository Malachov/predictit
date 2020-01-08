
from pandas import Series
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


def sarima(data, predicts=7, plot=0, p=1, d=1, q=1, pp=1, dd=1, qq=1, predicted_column_index=0, season=12, method='lbfgs', trend='n', enforce_invertibility=False, enforce_stationarity=False, forecast_type='out_of_sample'):
    """Seasonal autoregressive model from statsmodels library with moving average part and with integrated part.

    Args:
        data (np.ndarray): Time series data
        predicts (int, optional): Number of predicted values. Defaults to 7.
        plot (int, optional): Whether plot results. Defaults to 0.
        p (int, optional): 1st order of ARIMA. Defaults to 3.
        d (int, optional): 2nd order of ARIMA. Defaults to 1.
        q (int, optional): 3rd order of ARIMA. Defaults to 0.
        pp (int, optional): Seasonal 1st order of ARIMA. Defaults to 1.
        dd (int, optional): Seasonal 2nd order of ARIMA. Defaults to 1.
        qq (int, optional): Seasonal 3rd order of ARIMA. Defaults to 1.
        predicted_column_index (int, optional): If multidimensional, define what column is predicted. Defaults to 0.
        season (int, optional): Number of seasons that repeats. Defaults to 12.
        method (str, optional): Parameter of statsmodels fit function. Defaults to 'lbfgs'.
        trend (str, optional): Parameter of statsmodels SARIMAX function. Defaults to 'n'.
        enforce_invertibility (bool, optional): Parameter of statsmodels SARIMAX function. Defaults to False.
        enforce_stationarity (bool, optional): Parameter of statsmodels SARIMAX function. Defaults to False.
        forecast_type (str, optional): Whether in_sample or out_of_sample prediction. Defaults to 'out_of_sample'.

    Returns:
        np.ndarray: Predictions of input time series.

    """

    data_shape = data.shape
    if len(data_shape) > 1:
        data = data[predicted_column_index]

    param = (p, d, q)
    param_seasonal = (pp, dd, qq, season)


    model = sm.tsa.SARIMAX( data,
                            order = param,
                            seasonal_order = param_seasonal,
                            enforce_stationarity = enforce_stationarity,
                            enforce_invertibility = enforce_invertibility, 
                            trend = trend
    )

    model_fit = model.fit(method=method, disp=-1)

    if forecast_type == 'in_sample':
        predictions = model_fit.forecast(steps=predicts)

    if forecast_type == 'out_of_sample':
        start_index = len(data)
        end_index = start_index - 1  + predicts
        predictions = model_fit.predict(start=start_index, end=end_index)

    if plot == 1:
        plt.plot(predictions, color='red')
        plt.show()

    predictions = np.array(predictions).reshape(-1)

    return predictions
