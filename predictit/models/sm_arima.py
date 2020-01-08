from pandas import Series
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


def arima(data, predicts=7, plot=0, p=3, d=1, q=0, predicted_column_index=0, method='mle', ic='aic', trend='nc', solver='lbfgs', forecast_type='out_of_sample'):
    """Autoregressive model from statsmodels library with moving average part and with integrated part.

    Args:
        data (np.ndarray): Time series data.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        plot (int, optional): Whether plot results. Defaults to 0.
        p (int, optional): 1st order of ARIMA. Defaults to 3.
        d (int, optional): 2nd order of ARIMA. Defaults to 1.
        q (int, optional): 3rd order of ARIMA. Defaults to 0.
        predicted_column_index (int, optional): If multidimensional, define what column is predicted. Defaults to 0.
        method (str, optional): Parameter of statsmodels fit function. Defaults to 'mle'.
        ic (str, optional): Parameter of statsmodels fit functionParameter of statsmodels fit function. Defaults to 'aic'.
        trend (str, optional): Parameter of statsmodels fit functionParameter of statsmodels fit function. Defaults to 'nc'.
        solver (str, optional): Parameter of statsmodels fit functionParameter of statsmodels fit function. Defaults to 'lbfgs'.
        forecast_type (str, optional): Whether in_sample or out_of_sample prediction. Defaults to 'out_of_sample'.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    # data_shape = data.shape
    # if len(data_shape) > 1:
    #     data = data[predicted_column_index]

    order = (p, d, q)

    model = sm.tsa.ARIMA(data, order=order)
    model_fit = model.fit(method=method, ic=ic, trend=trend, solver=solver, disp=0)

    if forecast_type == 'in_sample':
        predictions = model_fit.forecast(steps=predicts)[0]

    if forecast_type == 'out_of_sample':
        start_index = len(data)
        end_index = start_index - 1 + predicts
        if d:
            predictions = model_fit.predict(start=start_index, end=end_index, typ='levels', dynamic='True')
        else:
            predictions = model_fit.predict(start=start_index, end=end_index, dynamic='True')

    predictions = np.array(predictions).reshape(-1)

    if plot == 1:
        plt.plot(predictions, color='red')
        plt.show()

    return predictions
