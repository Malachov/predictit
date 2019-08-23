
from pandas import Series
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

def sarima(data, predicts=7, plot=0, p=1, d=1, q=1, pp=1, dd=1, qq=1, predicted_column_index=0, season=12, method='lbfgs', trend='n', enforce_invertibility=False, enforce_stationarity=False, forecast_type='out_of_sample', verbose=0):
    """Seasonal ARIMAS
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

