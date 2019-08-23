from pandas import Series
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def arima(data, predicts=7, plot=0, p=3, d=1, q=0, predicted_column_index=0, method='mle', ic='aic', trend='nc', solver='lbfgs', forecast_type='out_of_sample'):
    """Autoregressive integrated moving average model
    """
    '''
    data_shape = data.shape
    if len(data_shape) > 1:
        data = data[predicted_column_index]
    '''
    order = (p, d, q)

    model = sm.tsa.ARIMA(data, order=order)
    model_fit = model.fit(method=method, ic=ic, trend=trend, solver=solver, disp=1)

    if forecast_type == 'in_sample':
        predictions = model_fit.forecast(steps=predicts)[0]

    if forecast_type == 'out_of_sample':
        start_index = len(data)
        end_index = start_index - 1  + predicts
        if d:
            predictions = model_fit.predict(start=start_index, end=end_index, typ='levels', dynamic='True')
        else:
            predictions = model_fit.predict(start=start_index, end=end_index, dynamic='True')

    predictions = np.array(predictions).reshape(-1)

    if plot == 1:
        plt.plot(predictions, color='red')
        plt.show()

    return predictions

