import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def arma(data, predicts=7, plot=0, p=3, q=1, predicted_column_index=0, method='mle', ic='aic', trend='nc', solver='lbfgs', forecast_type='out_of_sample'):
    """Autoregressive integrated moving average model
    """

    data = np.array(data)
    data_shape = data.shape

    order = (p, q)

    if len(data.shape) == 1:
        model = sm.tsa.ARMA(data, order=order)

    else:
        model = sm.tsa.ARMA(data[predicted_column_index], order=order)

    model_fit = model.fit(method=method, ic=ic, trend=trend, solver=solver, disp=0)

    if forecast_type == 'in_sample':
        predictions = model_fit.forecast(steps=predicts)[0]

    if forecast_type == 'out_of_sample':
        start_index = len(data)
        end_index = start_index - 1  + predicts
        predictions = model_fit.predict(start=start_index, end=end_index)

    predictions = np.array(predictions).reshape(-1)

    if plot == 1:
        plt.plot(predictions, color='red')
        plt.show()

    return predictions
