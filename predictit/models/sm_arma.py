import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def arma(data, predicts=7, plot=0, p=3, q=1, predicted_column_index=0, method='mle', ic='aic', trend='nc', solver='lbfgs', forecast_type='out_of_sample'):
    """Autoregressive model from statsmodels library with moving average part.

    Args:
        data (np.ndarray): Time series data
        predicts (int, optional): Number of predicted values. Defaults to 7.
        plot (int, optional): Whether plot results. Defaults to 0.
        p (int, optional): 1st order of ARMA. Defaults to 3.
        q (int, optional): 2nd order of ARMA. Defaults to 1.
        predicted_column_index (int, optional): If multidimensional, define what column is predicted. Defaults to 0.
        method (str, optional): Parameter of statsmodels fit function. Defaults to 'mle'.
        ic (str, optional): Parameter of statsmodels fit function. Defaults to 'aic'.
        trend (str, optional): Parameter of statsmodels fit function. Defaults to 'nc'.
        solver (str, optional): Parameter of statsmodels fit function. Defaults to 'lbfgs'.
        forecast_type (str, optional): Whether in_sample or out_of_sample prediction. Defaults to 'out_of_sample'.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    data = np.array(data)
    data_shape = np.shape(data)
    if len(data_shape) > 1:
        data = data[predicted_column_index]

    order = (p, q)

    model = sm.tsa.ARMA(data, order=order)

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
