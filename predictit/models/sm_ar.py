import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR, _ar_predict_out_of_sample
import numpy as np

def ar(data, predicts=7, plot=0, predicted_column_index=0, method='cmle', ic='aic', trend='nc', solver='lbfgs'):
    """Autoregressive model from statsmodels library.

    Args:
        data (np.ndarray): Time series data.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        plot (int, optional): Whether plot results. Defaults to 0.
        predicted_column_index (int, optional): If multidimensional, define what column is predicted. Defaults to 0.
        method (str, optional): Parameter of statsmodels fit function. Defaults to 'cmle'.
        ic (str, optional): Parameter of statsmodels fit function. Defaults to 'aic'.
        trend (str, optional): Parameter of statsmodels fit function. Defaults to 'nc'.
        solver (str, optional): Parameter of statsmodels fit function. Defaults to 'lbfgs'.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    data = np.array(data)
    data_shape = np.shape(data)
    if len(data_shape) > 1:
        data = data[predicted_column_index]

    model = AR(data)

    model_fit = model.fit(method=method, ic=ic, trend=trend, solver=solver, disp=-1)
    #window = model_fit.k_ar
    #coef = model_fit.params

    endogg = [i[0] for i in model.endog]
    predictions = _ar_predict_out_of_sample(endogg, model_fit.params, model.k_ar, model.k_trend, steps = predicts, start=0)

    if plot == 1:
        plt.plot(predictions, color='red')
        plt.show()

    predictions = np.array(predictions).reshape(-1)

    return predictions
