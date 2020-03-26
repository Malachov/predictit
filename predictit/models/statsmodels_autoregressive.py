import statsmodels


def train(data, model='ar', p=3, d=1, q=0, method='cmle', ic='aic', trend='nc', solver='lbfgs'):
    """Autoregressive model from statsmodels library. Only univariate data.

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

    if model == 'ar':
        model = statsmodels.tsa.ar_model.AR(data)

    if model == 'arma':
        order = (p, q)
        model = statsmodels.tsa.arima_model.ARMA(data, order=order)

    if model == 'arima':
        order = (p, d, q)
        model = statsmodels.tsa.arima_model.ARIMA(data, order=order)

    fitted_model = model.fit(method=method, ic=ic, trend=trend, solver=solver, disp=0)

    return fitted_model


def predict(data, fitted_model, predicts=7):

    try:
        predictions = fitted_model.forecast(steps=predicts)[0]

    except Exception:
        # Function AR have no forecast, so use predict. n_totobs is length of data for first predicted index
        predictions = fitted_model.predict(start=fitted_model.n_totobs, end=fitted_model.n_totobs - 1 + predicts)

    return predictions
