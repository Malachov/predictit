import statsmodels


def train(data, used_model='autoreg', p=5, d=1, q=0, cov_type='nonrobust', method='cmle', ic='aic', trend='nc', solver='lbfgs',
          # SARIMAX args
          seasonal=(1, 1, 1, 24), enforce_invertibility=False, enforce_stationarity=False):
    """Autoregressive model from statsmodels library. Only univariate data.
    Args:
        data (np.ndarray): Time series data.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        plot (int, optional): Whether plot results. Defaults to 0.
        predicted_column_index (int, optional): If multidimensional, define what column is predicted. Defaults to 0.
        method (str, optional): Parameter of statsmodels fit function. Defaults to 'cmle'.
        ic (str, optional): Parameter of statsmodels fit function. Defaults to 'aic'.
        trend (str, optional): Parameter of statsmodels fit function. Defaults to 'nc'.
        solver (str, optional):ort statsmodels.api as sm
      File "/home/dan/.local/lib/pyt Parameter of statsmodels fit function. Defaults to 'lbfgs'.
    Returns:
        np.ndarray: Predictions of input time series.
    """

    if used_model == 'ar':
        model = statsmodels.tsa.ar_model.AR(data)

    elif used_model == 'arma':
        order = (p, q)
        model = statsmodels.tsa.arima_model.ARMA(data, order=order)

    elif used_model == 'arima':
        order = (p, d, q)
        model = statsmodels.tsa.arima_model.ARIMA(data, order=order)

    elif used_model == 'sarimax':
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        order = (p, d, q)
        model = SARIMAX(data, order=order, seasonal_order=seasonal)

    if used_model in ('ar', 'arma', 'arima', 'sarimax'):
        fitted_model = model.fit(method=method, ic=ic, trend=trend, solver=solver, disp=0)

    elif used_model == 'autoreg':
        auto = statsmodels.tsa.ar_model.ar_select_order(data, maxlag=40)
        model = statsmodels.tsa.ar_model.AutoReg(data, lags=auto.ar_lags, trend=auto.trend, seasonal=auto.seasonal, period=auto.period)
        fitted_model = model.fit(cov_type=cov_type)

    fitted_model.my_name = used_model
    fitted_model.data_len = len(data)

    return fitted_model


def predict(data, fitted_model, predicts=7):

    start = fitted_model.data_len if len(data) > fitted_model.data_len else len(data)

    # Input data must have same starting point as data in train so the starting point be correct
    if fitted_model.my_name == 'arima':
        predictions = fitted_model.predict(start=start, end=start - 1 + predicts, typ='levels')[-predicts:]

    else:
        predictions = fitted_model.predict(start=start, end=start - 1 + predicts)[-predicts:]

    return predictions
