

def train(data, used_model='autoreg', p=5, d=1, q=0, cov_type='nonrobust', method='cmle', trend='nc', solver='lbfgs', maxlag=13,
          # SARIMAX args
          seasonal=(1, 1, 1, 24)  # , enforce_invertibility=False, enforce_stationarity=False
          ):
    """Autoregressive model from statsmodels library. Only univariate data.
    Args:
        data (np.ndarray): Time series data.
        used_model (str, optional): One of ['ar', 'arima', 'sarimax', 'autoreg'].
        p (int) - Order of ARIMA model (1st - proportional). Check statsmodels docs for more.
        d (int) - Order of ARIMA model.
        q (int) - Order of ARIMA model.
        cov_type, method, trend, solver, maxlag, seasonal: Parameters of model call or fit function of particular model.
            Check statsmodels docs for more.
    Returns:
        statsmodels.model: Trained model.
    """

    import statsmodels.tsa.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa import ar_model

    used_model = used_model.lower()

    if used_model == 'ar':
        model = sm.AR(data)
        fitted_model = model.fit(method=method, trend=trend, solver=solver, disp=0)

    elif used_model == 'arima':
        order = (p, d, q)
        model = ARIMA(data, order=order)
        fitted_model = model.fit()

    elif used_model == 'sarimax':
        order = (p, d, q)
        model = SARIMAX(data, order=order, seasonal_order=seasonal)
        fitted_model = model.fit(method=method, trend=trend, solver=solver, disp=0)

    elif used_model == 'autoreg':
        auto = ar_model.ar_select_order(data, maxlag=maxlag)
        model = ar_model.AutoReg(data, lags=auto.ar_lags, trend=auto.trend, seasonal=auto.seasonal, period=auto.period)
        fitted_model = model.fit(cov_type=cov_type)

    else:
        raise ValueError(f"Used model has to be one of ['ar', 'arima', 'sarimax', 'autoreg']. You configured: {used_model}")

    fitted_model.my_name = used_model
    fitted_model.data_len = len(data)

    return fitted_model


def predict(data, model, predicts=7):
    """Function that creates predictions from trained model and input data.

    Args:
        data (np.ndarray): Time series data
        model (list, class): Trained model. It can be list of neural weights or it can be fitted model class from imported library.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    start = model.data_len if len(data) > model.data_len else len(data)

    # Input data must have same starting point as data in train so the starting point be correct
    if model.my_name == 'arima':
        predictions = model.predict(start=start, end=start - 1 + predicts, typ='levels')[-predicts:]

    else:
        predictions = model.predict(start=start, end=start - 1 + predicts)[-predicts:]

    return predictions
