"""Some statsmodels models. Statsmodels has different data imput form. Uses no sequentions, but original series."""

from __future__ import annotations
from typing import Any

from typing_extensions import Literal
import numpy as np

# Lazy imports
# import statsmodels.tsa.api as sm
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa import ar_model


def train(
    data: np.ndarray,
    used_model: str = "autoreg",
    p: int = 5,
    d: int = 1,
    q: int = 0,
    cov_type="nonrobust",
    method="cmle",
    trend="nc",
    solver="lbfgs",
    maxlag=13,
    # SARIMAX args
    seasonal=(0, 0, 0, 0),
) -> Any:
    """Autoregressive model from statsmodels library. Only univariate data.

    Args:
        data (np.ndarray): Time series data.
        used_model (str, optional): Used model. Defaults to "autoreg".
        p (int, optional): Order of ARIMA model (1st - proportional). Check statsmodels docs for more. Defaults to 5.
        d (int, optional): Order of ARIMA model. Defaults to 1.
        q (int, optional): Order of ARIMA model. Defaults to 0.
        cov_type: Parameters of model call or fit function of particular model. Check statsmodels docs for more.
            Defaults to 'nonrobust'.
        method: Parameters of model call or fit function of particular model. Check statsmodels docs for more.
            Defaults to 'cmle'.
        trend: Parameters of model call or fit function of particular model. Check statsmodels docs for more.
            Defaults to 'nc'.
        solver: Parameters of model call or fit function of particular model. Check statsmodels docs for more.
            Defaults to 'lbfgs'.
        maxlag: Parameters of model call or fit function of particular model. Check statsmodels docs for more.
            Defaults to 13.
        seasonal: Parameters of model call or fit function of particular model. Check statsmodels docs for more.
            Defaults to (0, 0, 0, 0).

    Returns:
        statsmodels.model: Trained model.
    """

    import statsmodels.tsa.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa import ar_model

    used_model = used_model.lower()

    if used_model == "ar":
        model = sm.AR(data)
        fitted_model = model.fit(method=method, trend=trend, solver=solver, disp=0)

    elif used_model == "arima":
        order = (p, d, q)
        model = ARIMA(data, order=order)
        fitted_model = model.fit()

    elif used_model == "sarimax":
        order = (p, d, q)
        model = SARIMAX(data, order=order, seasonal_order=seasonal)
        fitted_model = model.fit(method=method, trend=trend, solver=solver, disp=0)

    elif used_model == "autoreg":
        auto = ar_model.ar_select_order(data, maxlag=maxlag)
        model = ar_model.AutoReg(
            data,
            lags=auto.ar_lags,
            trend=auto.trend,
            seasonal=auto.seasonal,
            period=auto.period,
        )
        fitted_model = model.fit(cov_type=cov_type)

    else:
        raise ValueError(
            f"Used model has to be one of ['ar', 'arima', 'sarimax', 'autoreg']. You configured: {used_model}"
        )

    setattr(fitted_model, "my_name", used_model)
    setattr(fitted_model, "data_length", len(data))

    return fitted_model


def predict(data: np.ndarray, model: Any, predicts: int = 7) -> np.ndarray:
    """Function that creates predictions from trained model and input data.

    Args:
        data (np.ndarray): Time series data
        model (Any, class): Trained model. It can be list of neural weights or it can
            be fitted model class from imported library.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    start = len(data) if len(data) > model.data_length else model.data_length

    # Input data must have same starting point as data in train so the starting point be correct
    if model.my_name == "arima":
        predictions = model.predict(start=start, end=start - 1 + predicts, typ="levels")

    else:
        predictions = model.predict(start=start, end=start - 1 + predicts)

    return predictions
