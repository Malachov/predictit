import numpy as np


def train(data, length=50):
    """Return average of last n data samples.

    Args:
        data (np.ndarray): Time series data
        length (int, optional): Length of used window (last n values). Defaults to 50.
    """

    return data.ravel()[-length:].mean()


def predict(_, model, predicts=7):
    """Model that return just arithmetical average from last few data points.

    Args:
        _ (None): Here just fo consistency of positional parameters with other models.
        model (float): Mean value.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    return np.array([model] * predicts)
