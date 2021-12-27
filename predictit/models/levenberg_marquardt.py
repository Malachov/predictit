from __future__ import annotations

import numpy as np

from .model import one_step_looper, get_inputs


def train(data: tuple[np.ndarray, np.ndarray], epochs: int = 50, learning_rate: float = 1.0):
    """No need for train in this model - just for consistency with other models.

    Args:
        data (tuple[np.ndarray, np.ndarray]) - Tuple (X, y) of input train vectors X and train outputs y.
            X should contain bias - constant 1 on first place of every sample
            (parameter constant in `mydatapreprocessing.create_model_inputs.make_sequences`).
        epochs(int, optional) - How many times is the algorithm repeated. Defaults to 50.
        learning_rate(str, optional) - Similar as in neural net and similar to penalization in regression.
            Defaults to 1.

    Returns:
        np.ndarray: Array of neural weights.
    """

    X, y = get_inputs(data)
    y = y.ravel()

    w = np.zeros(X.shape[1])

    for i in range(epochs):
        e = y - np.dot(X, w)
        w = w + np.dot(
            np.dot(np.linalg.inv(np.dot(X.T, X) + np.eye(X.shape[1]) / learning_rate), X.T),
            e,
        )

    return w


def predict(x_input: np.ndarray, model: np.ndarray, predicts: int = 7):
    """Model that return just arithmetical average from last few data points.

    Args:
        x_input (np.ndarray): Time series data inputting the models. Usually last few data points.
            Structure depends on X in train function (usually defined in mydatapreprocessing library).
        model (np.ndarray): Trained model. It's array of neural weights.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    return one_step_looper(lambda new_x_input: np.dot(new_x_input, model), x_input.ravel(), predicts)
