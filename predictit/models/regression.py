"""Simple Linear and Ridge regression."""

from __future__ import annotations

from typing_extensions import Literal

import numpy as np

from .models_functions.models_functions import one_step_looper, get_inputs


def train(
    data: tuple[np.ndarray, np.ndarray], model: Literal["linear", "ridge"] = "linear", lmbda: float = 0.1
):
    """No need for train in this model - just for consistency with other models.

    Args:
        data (tuple[np.ndarray, np.ndarray]) - Tuple (X, y) of input train vectors X and train outputs y.
            X should contain bias - constant 1 on first place of every sample (parameter constant
            in `mydatapreprocessing.create_model_inputs.make_sequences`).
        model(Literal['linear', 'ridge'], optional) - 'linear' or 'ridge'. Defaults to 'linear'.
        lmbda(float, optional) - Lambda parameter defining regularization. Defaults to 0.1.

    Returns:
        np.ndarray: Array of neural weights.
    """

    X, y = get_inputs(data)
    y = y.ravel()

    if model == "linear":
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    elif model == "ridge":
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lmbda * np.eye(X.shape[1])), X.T), y)

    else:
        raise ValueError("Model must be one of ['linear', 'ridge']")

    return w


def predict(x_input: np.ndarray, model: np.ndarray, predicts: float = 7):
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
