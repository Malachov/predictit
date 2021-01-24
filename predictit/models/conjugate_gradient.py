import numpy as np
from .models_functions.models_functions import one_step_looper


def train(data, epochs=100):
    """Conjugate gradient model.

    Args:
        data ((np.ndarray, np.ndarray)) - Tuple (X, y) of input train vectors X and train outputs y
                    X should contain bias - constant 1 on first place of every sample (parameter constant in `mydatapreprocessing.inputs.make_sequences`).
        epochs (int, optional): Number of epochs to evaluate. Defaults to 100.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    X = data[0]
    y = data[1].ravel()

    w = np.zeros(X.shape[1])

    b = np.dot(X.T, y)
    A = np.dot(X.T, X)
    re = b - np.dot(A, w)
    p = re.copy()

    for _ in range(epochs):

        alpha = np.dot(re.T, re) / (np.dot(np.dot(p.T, A), p))
        w = w + alpha * p
        re_prev = re.copy()
        re = re_prev - alpha * np.dot(A, p)
        if np.isnan(re).any():
            np.nan_to_num(re, copy=False)
        beta = np.dot(re.T, re) / np.dot(re_prev.T, re_prev)
        if np.isnan(re).any():
            np.nan_to_num(re, copy=False)
        p = re + beta * p
        if np.isnan(p).any():
            if np.isnan(p).all():
                break
            else:
                np.nan_to_num(p, copy=False)

    return w


def predict(x_input, model, predicts=7):
    """Function that creates predictions from trained model and input data.

    Args:
        x_input (np.ndarray): Time series data
        model (list, class): Trained model. It can be list of neural weights or it can be fitted model class from imported library.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    return one_step_looper(lambda x_input: np.dot(x_input, model), x_input.ravel(), predicts)
