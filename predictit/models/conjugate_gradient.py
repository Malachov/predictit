import numpy as np
from .models_functions.models_functions import one_step_looper
import mylogging


def train(data, epochs=100, early_stopping=True):
    """Conjugate gradient model.

    Args:
        data ((np.ndarray, np.ndarray)) - Tuple (X, y) of input train vectors X and train outputs y.
            X should contain bias - constant 1 on first place of every sample
            (parameter constant in `mydatapreprocessing.create_model_inputs.make_sequences`).
        epochs (int, optional): Number of epochs to evaluate. Defaults to 100.
        early_stopping (bool, optional): If mean error don't get lower, next epochs are not evaluated. Defaults to True.

    Returns:
        np.ndarray: Array of neural weights.
    """

    X = data[0]
    y = data[1].ravel()

    w = np.zeros(X.shape[1])

    b = np.dot(X.T, y)
    A = np.dot(X.T, X)
    re = b - np.dot(A, w)
    p = re.copy()

    error_previous = np.inf
    min_error = np.inf
    best_w = np.zeros(X.shape[1])
    worse = 0

    for _ in range(epochs):

        alpha = np.dot(re.T, re) / (np.dot(np.dot(p.T, A), p))
        w = w + alpha * p
        re_previous = re.copy()
        re = re_previous - alpha * np.dot(A, p)

        if early_stopping:
            error = np.abs(re_previous).sum()

            if error < min_error:
                min_error = error
                best_w = w

            if error > error_previous:
                worse += 1

            if worse > 10 or error < 10e-5:
                return best_w

            error_previous = error

        if np.isnan(re).any():
            np.nan_to_num(re, copy=False)
        beta = np.dot(re.T, re) / np.dot(re_previous.T, re_previous)
        if np.isnan(re).any():
            np.nan_to_num(re, copy=False)
        p = re + beta * p

        if np.isnan(p.min()):
            raise RuntimeError(mylogging.return_str("Model is unstable."))

    return w


def predict(x_input, model, predicts=7):
    """Function that creates predictions from trained model and input data.

    Args:
        x_input (np.ndarray): Time series data
        model (np.ndarray): Trained model. Array of neural weights.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    return one_step_looper(
        lambda new_x_input: np.dot(new_x_input, model), x_input.ravel(), predicts
    )
