import numpy as np
from .models_functions.models_functions import one_step_looper


def train(data, epochs=50, learning_rate=1):
    """No need for train in this model - just for consistency with other models.

    Args:
        data ((np.ndarray, np.ndarray)) - Tuple (X, y) of input train vectors X and train outputs y.
            X should contain bias - constant 1 on first place of every sample (parameter constant in `mydatapreprocessing.inputs.make_sequences`).
        epochs(int) - How many times is the algorithm repeated.
        learning_rate(str) - Similar as in neural net and similar to penalization in regression.
    """
    X = data[0]
    y = data[1].ravel()

    w = np.zeros(X.shape[1])

    for i in range(epochs):
        e = y - np.dot(X, w)
        w = w + np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + np.eye(X.shape[1]) / learning_rate), X.T), e)

    return w


def predict(x_input, model, predicts=7):
    """Model that return just aritmetical average from last few datapoints.

    Args:
        x_input (np.ndarray): Time series data inputting the models. Usually last few datapoints. Structure depends on X in train function (usually defined in mydatapreprocessing library).
        model (np.ndarray): Trained model. It's array of neural weights.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    return one_step_looper(lambda x_input: np.dot(x_input, model), x_input.ravel(), predicts)

