import numpy as np


def train(sequentions, epochs=100):
    """Conjugate gradient model.

    Args:
        X (np.ndarray): Input vectors. If you have no X, y and x_input, create it with function make sequences in data_prep module.
        y (np.ndarray): Output vectors.
        x_input (np.ndarray): Input that have no output - input for prediction.
        n_steps_in (int, optional): Number of regressive members - inputs to neural unit. Defaults to 50.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        predicted_column_index (int, optional): If multidimensional, define what column is predicted. Defaults to 0.
        epochs (int, optional): Number of epochs to evaluate. Defaults to 100.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    X = sequentions[0]
    y = sequentions[1].ravel()

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
        beta = np.dot(re.T, re) / np.dot(re_prev.T, re_prev)
        p = re + beta * p

    return w


def predict(x_input, model, predicts=7):
    """Function that creates predictions from trained model and input data.

    Args:
        x_input (np.ndarray): Time series data
        model (list, class): Trained model. It can be list of neural weigths or it can be fitted model class from imported library.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    x_input = x_input.ravel()
    predictions = []

    for _ in range(predicts):

        ypre = np.dot(model, x_input)
        predictions.append(ypre)
        x_input = np.insert(x_input, len(x_input), ypre)
        x_input = np.delete(x_input, 1)

    return np.array(predictions)
