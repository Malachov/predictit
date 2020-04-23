import numpy as np


def train(data):
    pass


def predict(data, model, predicts=7):
    """Model that return just aritmetical average from last few datapoints.

    Args:
        data (np.ndarray): Time series data.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        predicted_column_index (int, optional): If multidimensional, define what column is predicted. Defaults to 0.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    data = data[-predicts:]
    summed = sum(data)
    average = summed / len(data)
    predictions_list = [average] * predicts
    predictions = np.array(predictions_list)

    return predictions
