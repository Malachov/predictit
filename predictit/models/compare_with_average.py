import numpy as np


def train(data):
    """No need for train in this model - just for consistency with other models.

    Args:
        data (None): Just for consistency with other models.
    """
    pass


def predict(data, model, predicts=7):
    """Model that return just aritmetical average from last few datapoints.

    Args:
        data (np.ndarray): Time series data.
        model: None - argument just for consistent call with other models.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    data = data[-predicts:]
    summed = sum(data)
    average = summed / len(data)
    predictions_list = [average] * predicts

    return np.array(predictions_list)
