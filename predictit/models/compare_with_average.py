import numpy as np


def compare_with_average(data, predicts=7, predicted_column_index=0):
    """Model that return just aritmetical average from last few datapoints.
    
    Args:
        data (np.ndarray): Time series data.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        predicted_column_index (int, optional): If multidimensional, define what column is predicted. Defaults to 0.
    
    Returns:
        np.ndarray: Predictions of input time series.
    """    
    data_shape = data.shape

    if len(data_shape) != 1:
        data = data[predicted_column_index]
    data = data[-3 * predicts:]
    summed = sum(data)
    average = summed / len(data)
    predictions_list = [average] * predicts
    predictions = np.array(predictions_list)

    return predictions
