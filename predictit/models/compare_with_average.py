import numpy as np


def compare_with_average(data, predicts=7, predicted_column_index=0):

    data_shape = data.shape

    if len(data_shape) != 1:
        data = data[predicted_column_index]
    data = data[-3 * predicts:]
    summed = sum(data)
    average = summed / len(data)
    predictions_list = [average] * predicts
    predictions = np.array(predictions_list)

    return predictions
