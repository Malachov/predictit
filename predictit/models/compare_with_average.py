import numpy as np

def compare_with_average(data, predicts = 7, predicted_column_index=0):

    data = np.array(data)
    data_shape = data.shape

    if data_shape != 1:
        data = data[predicted_column_index]

    data_list = list(data)
    summed = sum(data_list)
    average  = summed / predicts
    predictions_list = [average] * predicts
    predictions = np.array(predictions_list)

    return(predictions)