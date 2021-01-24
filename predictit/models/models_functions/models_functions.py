import numpy as np


def one_step_looper(model_function, x_input, predicts, constant=True):

    predictions = []

    for _ in range(predicts):
        ypre = model_function(x_input)
        predictions.append(ypre)
        if not constant:
            x_input[:-1] = x_input[1:]
            x_input[-1] = ypre
        else:
            x_input[1:-1] = x_input[2:]
            x_input[-1] = ypre

    return np.array(predictions).reshape(-1)
