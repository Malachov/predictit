import numpy as np


def one_step_looper(model_function, x_input, predicts, constant=True):
    """Predict one value, generate new input (add new to last and delete first) and predict again.

    Args:
        model_function (function): Callable model with params x_input and predicts.
        x_input (np.ndarray): Input data.
        predicts (int): Number of predicted values.
        constant (bool, optional): Whether model is using bias (1 on every input beginning). Defaults to True.

    Returns:
        np.ndarray: Predicted values.
    """
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
