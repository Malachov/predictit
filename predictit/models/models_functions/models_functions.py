"""Module for functions shared across models."""

from __future__ import annotations

import numpy as np


def one_step_looper(model_function, x_input: np.ndarray, predicts: int, constant: bool = True) -> np.ndarray:
    """Predict one value, generate new input (add new to last and delete first) and predict again.

    Args:
        model_function (function): Callable model with params x_input and predicts.
        x_input (np.ndarray): Input data.
        predicts (int): Number of predicted values.
        constant (bool, optional): Whether model is using bias (1 on every input beginning). Defaults to True.

    Returns:
        np.ndarray: Predicted values.

    Note:
        It's important to correct constant param.
    """
    predictions = []

    input_vector = x_input.copy()

    if input_vector.ndim == 1:
        for _ in range(predicts):
            ypre = model_function(input_vector)
            predictions.append(ypre)
            if not constant:
                input_vector[:-1] = input_vector[1:]
                input_vector[-1] = ypre
            else:
                input_vector[1:-1] = input_vector[2:]
                input_vector[-1] = ypre

    elif input_vector.ndim == 2 and input_vector.shape[0] == 1:
        for _ in range(predicts):
            ypre = model_function(input_vector)
            predictions.append(ypre)
            if not constant:
                input_vector[0, :-1] = input_vector[0, 1:]
                input_vector[0, -1] = ypre
            else:
                input_vector[0, 1:-1] = input_vector[0, 2:]
                input_vector[0, -1] = ypre

    else:
        raise TypeError("Max ndim is 2.")

    return np.array(predictions).reshape(-1)
