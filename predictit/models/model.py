"""Module for functions shared across models."""

from __future__ import annotations
from typing import Callable, Any
from abc import ABC, abstractmethod

from typing_extensions import Literal
import numpy as np

import mylogging
import mydatapreprocessing as mdp
from mydatapreprocessing.create_model_inputs import Sequences


# class Model(ABC):
#     def __init__(
#         self, data, sequences_type, predicts, cross_validation: Literal["validate", "predict"] = "validate"
#     ) -> None:
#         self.cross_validation = cross_validation
#         self.predicts = predicts
#         self.data: np.ndarray = data

#         ###############
#         ### inputs outputs
#         ###############

#         if cross_validation == "validate":
#             test_unstandardized = mdp.misc.split(data, predicts=predicts)[1].values
#             models_test_outputs_unstandardized = [test_unstandardized]

#         else:
#             models_test_outputs_unstandardized = mdp.create_model_inputs.create_tests_outputs(
#                 data_for_predictions_df.values[:, 0],
#                 predicts=config.output.predicts,
#                 repeatit=config.prediction.repeatit,
#             )

#     if config.prediction.cross_validation == "validate":
#         data_for_predictions, test = mdp.misc.split(data_for_predictions, predicts=config.predicts)
#         models_test_outputs = [test]

#     else:
#         models_test_outputs = mdp.create_model_inputs.create_tests_outputs(
#             data_for_predictions[:, 0],
#             predicts=config.output.predicts,
#             repeatit=config.prediction.repeatit,
#         )
#         ###############
#         ### ANCHOR ### Feature selection
#         #############

#         # data_for_predictions_df TODO

#     @abstractmethod
#     def fit():
#         pass

#     @abstractmethod
#     def predict():
#         pass

#     @abstractmethod
#     def save():
#         pass

#     @abstractmethod
#     def load():
#         pass


def get_inputs(input: tuple[np.ndarray, np.ndarray] | Sequences) -> tuple[np.ndarray, np.ndarray]:

    if isinstance(input, Sequences):
        return input[0], input[1]

    if not isinstance(input, tuple):
        raise TypeError(
            mylogging.return_str("Data must be tuple of length 2 - input vector and output vector.")
        )

    if len(input) != 2:
        raise ValueError(
            mylogging.return_str("Data must be tuple of length 2 - input vector and output vector.")
        )

    return input[0], input[1]


def one_step_looper(
    model_function: Callable, x_input: np.ndarray, predicts: int, constant: bool = True
) -> np.ndarray:
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
