import numpy as np
from .. import misc
from ..best_params import optimize
from typing import Union, Tuple

import mylogging


def lnu_core(
    data: Tuple[np.ndarray],
    learning_rate: float,
    epochs: int,
    normalize_learning_rate: bool,
    early_stopping: bool = True,
    learning_rate_decay: float = 0.8,
    damping: Union[int, float] = 1,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    X = data[0]
    y_hat = data[1]

    y = np.zeros(len(y_hat))
    error = np.zeros(len(y_hat)) if return_all else np.zeros(1)
    w = np.zeros(X.shape[1])
    last_running_error = np.inf

    w_all = np.zeros((X.shape[1], X.shape[0])) if return_all else None

    for epoch in range(epochs):

        running_error = np.zeros(1)

        for j in range(X.shape[0]):

            current_index = j if return_all else 0

            y[j] = np.dot(w, X[j])
            if y[j] > y_hat.max() * 10e6:
                raise RuntimeError(mylogging.return_str("Model is unstable"))

            error[current_index] = y_hat[j] - y[j]
            running_error[0] = running_error[0] + abs(error[current_index])

            dydw = X[j]
            if normalize_learning_rate:
                minorm = learning_rate / (damping + np.dot(X[j], X[j].T))
                dw = minorm * error[current_index] * dydw
            else:
                dw = learning_rate * error[current_index] * dydw
            w = w + dw

            if return_all:
                w_all[:, j] = w

        if (early_stopping and epoch > 1) and (
            sum(np.abs(dw)) / len(w) < 10e-8
            or ((running_error[0] / len(y_hat)) - last_running_error) < 10e-5
        ):
            break

        last_running_error = running_error[0] / len(y_hat)

        if learning_rate_decay:
            learning_rate = learning_rate * learning_rate_decay

    if return_all:
        return w, w_all, y, error
    else:
        return w


def train(
    data: Tuple[np.ndarray],
    learning_rate: Union[str, float] = "infer",
    epochs: int = 10,
    normalize_learning_rate: bool = True,
    damping: Union[float, int] = 1,
    early_stopping: bool = True,
    learning_rate_decay: float = 0.9,
    predict_w: bool = False,
    predicted_w_number: int = 7,
    plot: bool = False,
    # random=0, w_rand_scope=1, w_rand_shift=0, rand_seed=0,
):
    """LNU. It's simple one neuron one-step neural net. It can predict not only predictions itself,
    but also use other faster method to predict weights evolution for out of sample predictions.
    In first iteration it will find best learning step and in the second iteration,
    it will train more epochs.

    Args:
        data ((np.ndarray, np.ndarray)) - Tuple (X, y) of input train vectors X and train outputs y
        learning_rate (float, optional): Learning rate. If not normalized must be much smaller.
            If "infer" then it's chosen automatically. Defaults to "infer.
        epochs (int, optional): Number of trained epochs. Defaults to 10.
        normalize_learning_rate (int, optional): Whether normalize learning rate. Defaults to True.
        damping (int, optional):Value of damp of standardized learning rate. Defaults to 1.
        early_stopping (bool, optional): If mean error don't get lower, next epochs are not evaluated. Defaults to True.
        learning_rate_decay (float, optional): With every other epoch, learning rate is a little bit lower
            (learning_rate * learning_rate_decay). It should be between 0 and 1. Defaults to 0.9.
        predict_w (bool): If predict weights with next predictions. Defaults to False.
        predicted_w_number (int, optional): Number of predicted values. Defaults to 7.
        plot (int, optional): Whether plot results. Defaults to False.


    Returns:
        np.ndarray: Weights of neuron that can be used for making predictions.
    """
    if not isinstance(data, (tuple, list)) or len(data) < 2:
        raise TypeError(mylogging.return_str("Data must be in defined shape."))

    X = data[0]
    y_hat = data[1]

    if learning_rate == "infer":
        infer_lightened_params = {
            "model_train": lnu_core,
            "model_predict": predict,
            "kwargs": {
                "learning_rate": 0.0001,
                "epochs": 2,
                "normalize_learning_rate": normalize_learning_rate,
                "damping": damping,
            },
            "model_train_input": (X[-300:-5], y_hat[-300:-5]),
            "model_test_inputs": X[-5:],
            "models_test_outputs": y_hat[-5:],
            "error_criterion": "mape",
            "fragments": 5,
            "iterations": 3,
        }

        # First find order
        learning_rate = optimize(
            kwargs_limits={"learning_rate": [10e-8, 10e-6, 10e-4, 10e-3, 10e-2, 1]},
            **infer_lightened_params
        )["learning_rate"]

        # First around favorite
        learning_rate = optimize(
            kwargs_limits={"learning_rate": [learning_rate / 10, learning_rate * 10]},
            **infer_lightened_params
        )["learning_rate"]

    if plot or predict_w:
        w, w_all, y, error = lnu_core(
            data=data,
            learning_rate=learning_rate,
            epochs=epochs,
            normalize_learning_rate=normalize_learning_rate,
            early_stopping=early_stopping,
            learning_rate_decay=learning_rate_decay,
            damping=damping,
            return_all=True,
        )

    else:
        w = lnu_core(
            data=data,
            learning_rate=learning_rate,
            epochs=epochs,
            normalize_learning_rate=normalize_learning_rate,
            early_stopping=early_stopping,
            learning_rate_decay=learning_rate_decay,
            damping=damping,
        )
    if plot:
        if not misc.GLOBAL_VARS._PLOTS_CONFIGURED:
            misc.setup_plots()

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 7))

        plt.subplot(3, 1, 1)
        plt.plot(y, label="Predictions")
        plt.xlabel("t")
        plt.plot(y_hat, label="Reality")
        plt.xlabel("t")
        plt.ylabel("y")
        plt.legend(loc="upper right")

        plt.subplot(3, 1, 2)
        plt.plot(error, label="Error")
        plt.grid()
        plt.xlabel("t")
        plt.legend(loc="upper right")
        plt.ylabel("Error")

        plt.subplot(3, 1, 3)
        plt.plot(w_all)
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("Weights")

        plt.suptitle("Predictions vs reality, error and weights", fontsize=20)
        plt.subplots_adjust(top=0.88)

        plt.show()

    if predict_w:
        from . import statsmodels_autoregressive

        wwt = np.zeros((X.shape[1], predicted_w_number))

        for i in range(X.shape[1]):
            try:
                wwt[i] = statsmodels_autoregressive.predict(
                    w_all[i],
                    statsmodels_autoregressive.train(w_all[i]),
                    predicts=predicted_w_number,
                )
            except (Exception,):
                wwt[i] = w_all[i][-1]
        w_final = wwt.T

    else:
        w_final = w

    return w_final


def predict(x_input: np.ndarray, model: np.ndarray, predicts: int = 7) -> np.ndarray:
    """Function that creates predictions from trained model and input data.

    Args:
        x_input (np.ndarray): Time series data.
        model (list, class): Trained model. It can be list of neural weights.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    x_input = x_input.ravel().copy()
    w = model

    predictions = []

    for i in range(predicts):

        ww = w[i] if w.ndim == 2 else w
        y_predicted = np.dot(ww, x_input)
        predictions.append(y_predicted)

        x_input[1:-1] = x_input[2:]
        x_input[-1] = y_predicted

    return np.array(predictions).reshape(-1)
