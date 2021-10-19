"""Tensorflow model, where you can define structure of neural net with function parameters that you can optimize afterwards."""
from __future__ import annotations
from typing import cast, Any
from pathlib import Path
import importlib.util
import os

from typing_extensions import Literal

import numpy as np

import mylogging

from .models_functions.models_functions import get_inputs


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# TODO test stateful=True on LSTM


def train(
    data: tuple[np.ndarray, np.ndarray],
    layers: Literal["lstm", "mlp"] | list[tuple[str, dict]] = "mlp",
    epochs: int = 100,
    load_trained_model: bool = True,
    update_trained_model: bool = True,
    save_model: bool = True,
    saved_model_path_string: str = "stored_models",
    optimizer: str = "adam",
    loss: str = "mse",
    summary: bool = False,
    verbose=0,
    used_metrics="accuracy",
    timedistributed=False,
    batch_size=64,
):
    """Tensorflow model. Neural nets - LSTM or MLP (dense layers). Layers are customizable with arguments.

    Args:
        data (tuple[np.ndarray, np.ndarray]) - Tuple (X, y) of input train vectors X and train outputs y
        layers (Literal["lstm", "mlp"] | list[tuple[str, dict]], optional) - List of tuples of layer name (e.g. 'lstm') and layer params dict e.g.
            (("lstm", {"units": 7, "activation": "relu"})). Check default layers list here for example.
            There are also some predefined architectures. You can use 'lstm' or 'mlp'. Defaults to 'mlp'.
        epochs (int, optional): Number of epochs to evaluate. Defaults to 100.
        load_trained_model (bool, optional): If True, load model from disk. Most of time is spend
            on training, so if loaded and not updated, it's very fast. Defaults to True.
        update_trained_model (bool, optional): Whether load_trained_model, it's updated with new input.
            Defaults to True.
        save_model (str, optional): If True, save model on disk on saved_model_path_string. Defaults to True.
        saved_model_path_string (str, optional): Full path to saved model with name. E.g. '/home/dan/mymodel.h5.
        If 'stored_models', then it's save to library folder models/stored_models. Defaults to 'stored_models'.
        optimizer (str, optional): Used optimizer. Defaults to 'adam'.
        loss (str, optional): Loss function. Defaults to 'mse'.
        summary (int, optional): Display model details table. Defaults to 0.
        verbose (int, optional): Whether display progress bar. Defaults to 0.
        used_metrics (str, optional): Used metrics. 'accuracy' or 'mape' Defaults to 'accuracy'.
        timedistributed (bool, optional): Whether add time distributed layer. Defaults to False.
        batch_size (int, optional): Used batch size. Defaults to 64.

    Returns:
        model: Trained model object.
    """

    if not importlib.util.find_spec("tensorflow"):
        raise ModuleNotFoundError(
            mylogging.return_str(
                "Tensorflow model configured, but tensorflow library not installed. It's not "
                "in general requirements, because very big and not work everywhere. If you "
                "want to use tensorflow model, install it via \n\n`pip install tensorflow`"
            )
        )

    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras import layers as tf_layers
    from tensorflow.keras import metrics as tf_metrics
    from tensorflow.keras import models as tf_models
    from tensorflow.keras import Model as tf_model_type

    X, y = get_inputs(data)

    X_ndim = X.ndim

    models = {
        "dense": tf_layers.Dense,
        "lstm": tf_layers.LSTM,
        "mlp": tf_layers.Dense,
        "gru": tf_layers.GRU,
        "conv2d": tf_layers.Conv2D,
        "rnn": tf_layers.SimpleRNN,
        "convlstm2d": tf_layers.ConvLSTM2D,
        "dropout": tf_layers.Dropout,
        "batchnormalization": tf_layers.BatchNormalization,
    }

    if used_metrics == "accuracy":
        metrics = [tf_metrics.Accuracy()]
    elif used_metrics == "mape":
        metrics = [tf_metrics.MeanAbsolutePercentageError()]
    else:
        raise ValueError("metrics has to be one from ['accuracy', 'mape']")

    if saved_model_path_string == "stored_models":
        saved_model_path_string = str(Path(__file__).resolve().parent / "stored_models" / "tensorflow.h5")

    if load_trained_model:
        try:
            model = tf_models.load_model(saved_model_path_string)
            model = cast(tf_model_type, model)
            model.load_weights(saved_model_path_string)

        except Exception:
            raise NameError("Model is not saved, first save_model = 1 in config")

        if update_trained_model:
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    else:

        if isinstance(layers, str):
            if layers == "lstm":
                layers = [
                    ("lstm", {"units": 32, "activation": "relu", "return_sequences": 1}),
                    ("dropout", {"rate": 0.1}),
                    ("lstm", {"units": 7, "activation": "relu"}),
                ]

            elif layers == "mlp":
                layers = [
                    ("dense", {"units": 32, "activation": "relu"}),
                    ("dropout", {"rate": 0.1}),
                    ("dense", {"units": 7, "activation": "relu"}),
                ]

            else:
                raise ValueError(
                    mylogging.return_str("Only possible predefined layers are 'lstm' and 'mlp'.")
                )

            layers = cast(list[tuple[str, dict[str, Any]]], layers)

        if layers[0][0] == "lstm":
            if X.ndim == 2:
                X = X.reshape(X.shape[0], X.shape[1], 1)
            layers[0][1]["input_shape"] = (X.shape[1], X.shape[2])

        elif layers[0][0] == "dense":
            layers[0][1]["input_shape"] = (X.shape[1],)
            if X.ndim > 2:
                raise ValueError(
                    mylogging.return_str(
                        "For dense first layer only univariate data supported (e.g. shape = (n_samples, n_features))"
                        "if ndim > 2: serialize first."
                    )
                )

        model = Sequential()

        for i in layers:
            model.add(models[i[0]](**i[1]))

        if timedistributed == 1:
            model.add(tf_layers.TimeDistributed(tf_layers.Dense(y.shape[1])))
        else:
            model.add(tf_layers.Dense(y.shape[1]))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if summary:
            model.summary()

        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    if save_model == 1:
        model.save(saved_model_path_string)

    model.layers_0_0 = layers[0][0]

    model.X_ndim = X_ndim
    model.y_shape_1 = y.shape[1]

    return model


def predict(x_input, model, predicts):
    """Function that creates predictions from trained model and input data.

    Args:
        x_input (np.ndarray): Time series data
        model (model): Trained model object from train function.
        predicts (Any): Just for other models consistency.

    Returns:
        np.ndarray: Array of predicted results
    """

    if model.X_ndim == 2 and model.layers_0_0 == "lstm":
        x_input = x_input.reshape(1, x_input.shape[1], x_input.shape[0])

    return model.predict(x_input)[0]


def get_optimizers_loses_activations():
    """Return list of tensorflow optimizers. It's used by optimize function.

    Returns:
        list: List of tensorflow optimizers.
    """

    if not importlib.util.find_spec("tensorflow"):
        raise ModuleNotFoundError(
            mylogging.return_str(
                "Tensorflow model configured, but tensorflow library not installed. It's not "
                "in general requirements, because very big and not work everywhere. If you "
                "want to use tensorflow model, install it via \n\n`pip install tensorflow`"
            )
        )

    import tensorflow as tf

    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0)
    adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.01, epsilon=None, decay=0.0)
    adadelta = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=None, decay=0.0)
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False,
    )
    adamax = tf.keras.optimizers.Adamax(
        learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0
    )
    nadam = tf.keras.optimizers.Nadam(
        learning_rate=0.002,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        schedule_decay=0.004,
    )

    return [sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam]
