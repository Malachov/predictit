import numpy as np
from pathlib import Path
from predictit.misc import traceback_warning


def train(sequentions, layers='default', predicts=7, epochs=200, already_trained=0, train_next=1, save=1, saved_model_path_string='stored_models', optimizer='adam', loss='mse', verbose=0, metrics='accuracy', timedistributed=0, batch_size=64):
    """Tensorflow LSTM model.

    Args:
        sequentions (tuple(np.ndarray, np.ndarray, np.ndarray)) - Tuple (X, y, x_input) of input train vectors X, train outputs y, and input for prediction x_input
        model (str) - Type of neural network. 'lstm', 'bidirectional_lstm' or 'mlp'. Defaults to 'lstm'.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        epochs (int, optional): Number of epochs to evaluate. Defaults to 100.
        units (int, optional): Number of neural units. Defaults to 50.
        layers (int): Number ob neuron layers. Defaults to 3.
        dropout (int, optional): Whether use dropout layer. Use values around 0.1. If 0, no dropout. Defaults to 0.
        timedistributed (int, optional): Whether use timedistributed layer. Defaults to 0.
        optimizer (str, optional): Used optimizer. Defaults to 'adam'.
        loss (str, optional): Loss function. Defaults to 'mse'.
        verbose (int, optional): Wheter display details. Defaults to 0.
        activation (str, optional): Used activation function. Defaults to 'relu'.
        metrics (str, optional): Used metrics. Defaults to 'accuracy'.

    Layers kwargs:
        BatchNormalization: momentum=0.99, epsilon=0.001
    Returns:
        np.ndarray: Predictions of input time series.
    """

    import keras
    import tensorflow as tf

    if layers == 'default':
        layers = [['lstm', {'units': 32, 'activation': 'relu', 'return_sequences': 1}],
                  ['dropout', {'rate': 0.1}],
                  ['lstm', {'units': 7, 'activation': 'relu'}]]

    X = sequentions[0]
    y = sequentions[1]
    X_ndim = X.ndim

    models = {'dense': keras.layers.Dense, 'lstm': keras.layers.LSTM, 'mlp': keras.layers.Dense, 'gru': keras.layers.GRU, 'conv2d': keras.layers.Conv2D, 'rnn': keras.layers.SimpleRNN, 'convlstm2d': keras.layers.ConvLSTM2D, 'dropout': keras.layers.Dropout, 'batchnormalization': keras.layers.BatchNormalization}

    if metrics == 'accuracy':
        metrics = [tf.keras.metrics.Accuracy()]

    if layers[0][0] == 'lstm':
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        layers[0][1]['input_shape'] = (X.shape[1], X.shape[2])

    if layers[0][0] == 'dense':
        layers[0][1]['input_shape'] = (X.shape[1],)
        assert (X.ndim != 3), 'For dense first layer only univariate data supported (e.g. shape = (n_samples, n_features))- if ndim > 2: serialize first.'

    if saved_model_path_string == 'stored_models':
        saved_model_path_string = str(Path(__file__).resolve().parents[0] / 'stored_models' / 'tensorflow.h5')

    if not already_trained:

        model = keras.models.Sequential()

        for i, j in enumerate(layers):

            model.add(models[j[0]](**j[1] if len(j) > 1 else {}))

        if timedistributed == 1:
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(y.shape[1])))
        else:
            model.add(keras.layers.Dense(y.shape[1]))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(X, y, epochs=epochs, batch_size=64, verbose=verbose)

        if save == 1:
            model.save(saved_model_path_string)

    if already_trained == 1:
        try:
            model = tf.keras.models.load_model(saved_model_path_string)
            model.load_weights(saved_model_path_string)
            if train_next:
                model.fit(X, y, epochs=epochs, batch_size=64, verbose=verbose)

        except Exception:
            raise NameError("Model is not saved, first saveit = 1 in config")

    model.layers_0_0 = layers[0][0]

    model.X_ndim = X_ndim
    model.y_shape_1 = y.shape[1]

    return model


def predict(x_input, model, predicts=7):

    if model.X_ndim == 2 and model.layers_0_0 == 'lstm':
        x_input = x_input.reshape(1, x_input.shape[1], x_input.shape[0])
    predictions = []

    ### One step univariate prediction
    if model.y_shape_1 == 1:
        for i in range(predicts):
            yhat = model.predict(x_input)
            x_input = np.insert(x_input, x_input.shape[1], yhat[0][0], axis=1)
            x_input = np.delete(x_input, 0, axis=1)
            predictions.append(yhat[0][0])
        predictions = np.array(predictions)

    ### Batch multivariate prediction
    else:
        predictions = model.predict(x_input)[0]

    return predictions


def get_optimizers_loses_activations():

    from keras import optimizers

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    lstm_optimizers = [sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam]

    return lstm_optimizers
