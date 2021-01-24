import numpy as np
from pathlib import Path
from mylogging import user_warning

from .models_functions.models_functions import one_step_looper


def train(data, layers='default', epochs=200, load_trained_model=0, update_trained_model=1, save_model=1,
          saved_model_path_string='stored_models', optimizer='adam', loss='mse', verbose=0, used_metrics='accuracy', timedistributed=0, batch_size=64):
    """Tensorflow model. Neural nets - LSTM or MLP (dense layers). Layers are customizable with arguments.

    Args:
        data ((np.ndarray, np.ndarray)) - Tuple (X, y) of input train vectors X and train outputs y
        layers (tuple) - Tuple of tuples of layer name (e.g. 'lstm') and layer params dict (e.g. {'units': 7, 'activation': 'relu'}). Check default layres tuple here or config example.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        epochs (int, optional): Number of epochs to evaluate. Defaults to 100.
        load_trained_model: If 1, load model from disk. Most of time is spend on training, so if loaded and not updated, it's very fast.
        update_trained_model: If load_trained_model, it's updated with new input.
        save_model: If 1, save model on disk on saved_model_path_string.
        saved_model_path_string: Full path to saved model with name. E.g. '/home/dan/mymodel.h5. If 'stored_models', then it's save to library folder models/stored_models.
        optimizer (str, optional): Used optimizer. Defaults to 'adam'.
        loss (str, optional): Loss function. Defaults to 'mse'.
        verbose (int, optional): Wheter display details. Defaults to 0.
        used_metrics (str, optional): Used metrics. 'accuracy' or 'mape' Defaults to 'accuracy'.

    Returns:
        model: Trained model object.
    """

    try:
        import tensorflow as tf
    except Exception:
        user_warning("Tensorflow model configured, but tensorflow library not installed. It's not "
                     "in general requirements, because very big and not work everywhere. If you "
                     "want to use tensorflow model, install it via \n\npip install tensorflow")
        return

    if layers == 'default':
        layers = (('lstm', {'units': 32, 'activation': 'relu', 'return_sequences': 1}),
                  ('dropout', {'rate': 0.1}),
                  ('lstm', {'units': 7, 'activation': 'relu'}))

    X = data[0]
    y = data[1]
    X_ndim = X.ndim

    models = {'dense': tf.keras.layers.Dense, 'lstm': tf.keras.layers.LSTM, 'mlp': tf.keras.layers.Dense, 'gru': tf.keras.layers.GRU, 'conv2d': tf.keras.layers.Conv2D, 'rnn': tf.keras.layers.SimpleRNN, 'convlstm2d': tf.keras.layers.ConvLSTM2D, 'dropout': tf.keras.layers.Dropout, 'batchnormalization': tf.keras.layers.BatchNormalization}

    if used_metrics == 'accuracy':
        metrics = [tf.keras.metrics.Accuracy()]
    elif used_metrics == 'mape':
        metrics = [tf.keras.metrics.MeanAbsolutePercentageError()]
    else:
        raise ValueError("metrics has to be one from ['accuracy', 'mape']")

    if saved_model_path_string == 'stored_models':
        saved_model_path_string = str(Path(__file__).resolve().parent / 'stored_models' / 'tensorflow.h5')

    if load_trained_model:
        try:
            model = tf.keras.models.load_model(saved_model_path_string)
            model.load_weights(saved_model_path_string)

        except Exception:
            raise NameError("Model is not saved, first saveit = 1 in config")

        if update_trained_model:
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    else:
        if layers[0][0] == 'lstm':
            if X.ndim == 2:
                X = X.reshape(X.shape[0], X.shape[1], 1)
            layers[0][1]['input_shape'] = (X.shape[1], X.shape[2])

        if layers[0][0] == 'dense':
            layers[0][1]['input_shape'] = (X.shape[1],)
            assert (X.ndim != 3), 'For dense first layer only univariate data supported (e.g. shape = (n_samples, n_features))- if ndim > 2: serialize first.'

        model = tf.keras.models.Sequential()

        for i, j in enumerate(layers):

            model.add(models[j[0]](**j[1] if len(j) > 1 else {}))

        if timedistributed == 1:
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(y.shape[1])))
        else:
            model.add(tf.keras.layers.Dense(y.shape[1]))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    if save_model == 1:
        model.save(saved_model_path_string)

    model.layers_0_0 = layers[0][0]

    model.X_ndim = X_ndim
    model.y_shape_1 = y.shape[1]

    return model


def predict(x_input, model, predicts=7):
    """Function that creates predictions from trained model and input data.

    Args:
        x_input (np.ndarray): Time series data
        model (model): Trained model object from train function.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    if model.X_ndim == 2 and model.layers_0_0 == 'lstm':
        x_input = x_input.reshape(1, x_input.shape[1], x_input.shape[0])
    predictions = []

    predictions = model.predict(x_input)[0]

    return predictions


def get_optimizers_loses_activations():
    """Return list of tensorflow optimizers. It's used by optimize function.

    Returns:
        list: List of tensorflow optimizers.
    """

    import tensorflow as tf

    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    adagrad = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    adadelta = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adamax = tf.keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    nadam = tf.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    return [sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam]
