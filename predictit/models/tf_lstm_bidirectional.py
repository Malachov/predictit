from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import numpy as np

from ..data_prep import make_sequences


def lstm_bidirectional(data, n_steps_in= 50, n_features = 1, predicts = 7, epochs = 100, units=50, optimizer = 'adam', loss='mse', verbose=0, activation='relu', dropout=0, metrics='acc', timedistributed=0):
    """Tensorflow LSTM model.

    Args:
        data (np.ndarray): Time series data.
        n_steps_in (int, optional): Number of regressive members - inputs to neural unit. Defaults to 50.
        n_features (int, optional): Number of columns. Defaults to 1.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        epochs (int, optional): Number of epochs to evaluate. Defaults to 100.
        units (int, optional): Number of neural units. Defaults to 50.
        optimizer (str, optional): Used optimizer. Defaults to 'adam'.
        loss (str, optional): Loss function. Defaults to 'mse'.
        verbose (int, optional): Wheter display details. Defaults to 0.
        activation (str, optional): Used activation function. Defaults to 'relu'.
        dropout (int, optional): Whether use dropout layer. Defaults to -1.
        metrics (str, optional): Used metrics. Defaults to 'acc'.
        timedistributed (int, optional): Whether use timedistributed layer. Defaults to 0.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    X, y = make_sequences(data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    x_input = data[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps, 1))

    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation=activation), input_shape=(n_steps, n_features)))
    if dropout is not -1:
        model.add(Dropout(dropout))

    if timedistributed == 1:
        model.add(TimeDistributed(Dense(1)))
    else:
        model.add(Dense(1))
        
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(X, y, epochs=epochs, verbose=0)

    predictions = []

    for i in range(predicts):
        yhat = model.predict(x_input, verbose=verbose)
        x_input = np.insert(x_input, n_steps, yhat[0][0], axis=1)
        x_input = np.delete(x_input,0, axis=1 )
        predictions.append(yhat[0][0])
    predictionsarray = np.array(predictions)

    return predictionsarray