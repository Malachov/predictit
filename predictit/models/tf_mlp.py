from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from pathlib import Path
import os

from ..data_prep import make_sequences

def mlp(data, n_steps, n_features=1, predicts=7, layers=3, epochs=100, units=32, save=1, already_trained=0, optimizer='adam', loss='mse', verbose=0, activation='relu', dropout=0.5, metrics='acc', timedistributed=0):

    script_dir = os.path.dirname(__file__)
    folder = Path('stored_models')
    modelname = 'lstmvanilla.h5'
    model_path = script_dir / folder / modelname
    model_path_string = model_path.as_posix()

    X, y = make_sequences(data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    x_input = data[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps, 1))
    input_len = len(X[0])

    if already_trained is not 1:
        model = Sequential()
        model.add(Dense(units, activation=activation, input_dim=input_len))
        if dropout is not -1:
            model.add(Dropout(dropout))

        for i in range(layers):
            model.add(Dense(units, activation=activation))
            if dropout is not -1:
                model.add(Dropout(dropout))

        if timedistributed == 1:
            model.add(TimeDistributed(Dense(1)))
        else:
            model.add(Dense(1))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(X, y, epochs=epochs, verbose=verbose)

        if save == 1:
            model.save(model_path_string)

    if already_trained == 1:
        try:
            model = tf.keras.models.load_model(model_path_string)
            model.load_weights(model_path_string)
        except:
            print("Model is not saved, first saveit = 1 in config")

    predictions = []

    for i in range(predicts):
        yhat = model.predict(x_input, verbose=verbose)
        x_input = np.insert(x_input, n_steps, yhat[0][0], axis=1)
        x_input = np.delete(x_input,0, axis=1)
        predictions.append(yhat[0][0])
    predictionsarray = np.array(predictions)
    return predictionsarray