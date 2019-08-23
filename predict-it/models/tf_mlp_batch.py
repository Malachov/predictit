from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from data_prep import make_sequences
from pathlib import Path
import os

def mlp_batch(data, n_steps, n_features=1, predicts=7, epochs=200,units=32, save=1, already_trained=0, optimizer='adam', loss='mse', verbose=0, activation='relu', metrics='acc', timedistributed=0):

    script_dir = os.path.dirname(__file__)
    folder = Path('stored_models')
    modelname = 'lstmvanilla.h5'
    model_path = script_dir / folder / modelname
    model_path_string = model_path.as_posix()

    X, y = make_sequences(data, n_steps_in, predicts)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    x_input = data[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps_in, n_features))

    if already_trained is not 1:
        model = Sequential()

        model.add(Dense(units, activation=activation, input_dim=input_len, metrics=metrics))
        if dropout is not -1:
            model.add(Dropout(dropout))
        if dropout is not -1:
            model.add(Dropout(dropout))

        for i in range(layers):
            model.add(Dense(units, activation=activation))

        if timedistributed == 1:
            model.add(TimeDistributed(Dense(1)))
        else:
            model.add(Dense(predicts))

        model.compile(optimizer=optimizer, loss=loss)
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

    x_input = data[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    predictions = model.predict(x_input, verbose=verbose)
    predictions = predictions[0]

    return predictions