from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from pathlib import Path
import os

from predictit.data_prep import make_sequences

def lstm(data, n_steps, n_features=1, predicts=7, epochs=200, units=50, save=1, already_trained=0, optimizer='adam', loss='mse', verbose=0, activation='relu', metrics='mape', timedistributed=0):

    # TODO batch and one-step in one model as parameter
    # TODO batch lstm.predict_sequencies_multiple(model, x_test, 50, 50)
    # TODO tf batch, in fit batch_size=predicts test
    # TODO tf performance difference input_shape=(x, ) and input_dim=x

    script_dir = os.path.dirname(__file__)
    folder = Path('stored_models')
    modelname = 'lstmvanilla.h5'
    model_path = script_dir / folder / modelname
    model_path_string = model_path.as_posix()

    X, y = make_sequences(data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    x_input = data[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps, 1))

    if already_trained is not 1:
        model = Sequential()
        model.add(LSTM(units, activation=activation, input_shape=(n_steps, n_features)))

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