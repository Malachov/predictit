from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import numpy as np

from predictit.data_prep import make_sequences

def lstm_bidirectional(data, n_steps_in= 50, n_features = 1, predicts = 7, epochs = 100, units=50, optimizer = 'adam', loss='mse', verbose=0, activation='relu', dropout=0, metrics='acc', timedistributed=0):
    """
    metrics=['mae', 'acc'])
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