
from data_prep import make_sequences
import numpy as np


def pokus(data, n_steps_in=50, n_features=1, predicts=7, epochs=100, units=50, optimizer='adam', loss='mse', verbose=0, activation='relu'):

    X, y = make_sequences(data, n_steps)

    return X
