from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from ..data_prep import make_sequences

def lstm_stacked_batch(data, n_steps_in=50, n_features=1, predicts=7, layers=2, epochs=100, units=32, optimizer='adam', loss='mse', verbose=0, activation='relu', activation_out='relu', stateful=False, metrics='acc', timedistributed=0):
    
    X, y = makeequences(data, n_steps_in=n_steps_in, predicts=predicts)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    model = Sequential()

    model.add(LSTM(units, activation=activation, return_sequences=True, input_shape=(n_steps_in, n_features), statful=stateful))
    
    for i in range(layers):
        model.add(LSTM(units, return_sequences=True, statful=stateful))
    model.add(LSTM(units, activation=activation, statful=stateful))

    if timedistributed == 1:
        model.add(TimeDistributed(Dense(1)))
    else:
        model.add(Dense(predicts))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(X, y, epochs=epochs, verbose=verbose, batch_size=predicts)

    x_input = data[-n_steps_in:]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    predictions = model.predict(x_input, verbose=verbose)
    predictions = predictions[0]

    return predictions
    