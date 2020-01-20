from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from ..data_prep import make_sequences

def lstm_stacked_batch(data, n_steps_in=50, n_features=1, predicts=7, layers=2, epochs=100, units=32, optimizer='adam', loss='mse', verbose=0, activation='relu', activation_out='relu', stateful=False, metrics='acc', timedistributed=0):
    """Tensorflow stacked batch LSTM model.

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
        activation_out (str, optional): Used output activation function. Defaults to 'relu'.
        stateful (bool, optional): Whether layers are stateful. Defaults to False.
        metrics (str, optional): Used metrics. Defaults to 'acc'.
        timedistributed (int, optional): Whether use timedistributed layer. Defaults to 0.

    Returns:
        np.ndarray: Time series predictions.
    """

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
