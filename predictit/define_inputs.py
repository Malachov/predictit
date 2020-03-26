from predictit.config import config
import numpy as np


def make_sequences(data, n_steps_in, n_steps_out=1, constant=None, predicted_column_index=0, serialize_columns=1, other_columns_length=None, predicts=1, repeatit=1):
    """Function that create inputs and outputs to models.
    From [1, 2, 3, 4, 5, 6, 7, 8]

        Creates [1, 2, 3]  [4]
                [5, 6, 7]  [8]

    Args:
        data (np.array): Time series data.
        n_steps_in (int): Number of input members.
        n_steps_out (int, optional): Number of output members. For one-step models use 1. Defaults to 1.
        constant (bool, optional): If use bias (add 1 to first place to every member). Defaults to None.
        predicted_column_index (int, optional): If multiavriate data, index of predicted column. Defaults to 0.
        serialize_columns(bool, optional): If multivariate data, serialize columns sequentions into one row.
        other_columns_length (int, optional): Length of non-predicted columns that are evaluated in inputs. If None, than same length as predicted column. Defaults to None.

    Returns:
        np.array, np.array: X and y. Inputs and outputs (that can be used for example in sklearn models).

    """

    if n_steps_out > n_steps_in:
        raise Exception('n_steps_out have to be smaller than n_steps_in!')

    if data.ndim == 2 and data.shape[0] == 1:

        data = data[0]

    shape = data.shape[:-1] + (data.shape[-1] - n_steps_in + 1, n_steps_in)
    strides = data.strides + (data.strides[-1],)
    X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    if data.ndim == 1:

        y = X[n_steps_out:, -n_steps_out:]

    if data.ndim == 2:
        if predicted_column_index != 0:
            X[[0, predicted_column_index], :] = X[[predicted_column_index, 0], :]

        y = X[0][n_steps_out:, -n_steps_out:]

        if serialize_columns:
            if other_columns_length:
                X = np.hstack([X[0, :]] + [X[i, :, -other_columns_length:] for i in range(1, len(X))])
            else:
                X = X.transpose(1, 0, 2).reshape(1, X.shape[1], -1)[0]

        else:
            X = X.transpose(1, 2, 0)

    if constant:
        X = np.hstack([np.ones((len(X), 1)), X])

    x_input = X[-1].reshape(1, -1)

    x_test_inputs = X[-config['predicts'] - config['repeatit']: -config['predicts'], :]
    x_test_inputs = x_test_inputs.reshape(x_test_inputs.shape[0], 1, x_test_inputs.shape[1])

    X = X[: -n_steps_out]

    return X, y, x_input, x_test_inputs


def create_inputs(input_name, data, predicted_column_index, multicolumn, predicts=1, repeatit=1):

    # Take one input type, make all derivated inputs (save memory, because only slices) and create dictionary of inputs for one iteration
    used_sequentions = {}

    if input_name == 'data_one_column' or input_name == 'one_in_one_out_constant' or input_name == 'one_in_one_out':
        data = data[predicted_column_index]

    if input_name == 'data' or input_name == 'data_one_column':
        used_sequentions = data

    else:
        used_sequentions = make_sequences(data, **config['input_types'][input_name])

    return used_sequentions