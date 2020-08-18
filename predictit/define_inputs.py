import numpy as np

from predictit.misc import user_warning


def make_sequences(data, config, n_steps_in, n_steps_out=1, constant=None, predicted_column_index=0, serialize_columns=1, default_other_columns_length=None):
    """Function that create inputs and outputs to models.

    Example for n_steps_in = 3 and n_steps_out = 1

    From [[1, 2, 3, 4, 5, 6, 7, 8]]

        Creates [1, 2, 3]  [4]
                [5, 6, 7]  [8]

    Args:
        data (np.array): Time series data.
        n_steps_in (int): Number of input members.
        n_steps_out (int, optional): Number of output members. For one-step models use 1. Defaults to 1.
        constant (bool, optional): If use bias (add 1 to first place to every member). Defaults to None.
        predicted_column_index (int, optional): If multiavriate data, index of predicted column. Defaults to 0.
        serialize_columns(bool, optional): If multivariate data, serialize columns sequentions into one row.
        default_other_columns_length (int, optional): Length of non-predicted columns that are evaluated in inputs. If None, than same length as predicted column. Defaults to None.

    Returns:
        np.array, np.array: X and y. Inputs and outputs (that can be used for example in sklearn models).

    """

    if n_steps_out > n_steps_in:
        n_steps_in = n_steps_out + 1
        user_warning('n_steps_out was bigger than n_steps_in - n_steps_in changed during prediction!')

    if default_other_columns_length == 0:
        data = data[:, 0].reshape(1, -1)

    data_T = data.T
    shape = data_T.shape[:-1] + (data_T.shape[-1] - n_steps_in + 1, n_steps_in)
    strides = data_T.strides + (data_T.strides[-1],)
    X = np.lib.stride_tricks.as_strided(data_T, shape=shape, strides=strides)

    if predicted_column_index != 0:
        X[[0, predicted_column_index], :] = X[[predicted_column_index, 0], :]

    y = X[0][n_steps_out:, -n_steps_out:]

    if serialize_columns:
        if default_other_columns_length:
            X = np.hstack([X[0, :]] + [X[i, :, -default_other_columns_length:] for i in range(1, len(X))])
        else:
            X = X.transpose(1, 0, 2).reshape(1, X.shape[1], -1)[0]

    else:
        X = X.transpose(1, 2, 0)

    if constant:
        X = np.hstack([np.ones((len(X), 1)), X])

    if X.ndim == 3:
        x_input = X[-1].reshape(1, X.shape[1], X.shape[2])
        x_test_inputs = X[-config.predicts - config.repeatit: -config.predicts, :, :]
        x_test_inputs = x_test_inputs.reshape(x_test_inputs.shape[0], 1, x_test_inputs.shape[1], x_test_inputs.shape[2])

    else:
        x_input = X[-1].reshape(1, -1)

        x_test_inputs = X[-config.predicts - config.repeatit: -config.predicts, :]
        x_test_inputs = x_test_inputs.reshape(x_test_inputs.shape[0], 1, x_test_inputs.shape[1])

    X = X[: -n_steps_out]

    return X, y, x_input, x_test_inputs


def create_inputs(input_name, data, config, predicted_column_index):

    # Take one input type, make all derivated inputs (save memory, because only slices) and create dictionary of inputs for one iteration
    used_sequentions = {}

    if input_name == 'data':
        used_sequentions = data

    elif input_name == 'data_one_column':
        used_sequentions = data[:, predicted_column_index]

    else:
        if input_name in ['one_in_one_out_constant', 'one_in_one_out', 'one_in_batch_out']:
            used_sequentions = data[:, predicted_column_index: predicted_column_index + 1]
        else:
            used_sequentions = data

        used_sequentions = make_sequences(used_sequentions, config, **config.input_types[input_name])

    # data_end = -config.predicts - config.repeatit - config.validation_gap if config.mode == 'validate' else None

    if isinstance(used_sequentions, tuple):
        model_train_input = (used_sequentions[0], used_sequentions[1])
        model_predict_input = used_sequentions[2]
        if config.mode == 'validate':
            model_test_inputs = [model_predict_input]
        else:
            model_test_inputs = used_sequentions[3]

    else:
        model_train_input = model_predict_input = used_sequentions
        if config.mode == 'validate':
            model_test_inputs = [model_predict_input]
        else:
            model_test_inputs = []
            if used_sequentions.ndim == 1:
                for i in range(config.repeatit):
                    model_test_inputs.append(used_sequentions[: - config.predicts - config.repeatit + i + 1])
            else:
                for i in range(config.repeatit):
                    model_test_inputs.append(used_sequentions[:, : - config.predicts - config.repeatit + i + 1])

    return model_train_input, model_predict_input, model_test_inputs
