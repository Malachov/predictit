from sklearn import linear_model
import sklearn
import numpy as np
from sklearn.multioutput import MultiOutputRegressor

from ..data_prep import make_sequences, make_x_input


def get_regressors():
    """ Create list of all regressors from sklearn (classes, that can be called). E.g. [bayes_ridge_regressor, linear_regressor]"""

    from importlib import import_module

    regressors=[]
    for module in sklearn.__all__:
        try:
            module = import_module(f'sklearn.{module}')
            regressors.extend([getattr(module, cls) for cls in module.__all__ if 'Regress' in cls])
        except Exception:
            pass
    regressors.append(sklearn.svm.SVR)

    return regressors


def regression( data, regressor='bayesianridge', n_steps_in=50, predicts=7, predicted_column_index=0, output_shape='one_step',
                other_columns_lenght=None, constant=None, alpha=0.0001, alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6,
                lambda_2=1.e-6, n_iter=300, epsilon=1.35, alphas=[0.1, 0.5, 1], gcv_mode='auto', solver='auto',
                n_hidden=20, rbf_width=0, activation_func='selu'):
    """Sklearn regression model. Regressor is input parameter. It can be linear, or ridge, or Huber. It use function that return
    all existing regressor (with function optimize it automaticcaly find the best one). It also contain extreme learning machine model from sklearn extensions.

    Args:
        data (np.ndarray): Time series data.
        regressor (str, optional): Regressor that sklearn use. Defaults to 'bayesianridge'.
        n_steps_in (int, optional): Number of regressive members entering the model. Defaults to 50.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        predicted_column_index (int, optional): If multidimensional, define what column is predicted. Defaults to 0.
        output_shape (str, optional): Whether one-step or batch evaluation. Defaults to 'one_step'.
        other_columns_lenght (int, optional): Number of members from not-predicted columns. Defaults to None.
        constant (int, optional): Whether use bias. Defaults to 1.
        alpha (float, optional): Parameter of some regressor. Defaults to 0.0001.
        alpha_1 (float, optional): Parameter of some regressor. Defaults to 1.e-6.
        alpha_2 (float, optional): Parameter of some regressor. Defaults to 1.e-6.
        lambda_1 (float, optional): Parameter of some regressor. Defaults to 1.e-6.
        lambda_2 (float, optional): Parameter of some regressor. Defaults to 1.e-6.
        n_iter (int, optional): Parameter of some regressor. Defaults to 300.
        epsilon (float, optional): Parameter of some regressor. Defaults to 1.35.
        alphas (list, optional): Parameter of some regressor. Defaults to [0.1, 0.5, 1].
        gcv_mode (str, optional): Parameter of some regressor. Defaults to 'auto'.
        solver (str, optional): Parameter of some regressor. Defaults to 'auto'.
        n_hidden (int, optional): Parameter of some regressor. Defaults to 20.
        rbf_width (int, optional): Parameter of some regressor. Defaults to 0.
        activation_func (str, optional): Parameter of some regressor. Defaults to 'selu'.

    Returns:
        np.ndarray: Predictions of input time series.
    """

    # Test - Use Multioutput regressor
    multi = 0

    data = np.array(data)
    data_shape = np.array(data).shape

    if data_shape[0] == 1:
        data = data.reshape(-1)
        data_shape = np.shape(data)

    if other_columns_lenght is None:
        other_columns_lenght = n_steps_in

    if output_shape == 'one_step':
        X, y = make_sequences(data, n_steps_in=n_steps_in, predicted_column_index=predicted_column_index, other_columns_lenght=other_columns_lenght, constant=constant)

    if output_shape == 'batch':
        X, y = make_sequences(data, n_steps_in=n_steps_in, n_steps_out=predicts, predicted_column_index=predicted_column_index, other_columns_lenght=other_columns_lenght, constant=constant)


    if regressor == 'bayesianridge' or regressor == 'default':
        reg = linear_model.BayesianRidge(n_iter=n_iter, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)

    elif regressor == 'huber':
        reg = linear_model.HuberRegressor(epsilon=epsilon, max_iter=200, alpha=alpha)

    elif regressor == 'lasso':
        reg = linear_model.Lasso(alpha=alpha)

    elif regressor == 'linear':
        reg = linear_model.LinearRegression()

    elif regressor == 'ridgecv':
        reg = linear_model.RidgeCV(alphas=alphas, gcv_mode=gcv_mode)

    elif regressor == 'ridge':
        reg = linear_model.Ridge(alpha=alpha, solver=solver)

    elif regressor == 'elm':
        from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
        reg = ELMRegressor(n_hidden=n_hidden, alpha=alpha, rbf_width=rbf_width, activation_func=activation_func)

    elif regressor == 'elm_gen':
        from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor
        reg = GenELMRegressor()

    else:
        reg = regressor()

    if multi:
        reg = MultiOutputRegressor(reg)

    reg.fit(X, y)

    # For one column data
    if len(data_shape) == 1:

        if output_shape == 'one_step':

            predictions = []
            x_input = make_x_input(data, n_steps_in=n_steps_in, constant=constant)

            for i in range(predicts):

                yhat = reg.predict(x_input)

                x_input = np.insert(x_input, n_steps_in, yhat[0], axis=1)
                x_input = np.delete(x_input, 0, axis=1)
                predictions.append(yhat[0])

        if output_shape == 'batch':

            x_input = make_x_input(data, n_steps_in=n_steps_in, constant=constant)

            predictions = reg.predict(x_input)
            predictions = predictions[0]

    else:

        if output_shape == 'one_step':

            from . import ar

            predictions = []
            nu_data_shape = data.shape

            for i in range(predicts):

                x_input = make_x_input(data, n_steps_in=n_steps_in, predicted_column_index=predicted_column_index, other_columns_lenght=other_columns_lenght, constant=constant)

                nucolumn = []
                for_prediction = data[predicted_column_index]

                yhat = reg.predict(x_input)
                yhat_flat = yhat[0]
                predictions.append(yhat_flat)

                for_prediction = np.append(for_prediction, yhat)

                for j in data:
                    new = ar(j, predicts=1)
                    nucolumn.append(new)

                nucolumn_T = np.array(nucolumn).reshape(nu_data_shape[0], 1)
                data = np.append(data, nucolumn_T, axis=1)
                data[predicted_column_index] = for_prediction

        if output_shape == 'batch':

            x_input = make_x_input(data, n_steps_in=n_steps_in, predicted_column_index=predicted_column_index, other_columns_lenght=other_columns_lenght, constant=constant)

            predictions = reg.predict(x_input)
            predictions = predictions[0]

    predictions = np.array(predictions).reshape(-1)

    return predictions
