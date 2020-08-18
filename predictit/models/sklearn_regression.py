import sklearn
import sklearn.ensemble
import sklearn.multioutput
import sklearn.tree
import sklearn.neighbors
import numpy as np
import sklearn_extensions.extreme_learning_machines.elm as elm

linear_model = sklearn.linear_model


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


def train(sequentions, regressor='bayesianridge', predicts=7, load_trained_model=0, update_trained_model=1, save_model=1,
          saved_model_path_string='stored_models', n_estimators=100,
          alpha=0.0001, alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6,
          lambda_2=1.e-6, n_iter=300, epsilon=1.35, alphas=[0.1, 0.5, 1], gcv_mode='auto', solver='auto',
          n_hidden=20, rbf_width=0, activation_func='selu', **kwargs):
    """Sklearn regression model. Regressor is input parameter. It can be linear, or ridge, or Huber. It use function that return
    all existing regressor (with function optimize it automaticcaly find the best one). It also contain extreme learning machine model from sklearn extensions.

    Args:
        sequentions (tuple(np.ndarray, np.ndarray, np.ndarray)) - Tuple (X, y, x_input) of input train vectors X, train outputs y, and input for prediction x_input
        regressor (str, optional): Regressor that sklearn use. Defaults to 'bayesianridge'.
        predicts (int, optional): Number of predicted values. Defaults to 7.
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

    if regressor == 'bayesianridge':
        model = linear_model.BayesianRidge(n_iter=n_iter, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
    elif regressor == 'huber':
        model = linear_model.HuberRegressor(epsilon=epsilon, max_iter=200, alpha=alpha)
    elif regressor == 'lasso':
        model = linear_model.Lasso(alpha=alpha)
    elif regressor == 'linear':
        model = linear_model.LinearRegression()
    elif regressor == 'ridgecv':
        model = linear_model.RidgeCV(alphas=alphas, gcv_mode=gcv_mode)
    elif regressor == 'ridge':
        model = linear_model.Ridge(alpha=alpha, solver=solver)
    elif regressor == 'Extra trees':
        model = sklearn.ensemble.ExtraTreesRegressor(n_estimators=n_estimators)
    elif regressor == 'Random forest':
        model = sklearn.ensemble.RandomForestRegressor()
    elif regressor == 'Decision tree':
        model = sklearn.tree.DecisionTreeRegressor()
    elif regressor == 'Gradient boosting':
        model = sklearn.ensemble.GradientBoostingRegressor()
    elif regressor == 'KNeighbors':
        model = sklearn.neighbors.KNeighborsRegressor()
    elif regressor == 'Bagging':
        model = sklearn.ensemble.BaggingRegressor()
    elif regressor == 'Stochastic gradient':
        model = sklearn.linear_model.SGDRegressor()
    elif regressor == 'Passive aggressive regression':
        model = sklearn.linear_model.PassiveAggressiveRegressor()
    elif regressor == 'elm':
        model = elm.ELMRegressor(n_hidden=n_hidden, alpha=alpha, rbf_width=rbf_width, activation_func=activation_func)
    elif regressor == 'elm_gen':
        model = elm.GenELMRegressor()

    if sequentions[1].shape[1] == 1:
        model.output_shape = 'one_step'

    else:
        model = sklearn.multioutput.MultiOutputRegressor(model)
        model.output_shape = 'batch'

    model.fit(sequentions[0], sequentions[1])

    return model


def predict(x_input, model, predicts=7):
    """Function that creates predictions from trained model and input data.

    Args:
        data (np.ndarray): Time series data
        model (list, class): Trained model. It can be list of neural weigths or it can be fitted model class from imported library.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    if model.output_shape == 'one_step':

        predictions = []

        for _ in range(predicts):

            yhat = model.predict(x_input)
            x_input = np.insert(x_input, x_input.shape[1], yhat[0], axis=1)
            x_input = np.delete(x_input, 0, axis=1)
            predictions.append(yhat[0])

    else:

        predictions = model.predict(x_input)
        predictions = predictions[0]

    predictions = np.array(predictions).reshape(-1)

    return predictions
