import sklearn
from sklearn import linear_model
import numpy as np
import sklearn_extensions.extreme_learning_machines.elm as elm


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


def train(sequentions, regressor='bayesianridge', predicts=7, n_estimators=100,
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

    regressor_dict = {
        'default': linear_model.BayesianRidge(n_iter=n_iter, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2),
        'bayesianridge': linear_model.BayesianRidge(n_iter=n_iter, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2),
        'huber': linear_model.HuberRegressor(epsilon=epsilon, max_iter=200, alpha=alpha),
        'lasso': linear_model.Lasso(alpha=alpha),
        'linear': linear_model.LinearRegression(),
        'ridgecv': linear_model.RidgeCV(alphas=alphas, gcv_mode=gcv_mode),
        'ridge': linear_model.Ridge(alpha=alpha, solver=solver),
        'Extra trees': sklearn.ensemble.ExtraTreesRegressor(n_estimators=n_estimators),
        'elm': elm.ELMRegressor(n_hidden=n_hidden, alpha=alpha, rbf_width=rbf_width, activation_func=activation_func),
        'elm_gen': elm.GenELMRegressor(),
        'Random forest': sklearn.ensemble.forest.RandomForestRegressor(),
        'Decision tree': sklearn.tree.tree.DecisionTreeRegressor(),
        'Gradient boosting': sklearn.ensemble.gradient_boosting.GradientBoostingRegressor(),
        'KNeighbors': sklearn.neighbors.regression.KNeighborsRegressor(),
        'Bagging': sklearn.ensemble.bagging.BaggingRegressor(),
        'Stochastic gradient': sklearn.linear_model.stochastic_gradient.SGDRegressor(),
        'Passive aggressive regression': sklearn.linear_model.PassiveAggressiveRegressor(),
    }

    if regressor in regressor_dict.keys():
        model = regressor_dict[regressor]
    else:
        model= regressor()

    model= sklearn.multioutput.MultiOutputRegressor(model)

    model.fit(sequentions[0], sequentions[1])

    model.output_shape = sequentions[1].shape[1]

    return model


def predict(x_input, model, predicts=7):

    if model.output_shape == 1:

        predictions = []

        for i in range(predicts):

            yhat = model.predict(x_input)
            x_input = np.insert(x_input, x_input.shape[1], yhat[0], axis=1)
            x_input = np.delete(x_input, 0, axis=1)
            predictions.append(yhat[0])

    else:

        predictions = model.predict(x_input)
        predictions = predictions[0]

    predictions = np.array(predictions).reshape(-1)

    return predictions
