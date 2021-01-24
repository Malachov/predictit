
"""
Module that use sklearn library to compute mostly time series predictions.

For model structure info, check models module docstrings - share structure with other models.

If main train function is optimized in best_params module (automatic best regressor choose), it can use function `get_regressors()` that return all existing regressor.
"""

import sklearn
import sklearn.ensemble
import sklearn.multioutput
import sklearn.tree
import sklearn.neighbors

from .models_functions.models_functions import one_step_looper

linear_model = sklearn.linear_model


def train(data, regressor='bayesianridge', n_estimators=100,
          alpha=0.0001, alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6,
          lambda_2=1.e-6, n_iter=300, epsilon=1.35, alphas=[0.1, 0.5, 1], gcv_mode='auto', solver='auto',
          n_hidden=20, rbf_width=0, activation_func='selu'
          #  load_trained_model=0, update_trained_model=1, save_model=1, saved_model_path_string='stored_models',
          ):
    """Sklearn regression model. Regressor is input parameter. It can be linear, or ridge, or Huber or much more else.
    It also contain extreme learning machine model from sklearn extensions.

    Args:
        data ((np.ndarray, np.ndarray)) - Tuple (X, y) of input train vectors X and train outputs y.
            Insert input with no constant column - added by default in sklearn. Check mydatapreprocessings how to generate output.
        regressor (str, optional): Regressor that sklearn use. Options:
            ['bayesianridge', 'huber', 'lasso', 'linear', 'ridgecv', 'ridge', 'Extra trees', 'Random forest',
             'Decision tree', 'Gradient boosting', 'KNeighbors', 'Bagging', 'Stochastic gradient',
             'Passive aggressive regression', 'elm', 'elm_gen']. Defaults to 'bayesianridge'.
        n_estimators (100):  Parameter of some regressor. Defaults to 100.
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
        import sklearn_extensions.extreme_learning_machines.elm as elm
        model = elm.ELMRegressor(n_hidden=n_hidden, alpha=alpha, rbf_width=rbf_width, activation_func=activation_func)
    elif regressor == 'elm_gen':
        import sklearn_extensions.extreme_learning_machines.elm as elm
        model = elm.GenELMRegressor()
    else:
        raise ValueError("regressor parameter must one of defined one, for example ['bayesianridge', 'lasso', 'Decision tree', ...]. Check docstrings for more info.")

    if data[1].shape[1] == 1:
        model.output_shape = 'one_step'
        output = data[1].ravel()

    else:
        model = sklearn.multioutput.MultiOutputRegressor(model)
        model.output_shape = 'multi_step'
        output = data[1]

    model.fit(data[0], output)

    return model


def predict(x_input, model, predicts=7):
    """Function that creates predictions from trained model and input data.

    Args:
        x_input (np.ndarray): Time series data inputting the models. Shape = (n_samples, n_features). Usually last few datapoints. Structure depends on X in train function (usually defined in mydatapreprocessing library).
        model (list, class): Trained model. It can be list of neural weights or it can be fitted model class from imported library.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    if model.output_shape == 'one_step':

        return one_step_looper(lambda x_input: model.predict(x_input), x_input, predicts, constant=False)

    else:

        return model.predict(x_input)[0].reshape(-1)


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
