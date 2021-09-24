"""
Module that use sklearn library to compute mostly time series predictions.

For model structure info, check models module docstrings - share structure with other models.

If main train function is optimized in best_params module (automatic best regressor choose),
it can use function ``get_all_models()`` that return all existing regressor.
"""

import mylogging

from .models_functions.models_functions import one_step_looper

# Lazy imports
# from importlib import import_module
# import sklearn
# from sklearn import multioutput, linear_model, ensemble, tree, neighbors, gaussian_process


def train(
    data,
    model="BayesianRidge",
    n_estimators=100,
    alpha=0.0001,
    alpha_1=1.0e-6,
    alpha_2=1.0e-6,
    lambda_1=1.0e-6,
    lambda_2=1.0e-6,
    n_iter=300,
    epsilon=1.35,
    alphas=[0.1, 0.5, 1],
    gcv_mode="auto",
    solver="auto",
    n_hidden=20,
    rbf_width=0,
    activation_func="selu"
    #  load_trained_model=0, update_trained_model=1, save_model=1, saved_model_path_string='stored_models',
):
    """Sklearn model. Models as input parameter. Can be linear, ridge, Huber or much more.
    It also contain extreme learning machine model from sklearn extensions.

    Note:
        There are many parameters in function, but all models use just a few of them.
        Usually default parameters are just enough.

        Some of models are regressors and some are classifiers. If it's classifier, it's optimal
        to have data sorted in limited number of bins.

    Args:
        data ((np.ndarray, np.ndarray)) - Tuple (X, y) of input train vectors X and train outputs y.
            Insert input with no constant column - added by default in sklearn.
            Check `mydatapreprocessing` how to generate output.
        model ((str, object), optional): Model that will be used. You can insert model itself or
            just a name of used class. All possible options below in docs. Defaults to 'BayesianRidge'.
        n_estimators (100, optional):  Parameter of some model. Defaults to 100.
        alpha (float, optional): Parameter of some model. Defaults to 0.0001.
        alpha_1 (float, optional): Parameter of some model. Defaults to 1.e-6.
        alpha_2 (float, optional): Parameter of some model. Defaults to 1.e-6.
        lambda_1 (float, optional): Parameter of some model. Defaults to 1.e-6.
        lambda_2 (float, optional): Parameter of some model. Defaults to 1.e-6.
        n_iter (int, optional): Parameter of some model. Defaults to 300.
        epsilon (float, optional): Parameter of some model. Defaults to 1.35.
        alphas (list, optional): Parameter of some model. Defaults to [0.1, 0.5, 1].
        gcv_mode (str, optional): Parameter of some model. Defaults to 'auto'.
        solver (str, optional): Parameter of some model. Defaults to 'auto'.
        n_hidden (int, optional): Parameter of some model. Defaults to 20.
        rbf_width (int, optional): Parameter of some model. Defaults to 0.
        activation_func (str, optional): Parameter of some model. Defaults to 'selu'.

    Returns:
        np.ndarray: Predictions of input time series.

    Options if string::

        ['PLSRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'BaggingRegressor',
        'GradientBoostingRegressor', 'AdaBoostRegressor', 'VotingRegressor', 'StackingRegressor',
        'RandomForestClassifier', 'ExtraTreesClassifier', 'BaggingClassifier', 'GradientBoostingClassifier',
        'AdaBoostClassifier', 'VotingClassifier', 'StackingClassifier', 'GaussianProcessRegressor',
        'GaussianProcessClassifier', 'IsotonicRegression', Regression', 'HuberRegressor', 'LinearRegression',
        'LogisticRegression', 'LogisticRegressionCV', 'PassiveAggressiveRegressor', 'SGDRegressor',
        'TheilSenRegressor', 'RANSACRegressor', 'PoissonRegressor', 'GammaRegressor', 'TweedieRegressor',
        'PassiveAggressiveClassifier', 'RidgeClassifier', 'RidgeClassifierCV', 'SGDClassifier', 'OneVsRestClassifier',
        'OneVsOneClassifier', 'OutputCodeClassifier', 'MultiOutputRegressor', 'RegressorChain',
        'MultiOutputClassifier', 'ClassifierChain', 'KNeighborsRegressor', 'RadiusNeighborsRegressor',
        'KNeighborsClassifier', 'RadiusNeighborsClassifier', 'MLPRegressor', 'MLPClassifier',
        'SelfTrainingClassifier', 'DecisionTreeRegressor', 'ExtraTreeRegressor', 'DecisionTreeClassifier',
        'ExtraTreeClassifier', 'TransformedTargetRegressor', 'BayesianRidge', 'ElasticNet', 'Hinge', 'Lars', 'LarsCV',
        'Lasso', 'LassoCV', 'LassoLarsIC', 'Log', 'ModifiedHuber', 'MultiTaskElasticNet', 'MultiTaskLasso',
        'MultiTaskLassoCV', 'OrthogonalMatchingPursuit', 'OrthogonalMatchingPursuitCV', 'Perceptron', 'Ridge',
        'RidgeCV', 'SquaredLoss', 'SVR',
        # Sklearn extensions
        'ELMClassifier', 'ELMRegressor', 'GenELMClassifier', 'GenELMRegressor']
    """
    from sklearn import (
        multioutput,
        linear_model,
        ensemble,
        tree,
        neighbors,
        gaussian_process,
    )

    # If string like 'LinearRegression', find class with such a name
    if isinstance(model, str):

        for i in [linear_model, ensemble, tree, neighbors, gaussian_process]:
            if model in i.__all__:
                model = getattr(i, model)
                break

        # If model is still string, not object from sklearn, it means it was not found,
        # may be from sklearnextensions library
        if isinstance(model, str):

            import sklearn_extensions.extreme_learning_machines.elm as elm

            model = getattr(elm, model)

            # Model defined by string not found
            if isinstance(model, str):

                raise AttributeError(
                    mylogging.return_str(
                        "You defined model that was not found in sklearn. You can use not only string, but also"
                        "object or class itself. You can use function `get_all_models` to get list of all"
                        "possible models and then use one of them."
                    )
                )

    # If class, but no object was configured, create instance
    if callable(model):
        model = model()

    params = {
        "n_estimators": n_estimators,
        "alpha": alpha,
        "alpha_1": alpha_1,
        "alpha_2": alpha_2,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "n_iter": n_iter,
        "epsilon": epsilon,
        "alphas": alphas,
        "gcv_mode": gcv_mode,
        "solver": solver,
        "n_hidden": n_hidden,
        "rbf_width": rbf_width,
        "activation_func": activation_func,
    }

    # Params, that are configured in function params as well as configurable in models
    used_params = {i: j for (i, j) in params.items() if i in model.get_params()}

    model.set_params(**used_params)

    if data[1].shape[1] == 1:
        model.output_shape = "one_step"
        output = data[1].ravel()

    else:
        if model._estimator_type == "regressor":
            model = multioutput.MultiOutputRegressor(model)
        elif model._estimator_type == "classifier":
            model = multioutput.MultiOutputClassifier(model)

        model.output_shape = "multi_step"
        output = data[1]

    model.fit(data[0], output)

    return model


def predict(x_input, model, predicts=7):
    """Function that creates predictions from trained model and input data.

    Args:
        x_input (np.ndarray): Time series data inputting the models. Shape = (n_samples, n_features).
            Usually last few data points. Structure depends on X in train function
            (usually defined in mydatapreprocessing library).
        model (list, class): Fitted model object from imported library.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    if model.output_shape == "one_step":

        return one_step_looper(
            lambda new_x_input: model.predict(new_x_input),
            x_input,
            predicts,
            constant=False,
        )

    else:

        return model.predict(x_input)[0].reshape(-1)


def get_all_models(
    regressors=True, classifiers=True, other_models=True, sklearn_extensions=True
):
    """Create list of around 80 various sklearn models where regressor or classifier is in class name
    plus some extra models.
    E.g. ["sklearn.ensemble._forest.ExtraTreesRegressor()", "sklearn.ensemble._bagging.BaggingRegressor()", ...]

    Note:
        If you want just names of models, you can use

        >>> models = [mod.__name__ for mod in get_all_models()]

    Args:
        regressors (bool): Whether add regressors.
        classifiers (bool): Whether add classifiers.
        other_models (bool): Whether add other models like BayesRidge.
        sklearn_extensions (bool): Whether add other models from sklearn extensions (other library).

    Returns:
        list: List of model objects.
    """

    from importlib import import_module
    import sklearn
    from sklearn import linear_model

    models = []

    for module_str in sklearn.__all__:
        try:
            module = import_module(f"sklearn.{module_str}")

            if regressors:
                models.extend(
                    [
                        getattr(module, cls)
                        for cls in module.__all__
                        if "Regressor" in cls or "Regression" in cls
                    ]
                )

            if classifiers:
                models.extend(
                    [
                        getattr(module, cls)
                        for cls in module.__all__
                        if "Classifier" in cls
                    ]
                )

        except (Exception,):
            pass

    if other_models:
        other_linear_models = [
            "BayesianRidge",
            "ElasticNet",
            "Hinge",
            "Lars",
            "LarsCV",
            "Lasso",
            "LassoCV",
            "LassoLarsIC",
            "Log",
            "ModifiedHuber",
            "MultiTaskElasticNet",
            "MultiTaskLasso",
            "MultiTaskLassoCV",
            "OrthogonalMatchingPursuit",
            "OrthogonalMatchingPursuitCV",
            "Perceptron",
            "Ridge",
            "RidgeCV",
            "SquaredLoss",
        ]

        models.extend(
            [
                getattr(linear_model, cls)
                for cls in linear_model.__all__
                if cls in other_linear_models
            ]
        )

        models.append(sklearn.svm.SVR)

    if sklearn_extensions:
        import sklearn_extensions.extreme_learning_machines.elm as elm

        extensions_models = [
            "ELMClassifier",
            "ELMRegressor",
            "GenELMClassifier",
            "GenELMRegressor",
        ]

        models.extend(
            [getattr(elm, cls) for cls in elm.__all__ if cls in extensions_models]
        )

    return models
