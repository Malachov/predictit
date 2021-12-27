"""Module with optimization functions. There are two main functions.

First is hyperparameter_optimization, which can be used for finding best parameters for some machine learning
models (parameters like alpha in regression models, number of layers in neural nets etc.). Most important
result in result class is `best_params` which is dict that can be used as **kwargs in optimized model.

Second is config_optimization for optimization of all the process which optimizes parameters like datalength,
number of regressive members inputting model (default_n_steps_in) etc. In result there is a dict of optimized
values that can be later used in config.update() function.
"""
from __future__ import annotations

from typing import Any, Callable
import itertools
import time

from typing_extensions import Literal
import numpy as np
import pandas as pd

import mylogging
import mypythontools

from . import _helpers
from . import analyze
from . import configuration
from . import evaluate_predictions
from . import models
from . import result_classes
from .main import predict

# Lazy load
# import plotly.graph_objects as go


def config_optimization(
    optimization: dict[str, Any] = None,
    config: configuration.Config = configuration.config,
) -> result_classes.ConfigOptimization:
    """There are a lot of variables in settings. It can be hard to find optimal values for prediction process.
    This functions evaluate predictions for all defined optimized variables values and then analyses the
    results to find what values are the best. Whereas hyperparameter_optimization iteratively computes new
    values of optimized parameter, this function use only the defined ones. There is also possibility to plot
    all predictions if parameters are tuned to achieve some particular behavior.

    Args:
        optimization (dict[str, Any], optional): Dictionary of config variables that will be optimized
            and its used values. It is also in config but positional parameter has priority. Default to None.
        config (configuration.Config, optional): Settings that will be used in prediction. It is
            the same settings as in predict `function`. Defaults to predictit.config.

    Returns:
        result_classes.ConfigOptimization: Result class with optimization results that can be
            further used in config.update() and analysis with another information from optimization.

    Example:
        >>> config.config_optimization.optimization = {"default_n_steps_in": [3, 5]}
        >>> config.data_input.datalength = 150
        >>> config_optimization(config=config, data="test_sin")

    """

    if optimization is not None:
        config.config_optimization.optimization = optimization

    models_enum = {j: i for i, j in enumerate(config.models.used_models)}

    list_of_df: list[pd.DataFrame] = []
    global_list_of_df: list[pd.DataFrame] = []

    results_dict: dict[str, result_classes.ConfigOptimizationSingle] = {}

    config.output.print_subconfig.print = False
    config.output.plot_subconfig.show_plot = False
    config.prediction.mode = "validate"

    # There are some data that need to be used from one of predictions, but any configuration can fail

    prediction_details = None

    for optimization_variable, optimization_values in config.config_optimization.optimization.items():

        evaluated_matrix = np.zeros((len(optimization_values), len(config.models.used_models)))
        evaluated_matrix.fill(np.nan)

        all_results = {}

        for optimization_index, optimization_value in enumerate(optimization_values):
            config.update({optimization_variable: optimization_value})

            try:

                result = predict(config=config)

                all_results[optimization_value] = result

                if not prediction_details:
                    prediction_details = result.details

                for model, error in result.results_df["Model error"].items():
                    evaluated_matrix[optimization_index, models_enum[model]] = error

                df = result.results_df

                df.reset_index(inplace=True, drop=False)
                df.rename(columns={"index": "Model"}, inplace=True)
                df["Config optimization value"] = optimization_value

                global_df = df.copy()
                global_df["Config optimization variable"] = optimization_variable
                global_list_of_df.append(global_df)

                df.set_index(["Model", "Config optimization value"], inplace=True)

                list_of_df.append(df)

            except Exception:
                raise RuntimeError(
                    mylogging.return_str(
                        f"Config optimization for {optimization_variable} and {optimization_value} failed."
                    )
                )

        all_results_df = pd.concat(list_of_df)

        _helpers.sort_df_index(all_results_df, config.output.sort_results_by)

        (_, best_values_dict, values_results, best_model_name, best_value,) = analyze.analyze_results(
            evaluated_matrix,
            optimization_values,
            config.models.used_models,
            config.prediction.error_criterion,
        )

        best_indexes = []

        for i in all_results_df.index:
            if i[1] == best_values_dict[i[0]]:
                best_indexes.append(i)

        best_results_df = all_results_df.loc[best_indexes]

        best_predictions = pd.DataFrame(
            best_results_df["Prediction"].values.tolist(),
            columns=prediction_details.prediction_index,
            index=best_results_df.index,
        ).T
        all_predictions = pd.DataFrame(
            all_results_df["Prediction"].values.tolist(),
            columns=prediction_details.prediction_index,
            index=all_results_df.index,
        ).T

        results_dict[optimization_variable] = result_classes.ConfigOptimizationSingle(
            variable=optimization_variable,
            values=optimization_values,
            best_value=best_value,
            best_values_dict=best_values_dict,
            values_results=values_results,
            best_results_df=best_results_df,
            all_results_df=all_results_df,
            best_predictions=best_predictions,
            all_predictions=all_predictions,
            best_model_name=best_model_name,
            all_results=all_results,
        )

    global_results = pd.concat(global_list_of_df)
    global_results.set_index(
        ["Model", "Config optimization variable", "Config optimization value"], inplace=True
    )
    _helpers.sort_df_index(global_results, config.output.sort_results_by)
    global_predictions = pd.DataFrame(
        global_results["Prediction"].values.tolist(),
        columns=prediction_details.prediction_index,
        index=global_results.index,
    ).T

    best_values = {i: j.best_value for i, j in results_dict.items()}

    optimization_result = result_classes.ConfigOptimization(
        best_values=best_values,
        results_dict=results_dict,
        global_results=global_results,
        global_predictions=global_predictions,
    )

    if config.config_optimization.plot_all:

        global_predictions = optimization_result.global_predictions.copy()
        global_predictions.columns = [
            " - ".join(tuple(map(str, t))) for t in global_predictions.columns.values
        ]

        global_predictions.insert(0, "Test", prediction_details.test)

        if config.output.plot_subconfig.plot_type == "with_history":
            plot_data = pd.concat(
                [
                    prediction_details.history,
                    global_predictions,
                ],
                sort=False,
            )
            plot_data.iloc[-config.output.predicts - 1, :] = prediction_details.last_value
        else:
            plot_data = global_predictions

        mypythontools.plots.plot(
            plot_data,
            plot_library=config.output.plot_subconfig.plot_library,
            title=config.output.plot_subconfig.plot_title,
            legend=config.output.plot_subconfig.plot_legend,
            highlighted_column="Test",
            save=config.output.plot_subconfig.save_plot,
            return_div=False,
        )

    return optimization_result


def hyperparameter_optimization(
    model_train: Callable,
    model_predict: Callable,
    kwargs: dict[str, Any],
    kwargs_limits: dict[str, tuple | list],
    model_train_input: Any,
    model_test_inputs: list | np.ndarray,
    models_test_outputs: np.ndarray,
    boosted: Literal[0, 1, 2] = 1,
    error_criterion: str = "mape",
    fragments: int = 10,
    iterations: int = 3,
    details: int = 0,
    time_limit: None | int | float = 50,
    name: str = "Your model",
    plot: bool = True,
    plot_all_predictions: bool = False,
) -> result_classes.HyperparameterOptimization:
    """Function to find optimal parameters of function. For example if we want to find minimum of function x^2,
    we can use limits from -10 to 10. If we have 4 fragments and 3 iterations. it will separate interval on 4 parts,
    so we have approximately points -10, -4, 4, 10. We evaluate the best one and make new interval to closest points,
    so new interval will ber -4 and 4. We divide again into 4 points. We repeat as many times as iterations variable
    defined.

    Note: If limits are written as int, it will be used only as int, so if you want to use float, write -10.0,
    10.0 etc... If you want to define concrete values to be evaluated, just use list of more than 2 values (also you
    can use strings).

    If we have many arguments, it will create many combinations of parameters, so beware, it can be very
    computationally intensive...

    Args:
        model_train (Callable): Model train function (eg: ridgeregression.train).
        model_predict (Callable): Model predict function (eg: ridgeregression.predict).
        kwargs (dict[str, Any]): Initial arguments (eg: {"alpha": 0.1, "n_steps_in": 10}).
        kwargs_limits (dict[str, Any]): Bounds of arguments (eg: {"alpha": [0.1, 1], "n_steps_in":[2, 30]}).
        model_train_input (Inputs): Data on which function is
            optimized. Use train data or sequences (tuple with (X, y)) - depends on model.
        model_test_inputs (list | np.ndarray): Error criterion is evaluated to
            be able to compare results. It has to be out of sample data, so data from test set.
        models_test_outputs (np.ndarray): Test set outputs.
        boosted (Literal[0, 1, 2], optional): If boosted, optimize kwargs one after another. It means much less combinations that need to be evaluated,
            so it will be much faster. On the other hand it may not find optimal parameters. If 0, no boosting,
            if 1 simple boosting, if 2, all parameters are boosted twice, because after all values changed,
            can converge to other value. Defaults to 2.
        error_criterion (str, optional): Error criterion used in evaluation. 'rmse' or 'mape'. Defaults to 'mape'.
        fragments (int, optional): Number of optimized intervals. Defaults to 10.
        iterations (int, optional): How many times will be initial interval divided into fragments. Defaults to 3.
        details (int, optional): 0 print nothing, 1 print best parameters of models, 2 print every new best parameters
            achieved, 3 prints all results. Bigger than 0 print percents of progress. Defaults to 0.
        time_limit (None | int | float, optional): How many seconds can one evaluation last. Defaults to 5.
        name (str, optional): Name of model to be displayed in details. Defaults to 'your model'.
        plot (bool, optional): Plot parameters analysis where you can see how the parameters influence
            the result error. Defaults to True.
        plot_all_predictions (bool, optional): It's possible to plot all parameters combinations to analyze it's influence.
            Defaults to False.

    Returns:
        HyperparameterOptimization: Optimized parameters of model.


    Example:
        >>> from models import autoreg_LNU
        >>> from mydatapreprocessing import create_model_inputs, generate_data
        ...
        >>> data = generate_data.sin(107)
        >>> train = data[:100]
        >>> expected_result = data[100:].reshape(1, -1)
        >>> train, test, input, test_output = create_model_inputs.make_sequences(data.reshape(-1, 1), 5)
        ...
        >>> optimization_result = hyperparameter_optimization(
        ...     model_train=autoreg_LNU.train,
        ...     model_predict=autoreg_LNU.predict,
        ...     kwargs_limits={
        ...         "learning_rate": (0.000001, 1),
        ...         "epochs": (1, 50),
        ...         "normalize_learning_rate": [True, False],
        ...     },
        ...     kwargs={"learning_rate": 0.0001, "epochs": 20, "normalize_learning_rate": False},
        ...         model_train_input=(train, test),
        ...         model_test_inputs=[input],
        ...         models_test_outputs=expected_result,
        ...     iterations = 1,  # Just for faster tests
        ...     plot=True
        ... )
        >>> optimization_result.best_params
        {'learning_rate': ...
    """

    start_time = time.time()

    def inner(
        kwargs,
        kwargs_limits,
        is_boosted=False,
    ) -> result_classes.HyperparameterOptimization:

        analysis = result_classes.AnalysisResult(error_criterion=error_criterion, boosted=boosted)
        result = result_classes.HyperparameterOptimization(analysis=analysis)

        result.analysis.default_params = kwargs

        kwargs_values = kwargs_limits.copy()
        best_optimized_params = None  # Do not include default params

        # If result isn't better during iteration, return results
        memory_result = 0
        all_combinations = []

        n_test_samples = models_test_outputs.shape[0]
        predicts = models_test_outputs.shape[1]
        last_printed_time = time.time()

        def evaluatemodel(model_kwargs: dict):
            """Evaluate error function for optimize function.

            Args:
                model_kwargs (dict): Arguments of model

            Returns:
                float: MAPE or RMSE depends on optimize function argument

            """

            modeleval = np.zeros(n_test_samples)

            try:
                trained_model = model_train(model_train_input, **{**kwargs, **model_kwargs})

                for repeat_iteration in range(n_test_samples):

                    create_plot = (
                        True if plot_all_predictions and repeat_iteration == n_test_samples - 1 else False
                    )

                    predictions = model_predict(
                        model_test_inputs[repeat_iteration],
                        trained_model,
                        predicts=predicts,
                    )
                    modeleval[repeat_iteration] = evaluate_predictions.compare_predicted_to_test(
                        predictions,
                        models_test_outputs[repeat_iteration],
                        error_criterion=error_criterion,
                        model_name=f"{name} - {model_kwargs}",
                        plot=create_plot,
                    )

                return np.mean(modeleval)

            except (Exception,):
                mylogging.traceback()
                return np.inf

        # Test default parameters (can be the best)
        default_result = evaluatemodel(result.analysis.default_params)

        result.best_params = result.analysis.default_params if default_result != np.inf else {}
        result.analysis.default_result = default_result

        if details > 0:
            print(
                f"\n\nOptimization of model {name}:\n\n  Default parameters result: {result.analysis.best_result}\n"
            )

        for iteration in range(iterations):
            if details > 0:
                print(f"    Iteration {iteration + 1} / {iterations} results: \n")

            kwargs_values = _get_kwargs_values(kwargs_values, best_optimized_params, fragments=fragments)

            combinations = list(itertools.product(*kwargs_values.values()))

            combi_len = len(combinations)
            percent = round(combi_len / 100, 1)

            list_of_combinations = []
            for j in combinations:
                combination_dict = {key: value for (key, value) in zip(kwargs_values.keys(), j)}
                list_of_combinations.append(combination_dict)

            counter = 0
            for k, combination in enumerate(combinations):
                counter += 1

                if combination in all_combinations:
                    continue

                all_combinations.append(combination)

                try:
                    if time_limit:
                        res = mypythontools.misc.watchdog(time_limit, evaluatemodel, list_of_combinations[k])
                    else:
                        res = evaluatemodel(list_of_combinations[k])

                    if res not in [None, np.nan, np.inf]:
                        if res < result.analysis.best_result:
                            result.analysis.best_result = res
                            result.best_params = list_of_combinations[k]

                            if details == 2:
                                print(
                                    f"\n  New best result {result.analysis.best_result} with parameters: \t {result.best_params}\n"
                                )

                        elif res > result.analysis.worst_result:
                            result.analysis.worst_result = res
                            result.analysis.worst_params = list_of_combinations[k]

                        if is_boosted:
                            result.analysis.all_boosted_results[
                                list(list_of_combinations[k].values())[0]
                            ] = res

                except TimeoutError:
                    mylogging.warn(
                        "Some kwargs combinations in hyperparameter optimization failed because of TimeoutError."
                    )
                    res = np.nan

                except (Exception,):
                    if details > 0:
                        mylogging.traceback(
                            f"Runtime error on model {name}: with params {list_of_combinations[k]}"
                        )
                    res = np.nan

                finally:

                    if details == 3:
                        print(f"    {res}  with parameters:  {list_of_combinations[k]}")

                    if (
                        details > 0
                        and percent > 0
                        and counter % 10 == 1
                        and time.time() - last_printed_time > 3
                    ):
                        print(f"\tOptimization is in {int(counter / percent)} %")
                        last_printed_time = time.time()

            if memory_result != 0 and (memory_result - result.analysis.best_result) < 10e-6:
                if details > 0:
                    print(
                        "  Optimization stopped, because converged. "
                        f"Best result {result.analysis.best_result} with parameters {result.best_params}"
                    )
                return result

            # None of params combinations finished
            elif not result.best_params:
                raise RuntimeError(f"  Optimization failed. None of parameters combinations finished.")

            memory_result = result.analysis.best_result

        if details > 0:
            print(
                f"  Optimization finished. Best result {result.analysis.best_result} with parameters {result.best_params}"
            )
        return result

    if not boosted:
        return inner(kwargs, kwargs_limits)

    else:

        def boosted_optimization(kwargs):

            boosted_kwargs = kwargs.copy()

            boosted_analysis = result_classes.AnalysisResult(error_criterion=error_criterion, boosted=boosted)
            boosted_result = result_classes.HyperparameterOptimization(analysis=boosted_analysis)

            boosted_result.analysis.default_params = kwargs
            boosted_description = {}

            for i in kwargs_limits:

                result = inner(boosted_kwargs, {i: kwargs_limits[i]}, is_boosted=True)

                if not boosted_result.analysis.default_result:
                    boosted_result.analysis.default_result = result.analysis.default_result

                boosted_result.analysis.all_boosted_results[i] = result.analysis.all_boosted_results

                if result.analysis.best_result < boosted_result.analysis.best_result:
                    boosted_result.analysis.best_result = result.analysis.best_result

                for i, j in boosted_result.analysis.all_boosted_results.items():
                    boosted_description[i] = [min(j.values()), max(j.values()), sum(j.values()) / len(j)]

                boosted_result.best_params[i] = result.best_params[i]
                boosted_kwargs[i] = result.best_params[i]

            boosted_result.analysis.boosted_result_description = pd.DataFrame.from_dict(
                boosted_description, orient="index", columns=["min", "max", "average"]
            )

            return boosted_result

        boosted_result = boosted_optimization(kwargs)

        if boosted == 2:
            boosted_result.best_params = boosted_optimization(boosted_result.best_params).best_params

        df_parts = []

        for i, j in boosted_result.analysis.all_boosted_results.items():

            df = pd.DataFrame.from_dict(j, orient="index", columns=["Error"])
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Value"}, inplace=True)
            df["Parameter"] = i

            df_parts.append(df)

        plot_df = pd.concat(df_parts)

        if plot:
            import plotly.express as px

            fig = px.strip(
                plot_df,
                x="Error",
                y="Parameter",
                hover_data={"Error": False, "Parameter": False, "Values": plot_df["Value"]},
            )
            # fig = px.strip(plot_df, x="Error", y="Parameter", hover_data=["Error", "Value"])

            fig.layout.update(
                mypythontools.plots.get_plotly_layout.categorical_scatter(title="Hyperparameter optimization")
            )

            fig.show()

        boosted_result.analysis.time = start_time - time.time()

        return boosted_result


def _get_kwargs_values(kwargs_values, best_kwargs, fragments=10):
    """Generate new possible possible parameters equally distributed around best value.
    If no best value, all interval is used.

    Args:
        kwargs_values (dict[str, tuple[int | float, int | float] | list[str | int | str]]): Tuple with minimum and maximums of optimized kwargs or list of possible values. it is string or more than two values,
            the best one is chosen.
        best_kwargs (dict): [description]
        fragments (int, optional): [description]. Defaults to 10.

    Returns:
        dict: Dict with kwargs values.

    Example:
        If there is no best value yet, new values within defined intervals are created or all values are used.
        >>> kwargs_values = {
        ...     "model": ['DecisionTreeRegressor', 'LinearRegression', 'Ridge'],
        ...     "alpha": (0.00001, 1.0),
        ...     "n_iter": (1, 30),
        ... }
        ...
        >>> new_values = _get_kwargs_values(kwargs_values, best_kwargs=None)
        ...
        >>> new_values["n_iter"]
        [1, 4, 7, 11, 14, 17, 20, 24, 27, 30]
        >>> new_values["model"]
        ['DecisionTreeRegressor', 'LinearRegression', 'Ridge']

        >>> kwargs_values = {
        ...     "model": ["DecisionTreeRegressor", "LinearRegression", "Ridge"],
        ...     "alpha": (0.00001, 1.0),
        ...     "n_iter": (1, 30),
        ... }
        ...
        >>> best_kwargs = {
        ...     "model": "LinearRegression",
        ...     "alpha": 1.0,
        ...     "n_iter": 30,
        ... }
        ...
        >>> new_values = _get_kwargs_values(kwargs_values, best_kwargs=best_kwargs)
        ...
        >>> new_values["n_iter"]
        [27, 28, 29, 30]
        >>> new_values["model"]
        ['LinearRegression']

    """
    values = {}

    for kwarg_name, iterated_values in kwargs_values.items():

        # kwarg_best_value = best_kwargs[kwarg_name] if best_kwargs else None

        if (
            not isinstance(iterated_values[0], (int, float, np.ndarray))
            or isinstance(iterated_values[0], bool)  # If first condition return False, because of int
            or len(iterated_values) != 2
        ):
            if best_kwargs:
                values[kwarg_name] = [best_kwargs[kwarg_name]]
            else:
                values[kwarg_name] = iterated_values
            continue

        else:
            if not best_kwargs:
                lower_bound = iterated_values[0]
                upper_bound = iterated_values[1]
            else:
                step = (max(kwargs_values[kwarg_name]) - min(kwargs_values[kwarg_name])) / fragments
                lower_bound = best_kwargs[kwarg_name] - step
                upper_bound = best_kwargs[kwarg_name] + step

                if lower_bound < min(iterated_values):
                    lower_bound = min(iterated_values)

                if upper_bound > max(iterated_values):
                    upper_bound = max(iterated_values)

        new_values = np.linspace(lower_bound, upper_bound, fragments)

        convert_to_list = True if isinstance(iterated_values[0], (int, float)) else False
        convert_to_int = True if isinstance(iterated_values[0], int) else False

        if convert_to_int:
            new_values = np.unique(np.rint(new_values).astype(int))

        if convert_to_list:
            new_values = new_values.tolist()

        values[kwarg_name] = new_values

    return values


def input_optimization(
    config: configuration.Config = configuration.config,
) -> result_classes.BestInput:
    """Function find best input for all the models.

    Args:
        config (configuration.Config, optional): Settings that will be used in prediction. It is
            the same settings as in predict `function`. Defaults to predictit.config.

    Returns:
        result_classes.BestInput: Results with dict with best inputs and results.
    """
    all_models = config.models.used_models
    all_parameters = config.models.models_parameters

    results = {}
    tables = {}
    best_data_dict = {}

    # This runs predict function, but beforehand it configure that used models are just variations of one
    # iterated model with various inputs.
    for i in all_models:
        config.models.models_parameters = {}
        config.models.used_models = []
        config.models.models_input = {}
        for j in config.models.data_inputs.keys():

            model_name_variations = f"{i} - {j}"
            config.models.used_models.append(model_name_variations)
            if i in all_parameters:
                config.models.models_parameters[model_name_variations] = all_parameters[i]
            config.models.models_input[model_name_variations] = j
            models.models_assignment[model_name_variations] = models.models_assignment[i]

        try:
            result = predict(config=config)
            results[i] = result
            tables[i] = result.tables.simple
            best_data_dict[i] = result.best_model_name[len(i) + 3 :]

        except Exception:
            pass

    return result_classes.BestInput(best_data_dict, tables, results)
