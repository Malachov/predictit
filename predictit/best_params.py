"""Module with main function optimize that find optimal input paramaters for function. Arguments are input
model, initial function arguments and arguments limits. More info is in optimize function documentation.
"""

from __future__ import annotations
from typing import Any, Callable
import itertools
import time

import numpy as np

import mylogging
from mypythontools.misc import watchdog
from mydatapreprocessing.create_model_inputs import Inputs

from . import evaluate_predictions


def optimize(
    model_train: Callable,
    model_predict: Callable,
    kwargs: dict[str, Any],
    kwargs_limits: dict[str, Any],
    model_train_input: Inputs,
    model_test_inputs: list | np.ndarray,
    models_test_outputs: np.ndarray,
    error_criterion: str = "mape",
    fragments: int = 10,
    iterations: int = 3,
    details: int = 0,
    time_limit: int | float = 5,
    name: str = "Your model",
    plot: bool = False,
) -> None | dict[str, Any]:
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
            optimized. Use train data or sequences (tuple with (X, y, x_input)) - depends on model. Defaults to None.
        model_test_inputs (list | np.ndarray): Error criterion is evaluated to
            be able to compare results. It has to be out of sample data, so data from test set.
        models_test_outputs (np.ndarray): Test set outputs.
        error_criterion (str, optional): Error criterion used in evaluation. 'rmse' or 'mape'. Defaults to 'mape'.
        fragments (int, optional): Number of optimized intervals. Defaults to 10.
        iterations (int, optional): How many times will be initial interval divided into fragments. Defaults to 3.
        details (int, optional): 0 print nothing, 1 print best parameters of models, 2 print every new best parameters
            achieved, 3 prints all results. Bigger than 0 print percents of progress. Defaults to 0.
        time_limit (int | float, optional): How many seconds can one evaluation last. Defaults to 5.
        name (str, optional): Name of model to be displayed in details. Defaults to 'your model'.
        plot (bool, optional): It's possible to plot all parameters combinations to analyze it's influence.
            Defaults to False.

    Returns:
        dict: Optimized parameters of model.

    """

    kwargs_fragments = {}
    constant_kwargs = (
        {key: value for (key, value) in kwargs.items() if key not in kwargs_limits} if kwargs else {}
    )
    kwargs = {key: value for (key, value) in kwargs.items() if key not in constant_kwargs} if kwargs else {}

    n_test_samples = models_test_outputs.shape[0]
    predicts = models_test_outputs.shape[1]
    last_best_params = {}
    last_printed_time = time.time()

    def evaluatemodel(model_kwargs):
        """Evaluate error function for optimize function.

        Args:
            model_kwargs (dict): Arguments of model

        Returns:
            float: MAPE or RMSE depends on optimize function argument

        """

        modeleval = np.zeros(n_test_samples)

        try:
            trained_model = model_train(model_train_input, **constant_kwargs, **model_kwargs)

            for repeat_iteration in range(n_test_samples):

                create_plot = True if plot and repeat_iteration == n_test_samples - 1 else False

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
            return np.inf

    # Test default parameters (can be the best)
    best_result = evaluatemodel(kwargs)

    if best_result != np.inf:
        best_params = kwargs
    else:
        best_params = {}

    if details > 0:
        print(f"\n\nOptimization of model {name}:\n\n  Default parameters result: {best_result}\n")

    # If result isn't better during iteration, return results
    memory_result = 0

    all_combinations = []

    for i, j in kwargs_limits.items():

        if not isinstance(j[0], (int, float, np.ndarray)) or len(j) != 2:
            kwargs_fragments[i] = j
        elif isinstance(j[0], int):
            help_var = np.linspace(j[0], j[1], fragments, dtype=int)
            kwargs_fragments[i] = list(set([int(round(j)) for j in help_var]))
        else:
            kwargs_fragments[i] = np.unique(np.linspace(j[0], j[1], fragments))

    for iteration in range(iterations):
        if details > 0:
            print(f"    Iteration {iteration + 1} / {iterations} results: \n")

        combinations = list(itertools.product(*kwargs_fragments.values()))

        combi_len = len(combinations)
        percent = round(combi_len / 100, 1)

        list_of_combinations = []
        for j in combinations:
            combination_dict = {key: value for (key, value) in zip(kwargs_limits.keys(), j)}
            list_of_combinations.append(combination_dict)

        counter = 0
        for k, combination in enumerate(combinations):
            counter += 1

            if combination in all_combinations:
                continue

            all_combinations.append(combination)

            try:
                if time_limit:
                    res = watchdog(time_limit, evaluatemodel, list_of_combinations[k])
                else:
                    res = evaluatemodel(list_of_combinations[k])

                if res is not None and res is not np.nan and res < best_result:
                    best_result = res
                    best_params = list_of_combinations[k]

                    if details == 2:
                        print(f"\n  New best result {best_result} with parameters: \t {best_params}\n")

            except (Exception,):
                if details > 0:
                    mylogging.traceback(f"Error on model {name}: with params {list_of_combinations[k]}")
                res = np.nan

            finally:

                if details == 3:
                    print(f"    {res}  with parameters:  {list_of_combinations[k]}")

                if details > 0 and percent > 0 and counter % 10 == 1 and time.time() - last_printed_time > 3:
                    print(f"\tOptimization is in {int(counter / percent)} %")
                    last_printed_time = time.time()

        if last_best_params != best_params and (memory_result != 0 and (memory_result - best_result) < 10e-6):
            if details > 0:
                print(
                    (
                        f"  Optimization stopped, because converged. "
                        "Best result {best_result} with parameters {best_params}"
                    )
                )
            return best_params

        # If last iteration, do not create intervals
        elif iteration + 1 == iterations:
            if details > 0:
                print(f"  Optimization finished. Best result {best_result} with parameters {best_params}")
            return best_params

        # None of params combinations finished
        elif not best_params:
            if details > 0:
                print(f"  Optimization failed. None of parameters combinations finished.")
            return best_params

        memory_result = best_result
        last_best_params = best_params

        for i, j in kwargs_limits.items():

            if not isinstance(j[0], (int, float, np.ndarray)) or len(j) != 2:
                kwargs_fragments[i] = [best_params[i]]
            else:
                step = (max(kwargs_fragments[i]) - min(kwargs_fragments[i])) / fragments
                if best_params[i] - step < j[0]:
                    kwargs_fragments[i] = np.linspace(best_params[i], best_params[i] + step, fragments)
                elif best_params[i] + step > j[1]:
                    kwargs_fragments[i] = np.linspace(best_params[i] - step, best_params[i], fragments)
                else:
                    kwargs_fragments[i] = np.linspace(best_params[i] - step, best_params[i] + step, fragments)
                kwargs_fragments[i] = np.unique(kwargs_fragments[i])
                if isinstance(j[0], int):
                    kwargs_fragments[i] = list(set([int(round(k)) for k in kwargs_fragments[i]]))
