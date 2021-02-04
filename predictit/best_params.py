"""Module with main function optimize that find optimal input paramaters for function.
Arguments are input model, initial function arguments and arguments limits.
More info is in optimize function documentation"""

import numpy as np
import itertools
import time
import sys
import mylogging

from . import evaluate_predictions


def optimize(model_train, model_predict, kwargs, kwargs_limits, model_train_input, model_test_inputs, models_test_outputs, error_criterion='mape',
             multicolumn_source=0, fragments=10, iterations=3, details=0, time_limit=5, predicted_column_index=0, name='Your model', plot=0):
    """Function to find optimal parameters of function. For example if we want to find minimum of function x^2,
    we can use limits from -10 to 10. If we have 4 fragments and 3 iterations. it will separate interval on 4 parts,
    so we have aproximately points -10, -4, 4, 10. We evaluate the best one and make new interval to closest points,
    so new interval will ber -4 and 4. We divide again into 4 points. We repeat as many times as iterations variable defined.

    If limits are written as int, it will be used only as int, so if you want to use float, write -10.0, 10.0 etc...

    If we have many arguments, it will create many combinations of parameters, so beware...

    Args:
        model_train (func): Model train function (eg: ridgeregression.train).
        model_predict (func): Model predict function (eg: ridgeregression.predict).
        kwargs (dict): Initial arguments (eg: {"alpha": 0.1, "n_steps_in": 10}).
        kwargs_limits (dict): Bounds of arguments (eg: {"alpha": [0.1, 1], "n_steps_in":[2, 30]}).
        train_input (np.array or tuple(np.ndarray, np.ndarray, np.ndarray)): Data on which function is optimized.
            Use train data or sequentions (tuple with (X, y, x_input)) - depends on model. Defaults to None.
        X (np.array): Input sequentions on which function is optimized. Use this or train_input - depends on model. Defaults to None.
        y (np.array): Output on which function is optimized. Use this or train_input - depends on model. Defaults to None.
        error_criterion (str, optional): Error criterion used in evaluation. 'rmse' or 'mape'. Defaults to 'mape'.
        fragments (int, optional): Number of optimized intervals. Defaults to 10.
        iterations (int, optional): How many times will be initial interval divided into fragments. Defaults to 3.
        details (int, optional): 0 print nothing, 1 print best parameters of models, 2 print every new best parameters achieved,
            3 prints all results. Bigger than 0 print precents of progress. Defaults to 0.
        time_limit (int, optional): How many seconds can one evaluation last. Defaults to 5.
        predicted_column_index (int, optional): If predicted data havbe more columns, which is predicted. Defaults to 0.
        name (str, optional): Name of model to be displayed in details. Defaults to 'your model'.

    Raises:
        TimeoutError: If evaluation take more time than defined limit.

    Returns:
        dict: Optimized parameters of model.

    """

    kwargs_fragments = {}
    constant_kwargs = {key: value for (key, value) in kwargs.items() if key not in kwargs_limits}
    kwargs = {key: value for (key, value) in kwargs.items() if key not in constant_kwargs}

    n_test_samples = models_test_outputs.shape[0]
    predicts = models_test_outputs.shape[1]


    def evaluatemodel(kwargs):
        """Evaluate error function for optimize function.

        Args:
            kwargs (**kwargs): Arguments of model

        Returns:
            float: MAPE or RMSE depends on optimize function argument

        """

        modeleval = np.zeros(n_test_samples)

        trained_model = model_train(model_train_input, **constant_kwargs, **kwargs)

        for repeat_iteration in range(n_test_samples):

            create_plot = 1 if plot and repeat_iteration == n_test_samples - 1 else 0

            predictions = model_predict(model_test_inputs[repeat_iteration], trained_model, predicts=predicts)
            modeleval[repeat_iteration] = evaluate_predictions.compare_predicted_to_test(
                predictions, models_test_outputs[repeat_iteration], error_criterion=error_criterion,
                modelname=f"{name} - {kwargs}", plot=create_plot)

        return np.mean(modeleval)


    def watchdog(timeout, code, *args, **kwargs):
        """Time-limited execution for python function."""
        def tracer(frame, event, arg, start=time.time()):
            "Helper."
            now = time.time()
            if now > start + timeout:
                raise TimeoutError('Time exceeded')
            return tracer if event == "call" else None

        old_tracer = sys.gettrace()
        try:
            sys.settrace(tracer)
            result = code(*args, **kwargs)
            return result

        except TimeoutError:
            if details > 1:
                print('Time exceeded')

        finally:
            sys.settrace(old_tracer)


    # Test default parameters (can be the best)
    best_result = evaluatemodel(kwargs)

    if details > 0:
        print(f"\n\nOptimization of model {name}:\n\n\tDefault parameters result: {best_result}\n")

    best_params = kwargs

    # If result isn't better during iteration, return results
    memory_result = 0

    # If such a warning occur, parameters combination skipped
    # warnings.filterwarnings('error', category=RuntimeWarning)
    # warnings.filterwarnings('error', message=r".*ambiguous*")

    for i, j in kwargs_limits.items():

        if not isinstance(j[0], (int, float, np.ndarray, np.generic)) or len(j) != 2:
            kwargs_fragments[i] = j
        elif isinstance(j[0], int):
            pomoc = np.linspace(j[0], j[1], fragments, dtype=int)
            kwargs_fragments[i] = list(set([int(round(j)) for j in pomoc]))
        else:
            kwargs_fragments[i] = np.unique(np.linspace(j[0], j[1], fragments))

    for iteration in range(iterations):
        if details > 0:
            print(f"Iteration {iteration + 1} / {iterations} results: \n")

        combinations = list(itertools.product(*kwargs_fragments.values()))

        combi_len = len(combinations)
        percent = round(combi_len / 100, 1)

        kombi = []
        for j in combinations:
            combination_dict = {key: value for (key, value) in zip(kwargs_limits.keys(), j)}
            kombi.append(combination_dict)

        counter = 0
        for k in range(len(combinations)):
            counter +=1

            try:
                res = watchdog(time_limit, evaluatemodel, kombi[k])

                if res is not None and res is not np.nan and res < best_result:
                    best_result = res
                    best_params = kombi[k]

                    if details == 2:
                        print(f"\n\t\tNew best result {best_result} with parameters: \t {best_params}\n")

            except Exception:
                if details > 0:
                    mylogging.traceback(f"Error on model {name}: with params {kombi[k]}")
                res = np.nan

            finally:

                if details == 3:
                    print(f"    {res}  with parameters:  {kombi[k]}")

                if counter == 0:
                    return best_params

                if details > 0 and percent > 0 and counter % 10 == 1:
                    print(f"Optimization is in {counter / percent} %")

        # If result not getting better through iteration, not worth of continuing. If last iteration, do not create intervals
        if (memory_result !=0 and round(memory_result, 6) == round(best_result, 6)) or iteration + 1 == iterations or not best_params:
            if details > 0:
                print(f"Best result {best_result} with parameters {best_params}")
            return best_params

        memory_result = best_result

        for i, j in kwargs_limits.items():

            if not isinstance(j[0], (int, float, np.ndarray, np.generic)):
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
