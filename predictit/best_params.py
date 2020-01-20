"""Module with main function optimize that find optimal input paramaters for function. Arguments are input model, initial function arguments and arguments limits.
More info is in optimize function documentation"""

import numpy as np
import itertools
import warnings
import time
import sys
import traceback

from . import data_prep, evaluate_predictions


def optimize(model, kwargs, kwargs_limits, data, criterion='mape', fragments=10, iterations=3, predicts=7, details=0, time_limit=5, predicted_column_index=0, name='your model'):
    """Function to find optimal parameters of function. For example if we want to find minimum of function x^2,
    we can use limits from -10 to 10. If we have 4 fragments and 3 iterations. it will separate interval on 4 parts,
    so we have aproximately points -10, -4, 4, 10. We evaluate the best one and make new interval to closest points,
    so new interval will ber -4 and 4. We divide again into 4 points. We repeat as many times as iterations variable defined.

    If limits are written as int, it will be used only as int, so if you want to use float, write -10.0, 10.0 etc...

    If we have many arguments, it will create many combinations of parameters, so beware...

    Args:
        model (func): Function to be optimized (eg: ridgeregression).
        kwargs (dict): Initial arguments (eg: {"alpha": 0.1, "n_steps_in": 10}).
        kwargs_limits (dict): Bounds of arguments (eg: {"alpha": [0.1, 1], "n_steps_in":[2, 30]}).
        data (list, array, dataframe col): Data on which function is optimized.
        criterion (str, optional): Error criterion used in evaluation. 'rmse' or 'mape'. Defaults to 'mape'.
        fragments (int, optional): Number of optimized intervals. Defaults to 10.
        iterations (int, optional): How many times will be initial interval divided into fragments. Defaults to 3.
        predicts (int, optional): Number of predicted values. Defaults to 7.
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
    train, test = data_prep.split(data, predicts=predicts, predicted_column_index=predicted_column_index)

    print(f'\n {name} \n')

    def evaluatemodel(kwargs):
        """Evaluate error function for optimize function.

        Args:
            kwargs (**kwargs): Arguments of model

        Returns:
            float: MAPE or RMSE depends on optimize function argument

        """
        predictions = model(train, **constant_kwargs, **kwargs)
        modeleval = evaluate_predictions.compare_predicted_to_test(predictions, test, criterion=criterion, plot=0)
        return modeleval

    def watchdog(timeout, code, *args, **kwargs):
        """Time-limited execution for optimization function."""
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
    best_params = kwargs

    # If result isn't better during iteration, return results
    memory_result = 0

    # If such a warning occur, parameters combination skipped
    if details == 2:
        warnings.filterwarnings('once')

    else:
        warnings.filterwarnings('ignore')


    warnings.filterwarnings('error', category=RuntimeWarning)
    #warnings.filterwarnings('error', message=r".*ambiguous*")


    for i, j in kwargs_limits.items():

        if not isinstance(j[0], (int, float, np.ndarray, np.generic)):
            kwargs_fragments[i] = j
        elif isinstance(j[0], int):
            pomoc = np.linspace(j[0], j[1], fragments, dtype=int)
            kwargs_fragments[i] = list(set([int(round(j)) for j in pomoc]))
        else:
            kwargs_fragments[i] = np.unique(np.linspace(j[0], j[1], fragments))

    for i in range(iterations):
        if details > 0:
            print(f"Iteration {i + 1} / {iterations}")

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
                        print(f"New best result {best_result} on model {name} with parameters \n \t {best_params}")

            except Exception:
                if details == 2:
                    warnings.warn(f"\n \t Error on model {name}: with params {kombi[k]} - {traceback.format_exc()} \n")

            finally:

                if details == 3:
                    print(f"Result {res} on model {name} with parameters \n \t {kombi[k]}")

                if counter == 0:
                    return best_params

                if details > 0 and percent > 0 and counter % 10 == 1:
                    print("Optimization is in {} %".format(counter / percent))

        # If result not getting better through iteration, not worth of continuing. If last iteration, do not create intervals
        if round(memory_result, 4) == round(best_result, 4) or i + 1 == iterations:
            if details == 1:
                print(f"Best result {best_result} with parameters {best_params} on model {name} \n")

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

        if details == 1:
            print(f"Best result {best_result} with parameters {best_params} on model {name} \n")


