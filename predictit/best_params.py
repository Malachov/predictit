import numpy as np
import itertools
import warnings
import time
import sys

def optimize(model, kwargs, kwargs_limits, data, criterion='mape', fragments=10, iterations=3, predicts=7, details=0, time_limit = 5, predicted_column_index=0, name='your model'):

    """Function to find optimal parameters of function
    ======
    Output:
    ------
        Optimized parameters {dict}

    Arguments:
    ------
        model {func} -- Function to be optimized (eg: ridgeregression)
        kwargs {dict} -- Initial arguments (eg: {"alpha": 0.1, "n_steps_in": 10})
        kwargs_limits {dict} -- Bounds of arguments (eg: {"alpha": [0.1, 1], "n_steps_in":[2, 30]})
        data {list, array, dataframe col} -- Data on which function is optimized (eg: data1)
        fragments {int} -- Number of optimized intervals (default: 10)
        predicts {int} -- Number of predicted values (default: 7)
    """

    kwargs_fragments = {}
    constant_kwargs = {key: value for (key, value) in kwargs.items() if key not in kwargs_limits}
    kwargs = {key: value for (key, value) in kwargs.items() if key not in constant_kwargs}
    train, test = split(data, predicts=predicts, predicted_column_index=predicted_column_index)


    def evaluatemodel(kwargs):
        predictions = model(train, **constant_kwargs, **kwargs)
        modeleval = test_pre(predictions, test, criterion='mape', predicts=predicts, plot=0)
        return modeleval

    def watchdog(timeout, code, *args, **kwargs):
        "Time-limited execution."
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
            if details == 2:
                print('Time exceeded')

        finally:
            sys.settrace(old_tracer)

    # Testuje, jestli defaultní parametry nejsou nejlepší
    best_result = evaluatemodel(kwargs)
    best_result_memory = best_result
    best_params = kwargs
    
    # Pokud se během iterace výsledek nezlepší, vrátí výsledky
    memory_result = 0

    # Dojde-li k následujícímu warningu, výpočet kroku přerušen
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
            pomoc = np.linspace(j[0], j[1], fragments, dtype = int)
            kwargs_fragments[i] = list(set([int(round(j)) for j in pomoc]))
        else:
            kwargs_fragments[i] = np.unique(np.linspace(j[0], j[1], fragments))

    for i in range(iterations):

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
                res = watchdog(time_limit, evaluatemodel, kombi[k])# za 0.1 doplnit *args **kwargs

                if res < best_result:
                    best_result = res
                    best_params = kombi[k]

                    if details == 2:
                        print("New best result {} on model {} with parameters \n \t {}".format(best_result, name, best_params))

            except Exception as unknown:
                if details == 2:

                    print(" \n Unknown error on model {}: with params {}".format(name, kombi[k]), unknown)

            finally:
                if counter == 0:
                    return best_params

                if details > 0 and counter % 10 == 1:
                    print("Optization is in {} %".format(counter / percent))

        # Pokud se výsledek během iterace nezlepšil, nemá cenu pokračovat

        if round(memory_result, 4) == round(best_result, 4):
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
            print("Best result {} with parameters {} on model {}".format(best_result, best_params, name))

    return best_params
