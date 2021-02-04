"""Module with function compare_predicted_to_test that Compare tested model with reality. It return some error criterion based on Config.py"""

import numpy as np
import mylogging
from predictit import misc
import warnings


def compare_predicted_to_test(predicted, test, error_criterion='mape', plot=0, modelname="Default model", dataname="default data"):
    """Compare tested model with reality.

    Args:
        predicted (np.ndarray): Model output.
        test (np.ndarray): Correct values or output from data_pre funcs.
        error_criterion (str, optional): 'mape' or 'rmse'. Defaults to 'mape'.
        plot (int, optional): Whether create plot. Defaults to 0.
        modelname (str, optional): Model name for plot. Defaults to "Default model".
        dataname (str, optional): Data name for plot. Defaults to "default data".

    Returns:
        float: Error criterion value (mape or rmse).
        If in setup - return  plot of results.
    """

    predicts = len(predicted)

    if predicts != len(test):
        print('Test and predicted length not equeal')
        return np.nan

    if predicted is not None:
        if plot:

            if misc._JUPYTER:
                from IPython import get_ipython
                get_ipython().run_line_magic('matplotlib', 'inline')

            if misc._IS_TESTED:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import matplotlib
                    matplotlib.use('agg')

            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(test, label='Reality')
            plt.plot(predicted, label='Prediction')
            plt.legend(loc="upper right")
            plt.xlabel('t')
            plt.ylabel("Predicted value")
            plt.title("Prediction with \n {} with data {}".format(modelname, dataname))
            plt.show()

        error = np.array(predicted) - np.array(test)

        '''
        abserror = [abs(i) for i in error]
        sumabserror = sum(abserror)
        mae = sumabserror / predicts
        '''

        if error_criterion == 'mse_sklearn':
            from sklearn.metrics import mean_squared_error
            criterion_value = mean_squared_error(test, predicted)

        elif error_criterion == 'rmse':
            rmseerror = error ** 2
            criterion_value = (sum(rmseerror) / predicts) ** (1 / 2)

        elif error_criterion == 'mape':
            no_zero_test = np.where(abs(test)>=1, test, 1)
            criterion_value = np.mean(np.abs((test - predicted) / no_zero_test)) * 100

        elif error_criterion == 'dtw':

            try:
                from dtaidistance import dtw
            except Exception:
                raise ImportError(mylogging.return_str(
                    "Library dtaidistance necessary for configured dtw (dynamic time warping) "
                    "error criterion is not installed! Instal it via \n\npip install dtaidistance"))

            predicted_double = predicted.astype('double')
            test_double = test.astype('double')
            criterion_value = dtw.distance_fast(predicted_double, test_double)

        else:
            raise KeyError(mylogging.return_str(f"bad 'error_criterion' Config - '{error_criterion}'. Use some from options from config comment..."))

        return criterion_value
