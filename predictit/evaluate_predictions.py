"""Module with function compare_predicted_to_test that Compare tested model with reality. It return some error criterion based on config.py"""

import matplotlib.pyplot as plt
import numpy as np


def compare_predicted_to_test(predicted, test, train=None, criterion='mape', plot=0, modelname="Default model", dataname="default data", details=0):
    """Compare tested model with reality.

    Args:
        predicted (np.ndarray): Model output.
        test (np.ndarray): Correct values or output from data_pre funcs.
        train (np.ndarray, optional): Real history values for plotting - for plot olny!. Defaults to None.
        criterion (str, optional): 'mape' or 'rmse'. Defaults to 'mape'.
        plot (int, optional): Whether create plot. Defaults to 0.
        modelname (str, optional): Model name for plot. Defaults to "Default model".
        dataname (str, optional): Data name for plot. Defaults to "default data".
        details (int, optional): Whether print details. Defaults to 0.

    Returns:
        float: Error criterion value (mape or rmse).
        If in setup - return  plot of results.

    """

    predicts = len(predicted)

    if (len(predicted) != len(test)):
        print('Test and predicted lenght not equeal')
        return np.nan

    if predicted is not None:
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(test, label='Reality')
            plt.plot(predicted, label='Prediction')
            plt.legend(loc="upper right")
            plt.xlabel('t')
            plt.ylabel("Predicted value")
            plt.title("Prediction with \n {} with data {}".format(modelname, dataname))
            plt.show()

            if train is not None:  # TODO delete if work also for date
                plt.figure(figsize=(10, 6))
                tt = range(len(predicted) * 10)
                window = len(predicted) * 9
                predictedpluslast = np.insert(predicted, 0, train[-1])

                plt.plot(tt[:window], train[-window:])
                plt.plot(tt[window - 1:], predictedpluslast)
                plt.xlabel('t')
                plt.ylabel("Predicted value")
                plt.title("History + predictions with \n {} with data {}".format(modelname, dataname))
                plt.show()

        error = np.array(predicted) - np.array(test)

        '''
        abserror = [abs(i) for i in error]
        sumabserror = sum(abserror)
        mae = sumabserror / predicts
        '''

        if criterion == 'rmse':
            rmseerror = error ** 2
            criterion_value = (sum(rmseerror) / predicts) ** (1 / 2)

        if criterion == 'mape':
            no_zero_test = np.where(abs(test)>=1, test, 1)
            criterion_value = np.mean(np.abs((test - predicted) / no_zero_test)) * 100

        if criterion == 'dwt':
            from dtaidistance import dtw
            criterion_value = dtw.distance_fast(predicted, test)

        if details == 1:
            print(f"Error of model {modelname} on data {dataname}: {criterion}={criterion_value}")

        return criterion_value
