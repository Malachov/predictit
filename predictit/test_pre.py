import matplotlib.pyplot as plt
import numpy as np

def test_pre(predicted, test, train=None, criterion='mape', plot=0, modelname = "Default model", dataname="default data", details = 0):

    """Compare tested model with reality
    =========
    Output:
    --------
        Error criterion -- MAPE or RMSE{f}
        Graph of predictions and reality (Optional){plot}
        Graph of history and predictions (Optional){plot}

    Arguments:
    --------
        predicted - Model output
        test -- Correct values or output from data\_pre func
        criterion -- 'mape' or 'rmse'
        train -- Real history values for plotting - for plot olny! (Optional)
        plot -- Whether create plot (Optional)
        modelname -- Model name for plot (Optional)
        dataname -- Data name for plot (Optional)
        details -- Whether print details (Optional)
    """

    predicts = len(predicted)
    if predicted is None:
        return np.nan, np.nan, np.nan

    if (len(predicted) != len(test)):
        print('Test and predicted lenght not equeal')
        return np.nan, np.nan, np.nan

    if predicted is not None:
        if plot:
            plt.figure(figsize=(10,6))
            plt.plot(test, label='Skutečnost')
            plt.plot(predicted, label='Predikce')
            plt.legend(loc="upper right")
            plt.xlabel('t')
            plt.ylabel("Predikovaná hodnota")
            plt.title("Predikce pomocí \n {} s daty {}".format(modelname, dataname))
            plt.show()

            if train is not None: # TODO vymazat jestli bude platit i pro date
                plt.figure(figsize=(10,6))
                tt = range(len(predicted) * 10)
                window = len(predicted) * 9
                predictedpluslast = np.insert(predicted, 0, train[-1]) 

                plt.plot(tt[:window], train[-window:])
                plt.plot(tt[window-1:], predictedpluslast)
                plt.xlabel('t')
                plt.ylabel("Predikovaná hodnota")
                plt.title("Historie plus predikce pomocí \n {} s daty {}".format(modelname, dataname))
                plt.show()

        error = np.array(predicted) - np.array(test)

        '''
        abserror = [abs(i) for i in error]
        sumabserror = sum(abserror)
        mae = sumabserror / predicts
        '''

        if criterion == 'rmse':
            rmseerror = error ** 2
            rmse = (sum(rmseerror) / predicts) ** (1/2)

        if criterion == 'mape':
            mape = np.mean(np.abs((test - predicted) / test)) * 100

        if details == 1:
            print("Chyba predikcí modelu {} s daty {}: {}={}".format(modelname, dataname, criterion, mae))

        if criterion == 'mape':
            return mape

        if criterion == 'rmse':
            return rmse
