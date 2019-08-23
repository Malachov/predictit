import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR, _ar_predict_out_of_sample
import numpy as np

def ar(data, predicts=7, plot=0, predicted_column_index=0, method='cmle', ic='aic', trend='nc', solver='lbfgs'):
    """Autoregressive function
    Return [predictions]
    Inputs
        data - Dataframe, CSV, array or list
        predicts - Number of predicted values
        plot - If 1, plot 
    """

    data = np.array(data)
    data_shape = data.shape

    try:
        if len(data_shape) == 1:
            model = AR(data)
        else:
            model = AR(data[predicted_column_index])

        model_fit = model.fit(method=method, ic=ic, trend=trend, solver=solver, disp=-1)
        #window = model_fit.k_ar
        #coef = model_fit.params

        endogg = [i[0] for i in model.endog]
        predictions = _ar_predict_out_of_sample(endogg, model_fit.params, model.k_ar, model.k_trend, steps = predicts, start=0)

        if plot == 1:
            plt.plot(predictions, color='red')
            plt.show()

        predictions = np.array(predictions).reshape(-1)
        return predictions

    except Exception as err:
        print("\t", err)
        return [np.nan] * predicts

        