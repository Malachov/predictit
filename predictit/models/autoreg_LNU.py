#%%
import numpy as np


def train(sequentions, predicts=30, mi=1, mi_multiple=1, mi_linspace=(1e-8, 10, 20), epochs=10, w_predict=0, minormit=1, damping=1, plot=0, random=0, w_rand_scope=1, w_rand_shift=0, rand_seed=0):
    """Autoregressive linear neural unit with weight prediction. It's simple one neuron one-step net that predict not only predictions itself,
    but also use other faster method to predict weights evolution. In first iteration it will find best weight and in the second iteration, it will train more epochs.

    Args:
        sequentions (tuple(np.ndarray, np.ndarray, np.ndarray)) - Tuple (X, y, x_input) of input train vectors X, train outputs y, and input for prediction x_input
        predicts (int, optional): Number of predicted values. Defaults to 7.
        mi (float, optional): Learning rate. If not normalized must be much smaller. Defaults to 0.1.
        mi_multiple (bool): If try more weigths and try to find the best.
        mi_linspace (np.ndarray, optional): Used learning rate numpy linspace arguments. Defaults to mi_linspace=(1e-8, 10, 20).
        epochs (int): Numer of trained epochs.
        w_predict (bool): If predict weights with next predictions.
        minormit (int, optional): Whether normalize learning rate. Defaults to 0.
        damping (int, optional): Whether damp learning rate or not. Defaults to 1.
        plot (int, optional): Whether plot results. Defaults to 0.
        random (int, optional): Whether initial weights are random or not. Defaults to 0.
        w_rand_scope (int, optional): w_rand_scope of random weights. Defaults to 1.
        w_rand_shift (int, optional): w_rand_shift of random weights. Defaults to 0.
        rand_seed (int, optional): Random weights but the same everytime. Defaults to 0.

    Returns:
        np.ndarray: Predictions of input time series.
    """
    X = sequentions[0]
    y_hat = sequentions[1]

    if mi_multiple:
        miwide = np.linspace(mi_linspace[0], mi_linspace[1], mi_linspace[2])
    else:
        miwide = np.array([mi])

    miwidelen = len(miwide)
    length = X.shape[0]
    features = X.shape[1]

    y = np.zeros((miwidelen, length))
    e = np.zeros(length)
    w_last = np.zeros((miwidelen, features))
    mi_error = np.zeros(miwidelen)

    bound = 100 * np.amax(X)

    if rand_seed != 0:
        random.seed(rand_seed)

    if random == 1:
        w = np.random.rand(X.shape[1]) * w_rand_scope + w_rand_shift
    else:
        w = np.zeros(X.shape[1])

    for i in range(miwidelen):

        for j in range(length):

            y[i, j] = np.dot(w, X[j])

            if y[i, j] > bound:
                mi_error[i] = np.inf
                break

            e[j] = y_hat[j] - y[i, j]
            dydw = X[j]  # TODO i + 1
            if minormit == 1:
                minorm = miwide[i] / (damping + np.dot(X[j], X[j].T))
                dw = minorm * e[j] * dydw
            else:
                dw = miwide[i] * e[j] * dydw
            w = w + dw

        mi_error[i] = np.sum(abs(e[-predicts * 3:]))
        w_last[i] = w

    bestmiindex = np.argmin(mi_error)

    if epochs:

        y_best_mi = np.zeros(length)
        e_best_mi = np.zeros(length)
        mi_best = miwide[bestmiindex]
        wall = np.zeros((features, length))
        w_best_mi = w_last[bestmiindex]

        for _ in range(epochs):
            for j in range(length):

                y_best_mi[j] = np.dot(w_best_mi, X[j])
                e_best_mi[j] = y_hat[j] - y_best_mi[j]
                dydw = X[j]
                if minormit == 1:
                    minorm = mi_best / (damping + np.dot(X[j], X[j].T))
                    dw = minorm * e_best_mi[j] * dydw
                else:
                    dw = mi_best * e_best_mi[j] * dydw
                w_best_mi = w_best_mi + dw

                wall[:, j] = w_best_mi

    if w_predict:
        from . import statsmodels_autoregressive
        wwt = np.zeros((features, predicts))

        for i in range(features):
            wwt[i] = statsmodels_autoregressive.predict(wall[i], statsmodels_autoregressive.train(wall[i]), predicts=predicts)

        w_final = wwt.T

    else:
        w_final = w_best_mi

    if plot == 1:

        from predictit.misc import _JUPYTER
        if _JUPYTER:
            from IPython import get_ipython
            get_ipython().run_line_magic('matplotlib', 'inline')

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 7))

        plt.subplot(3, 1, 1)
        plt.plot(y_best_mi, label='Predictions')
        plt.xlabel('t')
        plt.plot(y_hat, label='Reality')
        plt.xlabel('t'); plt.ylabel("y"); plt.legend(loc="upper right")

        plt.subplot(3, 1, 2)
        plt.plot(e_best_mi, label='Error'); plt.grid(); plt.xlabel('t')
        plt.legend(loc="upper right"); plt.ylabel("Chyba")

        plt.subplot(3, 1, 3)
        plt.plot(wall)
        plt.grid(); plt.xlabel('t'); plt.ylabel("Weights")

        plt.suptitle("Predictions vs reality, error and weights", fontsize=20)
        plt.subplots_adjust(top=0.88)

        plt.show()

    return w_final


def predict(x_input, model, predicts=7):
    """Function that creates predictions from trained model and input data.

    Args:
        x_input (np.ndarray): Time series data.
        model (list, class): Trained model. It can be list of neural weigths .
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        np.ndarray: Array of predicted results
    """

    x_input = x_input.ravel()

    predictions = np.zeros(predicts)
    predictions.fill(np.nan)

    for j in range(predicts):
        ww = model[j] if model.ndim == 2 else model

        predictions[j] = np.dot(ww, x_input)

        x_input = np.insert(x_input, len(x_input), predictions[j])
        x_input = np.delete(x_input, 1)

    return predictions
