#%%
import numpy as np


def train(sequentions, predicts=7, mi=1, mi_multiple=1, mi_linspace=(1e-4, 10, 20), epochs=10, w_predict=0, minormit=1, damping=1, plot=0, random=0, w_rand_scope=1, w_rand_shift=0, rand_seed=0):
    """Autoregressive linear neural unit with weight prediction. It's simple one neuron one-step net that predict not only predictions itself,
    but also use other faster method to predict weights evolution.

    Args:
        data (np.ndarray): Time series data.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        mi (float, optional): Learning rate. If not normalized must be much smaller. Defaults to 0.1.
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
        miwide = np.arange
        miwide = np.linspace(mi_linspace[0], mi_linspace[1], mi_linspace[2])
    else:
        miwide = np.array([mi])

    miwidelen = len(miwide)
    length = X.shape[0]

    y = np.zeros((miwidelen, length))
    e = np.zeros((miwidelen, length))
    wall = np.zeros((miwidelen, length, X.shape[1]))
    mi_error = np.zeros(miwidelen)


    if rand_seed != 0:
        random.seed(rand_seed)

    if random == 1:
        w = np.random.rand(X.shape[1]) * w_rand_scope + w_rand_shift
    else:
        w = np.zeros((miwidelen, X.shape[1]))

        for i in range(miwidelen):

            for j in range(length):

                y[i][j] = np.dot(w[i], X[j])

                if y[i][j] > 100 * np.amax(X):
                    mi_error[i] = np.inf
                    break

                e[i][j] = y_hat[j] - y[i][j]
                dydw = X[i]  # TODO i + 1
                if minormit == 1:
                    minorm = miwide[i] / (damping + np.dot(X[j], X[j].T))
                    dw = minorm * e[i][j] * dydw
                else:
                    dw = miwide[i] * e[i][j] * dydw
                w[i] = w[i] + dw
                wall[i][j][:] = w[i]

                mi_error[i] = np.sum(abs(e[-predicts:]))
                bestmiindex = np.argmin(mi_error)

        wall_best_mi = wall[bestmiindex]

    if epochs:

        y_best_mi = np.zeros(length)
        e_best_mi = np.zeros(length)
        mi_best = miwide[bestmiindex]

        w_best_mi = wall_best_mi[-1]

        for i in range(epochs):
            for j in range(length):

                y_best_mi[j] = np.dot(w_best_mi, X[j])
                e_best_mi[j] = y_hat[j] - y_best_mi[j]
                dydw = X[j]  # TODO i + 1
                if minormit == 1:
                    minorm = mi_best / (damping + np.dot(X[j], X[j].T))
                    dw = minorm * e_best_mi[j] * dydw
                else:
                    dw = mi_best * e_best_mi[j] * dydw
                w_best_mi = w_best_mi + dw

    if w_predict:
        #from .sm_ar import ar
        from predictit.models import ar
        wwt = np.zeros((X.shape[1], predicts))

        wall_best_mi_t = wall_best_mi.T
        for i in range(X.shape[1]):
            wwt[i] = ar(wall_best_mi_t[i], predicts=predicts)

        w = wwt.T

    else:
        w = w_best_mi

    if plot == 1:

        import plotly as plt

        plt.figure(figsize=(12, 7))

        plt.subplot(3, 1, 1)
        plt.plot(y[bestmiindex], label='Predictions')
        plt.xlabel('t')
        plt.plot(y_hat, label='Reality')
        plt.xlabel('t'); plt.ylabel("y"); plt.legend(loc="upper right")

        plt.subplot(3, 1, 2)
        plt.plot(e_best_mi, label='Error'); plt.grid(); plt.xlabel('t')
        plt.legend(loc="upper right"); plt.ylabel("Chyba")

        plt.subplot(3, 1, 3)
        plt.plot(wall[bestmiindex])
        plt.grid(); plt.xlabel('t'); plt.ylabel("Weights")

        plt.suptitle("Predictions vs reality, error and weights", fontsize=20)
        plt.subplots_adjust(top=0.88)

        plt.show()

    return w



def predict(x_input, w, predicts=7):

    x_input = x_input.ravel()

    predictions = np.zeros(predicts)
    predictions.fill(np.nan)

    for j in range(predicts):
        ww = w[j] if w.ndim == 2 else w

        predictions[j] = np.dot(ww, x_input)

        x_input = np.insert(x_input, len(x_input), predictions[j])
        x_input = np.delete(x_input, 1)

    return predictions
