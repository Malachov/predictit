#%%
import matplotlib.pyplot as plt
import numpy as np
from .sm_ar import ar

def autoreg_LNU_withwpred(data, predicts=7, lags=100, predicted_column_index=0, mi = 0.1, minormit=0, tlumenimi = 1, plot=0, random=0, seed = 0):

    data = np.array(data)
    data_shape = data.shape

    if len(data_shape) > 1:
        data = data[predicted_column_index]

    miwide = np.array([mi * 100, mi * 10, mi, mi / 10, mi / 100, mi/1000, mi/10000, mi/100000, mi/10000000, mi/100000000, mi/100000000000])
    miwidelen = len(miwide)
    leng = len(data)
    y = np.zeros((miwidelen, leng))
    w = np.zeros((miwidelen, lags + 1))
    x = np.zeros((miwidelen, lags + 1))
    e = np.zeros((miwidelen, leng))
    eabs = np.zeros((miwidelen, leng))
    predictions = np.zeros(predicts)
    wall = np.zeros((miwidelen, leng, lags + 1))


    if seed is not 0:
        random.seed(seed)

    for i in range(miwidelen):
        if random == 1:
            w[i] = np.random.rand(lags + 1) * scope + shift # TODO rozsah a posunuti jen u slozitejsich modelu
        x[i][0] = 1
        for j in range(leng): # NOTE nároky na paměť i čas?? možná neponechávat index i ale pouze konečné sumy a hodnoty přepisovat - nebo ponechat pro predikci w??
            y[i][j] = np.dot(w[i], x[i])
            if y[i][j] > 100 * max(data):
                e[i][-1] = 1000000
                break
            e[i][j] = data[j] - y[i][j]
            x[i][2:] = x[i][1:-1]
            if j>0:
                x[i][1] = data[j - 1]
            dydw = x[i]
            if minormit == 1:
                minorm = miwide[i] / (tlumenimi + np.dot(x[i], x[i].T))
                dw = minorm * e[i][j] * dydw
            else:
                dw = miwide[i] * e[i][j] * dydw
            w[i] = w[i] + dw
            wall[i][j][:] = w[i]
    bestmi = [0] * miwidelen
    for k in range(miwidelen):
        eabs[k] = [abs(i) for i in e[k]]
        bestmi[k] = sum(eabs[k][-predicts:])
    bestmivalue = min(bestmi)
    bestmiindex = [i for i, j in enumerate(bestmi) if j == bestmivalue][0]

    wwlenght = lags + 1
    wwhist = np.zeros((lags + 1, leng ))
    wwt = np.zeros((lags + 1, predicts))
    wwhist = wall[bestmiindex].T

    for i in range(lags + 1):
        wwt[i] = ar(wwhist[i], predicts = predicts)

    ww = wwt.T

    for j in range(predicts):
        predictions[j] = np.dot(ww[j], x[bestmiindex])
        x[bestmiindex][2:] = x[bestmiindex][1:-1]
        if i>0:
            x[bestmiindex][1] = y[bestmiindex][-1]

    if plot == 1:
        plt.figure(figsize=(12,7))

        plt.subplot(3, 1, 1)
        plt.plot(y[bestmiindex], label='Predikce'); plt.xlabel('t')
        plt.plot(data, label='Skutečnost'); plt.xlabel('t')
        plt.legend(loc="upper right")
        plt.ylabel("y")

        plt.subplot(3, 1, 2)
        plt.plot(e[bestmiindex], label='Chyba při tvorbě modelu'); plt.grid(); plt.xlabel('t')
        plt.legend(loc="upper right")
        plt.ylabel("Chyba")

        plt.subplot(3, 1, 3)
        plt.plot(wall[bestmiindex]); plt.grid(); plt.xlabel('t')
        plt.ylabel("Hodnoty vah")

        plt.suptitle("Predikovaná vs. skutečná hodnota, chyba a váhy", fontsize=20)
        plt.subplots_adjust(top=0.88)

        plt.show()

    if max(predictions) > 3 * max(data) or min(predictions) < 3 * min(data):
        return None

    return(predictions)