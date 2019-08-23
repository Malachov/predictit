#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def analyze_data(data, lags=50, window=30, predicted_column_index=0):

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data=data)

    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(data[data.columns[predicted_column_index]])
    plt.xlabel('t')
    plt.ylabel("f(x)")

    plt.subplot(1, 2, 2)
    sns.distplot(data[data.columns[predicted_column_index]], bins=100, kde=True, color='skyblue')
    plt.xlabel('f(x)')
    plt.ylabel("Rozdělení") 

    plt.tight_layout()
    plt.suptitle("Vykreslení funkce a jejího rozdělení", fontsize=20)
    plt.subplots_adjust(top=0.88)
    plt.show()

    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(10,5))

    try:

        plot_acf(data[data.columns[predicted_column_index]], lags=lags, ax=ax)
        ax.set_xlabel('Lag')
        plot_pacf(data[data.columns[predicted_column_index]], lags=lags, ax=ax2)
        ax2.set_xlabel('Lag')
        plt.show()

    except Exception as excp:
        print(excp, 'Wrong datatype for more stats or more lags, than values')



    print('Data description', data.describe())
    print('\n Data tail', data.tail())
    print('\n Nan values in columns', data.isna().sum())  # Print??


    # Moving average
    rolling_mean = data[data.columns[predicted_column_index]].rolling(window).mean()  #[0]
    rolling_std = data[data.columns[predicted_column_index]].rolling(window).std()  #[0]

    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(rolling_mean)
    plt.xlabel('t')
    plt.ylabel("Plovoucí průměr x")

    plt.subplot(1, 2, 2)
    plt.plot(rolling_std)
    plt.xlabel('f(x)')
    plt.ylabel("Plovoucí směrodatná odchylka x")

    plt.tight_layout()
    plt.suptitle("Vykreslení funkce a jejího rozdělení", fontsize=20)
    plt.subplots_adjust(top=0.88)
    plt.show()

    # Dick Fuller test na stacionaritu
    pvalue = adfuller(data[data.columns[predicted_column_index]])[1]
    cutoff=0.01
    if pvalue < cutoff:
        print('p-value = {} : Data series is probably stationary'.format(pvalue))
    else:
        print('p-value = {} : Data series is probably stationary'.format(pvalue))

    ### Korelační matice obrázek
    sns.pairplot(data, diag_kind="kde")
    plt.show()

def decompose(data, freq=365, model='additive'):

    decomposition = seasonal_decompose(data, model=model, freq=freq)

    plt.figure(figsize=(15,8))
    plt.subplot(4, 1, 1)
    plt.plot(decomposition.observed)
    plt.xlabel('Datum')
    plt.ylabel("Skutečné hodnoty")

    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.xlabel('Datum')
    plt.ylabel("Trend")

    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.xlabel('Datum')
    plt.ylabel("Sezónnost")

    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.xlabel('Datum')
    plt.ylabel("Rezidua")

    plt.suptitle("Sezónní dekompozice", fontsize=20)
    plt.subplots_adjust(top=0.88)
    
    plt.show()
