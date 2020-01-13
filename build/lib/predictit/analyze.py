"""This is module for data analysis. It create plots of data, it's distribution, it's details, autocorrelation function etc.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# TODO plotly on input data
# def plot_input_data():


def analyze_data(data, lags=5, window=5):
    """Function that plot data, it's distribution, some details like minimum, maximum, std, mean etc.
    It also create autocorrelation and partial autocorrelation (good for ARIMA models) and plot rolling mean and rolling std.
    It also tell if data are probably stationary or not.

    Args:
        data (pd.DataFrame): Time series data.
        lags (int, optional): Lags used for autocorrelation. Defaults to 5.
        window (int, optional): Window for rolling average and rolling std. Defaults to 5.

    """

    if not isinstance(data, pd.DataFrame):
        data_frame = pd.DataFrame(data=data)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(data)
    plt.xlabel('t')
    plt.ylabel("f(x)")

    plt.subplot(1, 2, 2)
    sns.distplot(data, bins=100, kde=True, color='skyblue')
    plt.xlabel('f(x)')
    plt.ylabel("Distribution")

    plt.tight_layout()
    plt.suptitle("Data and it's distribution", fontsize=20)
    plt.subplots_adjust(top=0.88)
    plt.show()

    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    try:

        plot_acf(data, lags=lags, ax=ax)
        ax.set_xlabel('Lag')
        plot_pacf(data, lags=lags, ax=ax2)
        ax2.set_xlabel('Lag')
        plt.show()

    except Exception as excp:
        print(f'\n Error: {excp} \n Maybe wrong datatype for more stats or more lags, than values')

    print('\n Data description \n', data_frame.describe())
    print('\n Data tail \n', data_frame.tail())
    print('\n Nan values in columns \n', data_frame.isna().sum())  # Print??


    # Moving average
    rolling_mean = data_frame.rolling(window).mean()
    rolling_std = data_frame.rolling(window).std()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rolling_mean)
    plt.xlabel('t')
    plt.ylabel("Rolling average x")

    plt.subplot(1, 2, 2)
    plt.plot(rolling_std)
    plt.xlabel('f(x)')
    plt.ylabel("Rolling standard deviation x")

    plt.tight_layout()
    plt.suptitle("Function and it's distribution plotting", fontsize=20)
    plt.subplots_adjust(top=0.88)
    plt.show()

    # Dick Fuller test na stacionaritu
    pvalue = adfuller(data)[1]
    cutoff=0.01
    if pvalue < cutoff:
        print('p-value = {} : Data series is probably stationary'.format(pvalue))
    else:
        print('p-value = {} : Data series is probably stationary'.format(pvalue))


def analyze_correlation(data):
    """Plot correlation graph.
    
    Args:
        data (np.ndarray): Time series data
    """    
    sns.pairplot(data, diag_kind="kde")
    plt.show()


def decompose(data, freq=365, model='additive'):
    """Plot decomposition graph. Analze if data are seasonal.
    
    Args:
        data (np.ndarray): Time series data
        freq (int, optional): Seasonal interval. Defaults to 365.
        model (str, optional): Additive or multiplicative. Defaults to 'additive'.
    """    
    decomposition = seasonal_decompose(data, model=model, freq=freq)

    plt.figure(figsize=(15, 8))
    plt.subplot(4, 1, 1)
    plt.plot(decomposition.observed)
    plt.xlabel('Date')
    plt.ylabel("Real values")

    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.xlabel('Date')
    plt.ylabel("Trend")

    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.xlabel('Date')
    plt.ylabel("Seasonality")

    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.xlabel('Date')
    plt.ylabel("Residuals")

    plt.suptitle("Seasonal decomposition", fontsize=20)
    plt.subplots_adjust(top=0.88)

    plt.show()
