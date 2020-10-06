"""This is module for data analysis. It create plots of data, it's distribution, it's details, autocorrelation function etc.
Matplotlib lazyload because not using in gui.
"""

import numpy as np
import pandas as pd
import mylogging
import mydatapreprocessing

from predictit import misc


def analyze_column(data, lags=5, window=5):
    """Function one-dimensional data (predicted column), that plot data, it's distribution, some details like minimum, maximum, std, mean etc.
    It also create autocorrelation and partial autocorrelation (good for ARIMA models) and plot rolling mean and rolling std.
    It also tell if data are probably stationary or not.

    Args:
        data (np.array, pd.DataFrame): Time series data.
        lags (int, optional): Lags used for autocorrelation. Defaults to 5.
        window (int, optional): Window for rolling average and rolling std. Defaults to 5.

    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_pacf
    from statsmodels.tsa.stattools import adfuller
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    try:
        from IPython import get_ipython
        if misc._JUPYTER:
            get_ipython().run_line_magic('matplotlib', 'inline')
    except Exception:
        pass

    data = np.array(data)

    if data.ndim != 1 and 1 not in data.shape:
        raise ValueError(mylogging.user_message(
            "Select column you want to analyze",
            caption="analyze_data function only for one-dimensional data!"))

    data = data.ravel()

    print(f"Length:  {len(data)} \n"
          f"Minimum:  {np.nanmin(data)} \n"
          f"Maximun:  {np.nanmax(data)} \n"
          f"Mean:  {np.nanmean(data)} \n"
          f"Std:  {np.nanstd(data)} \n"
          f"First few values:  {data[-5:]} \n"
          f"Middle values:  {data[int(-len(data)/2): int(-len(data)/2) + 5]} \n"
          f"Last few values:  {data[-5:]} \n"
          f"Number of nan (not a number) values: {np.count_nonzero(np.isnan(data))} \n")

    # Data and it's distribution

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
    fig.suptitle('Repeating patterns - autocorrelation')

    try:

        plot_acf(data, lags=lags, ax=ax)
        ax.set_xlabel('Lag')
        plot_pacf(data, lags=lags, ax=ax2)
        ax2.set_xlabel('Lag')
        plt.show()

    except Exception:
        mylogging.traceback_warning("Error in analyze_column function - in autocorrelation function: Maybe more lags, than values")

    # Moving average
    rolling_mean = np.sum(mydatapreprocessing.preprocessing.rolling_windows(data, window), 1)
    rolling_std = np.std(mydatapreprocessing.preprocessing.rolling_windows(data, window), 1)

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
    plt.suptitle("Rolling average and rolling standard deviation", fontsize=20)
    plt.subplots_adjust(top=0.88)
    plt.show()

    # Dick Fuller test na stacionaritu
    pvalue = adfuller(data)[1]
    cutoff=0.05
    if pvalue < cutoff:
        print(f"\np-value = {pvalue} : Analyzed column is probably stationary.\n")
    else:
        print(f"\np-value = {pvalue} : Analyzed column is probably not stationary \n")


def analyze_data(data, pairplot=0):
    """Analyze n-dimendional data. Describe data types, nan values, minimumns etc...
    Plot correlation graph.

    Args:
        data (pd.DataFrame, np.ndarray): Time series data.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    data = pd.DataFrame(data)

    print('\n Data description \n', data.describe(include='all'))
    print('\n Data tail \n', data.tail())
    print('\n Nan values in columns \n', data.isna().sum())

    # Pairplot unfortunately very slow
    if pairplot:
        sns.pairplot(data, diag_kind="kde")

    corr = data.corr()
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()


def decompose(data, period=365, model='additive'):
    """Plot decomposition graph. Analze if data are seasonal.

    Args:
        data (np.ndarray): Time series data
        period (int, optional): Seasonal interval. Defaults to 365.
        model (str, optional): Additive or multiplicative. Defaults to 'additive'.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    import matplotlib.pyplot as plt

    try:
        decomposition = seasonal_decompose(data, model=model, period=period)

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

    except ValueError:
        mylogging.traceback_warning("Number of samples is probably too low to compute.")
