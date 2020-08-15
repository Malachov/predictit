"""This is module for data analysis. It create plots of data, it's distribution, it's details, autocorrelation function etc.
Matplotlib lazyload because not using in gui.
"""

from predictit import misc


def analyze_data(array, df, lags=5, window=5):
    """Function that plot data, it's distribution, some details like minimum, maximum, std, mean etc.
    It also create autocorrelation and partial autocorrelation (good for ARIMA models) and plot rolling mean and rolling std.
    It also tell if data are probably stationary or not.

    Args:
        data (pd.DataFrame): Time series data.
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

    # Data and it's distribution

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(array)
    plt.xlabel('t')
    plt.ylabel("f(x)")

    plt.subplot(1, 2, 2)
    sns.distplot(df, bins=100, kde=True, color='skyblue')
    plt.xlabel('f(x)')
    plt.ylabel("Distribution")

    plt.tight_layout()
    plt.suptitle("Data and it's distribution", fontsize=20)
    plt.subplots_adjust(top=0.88)
    plt.show()

    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    fig.suptitle('Repeating patterns - autocorrelation')

    try:

        plot_acf(df, lags=lags, ax=ax)
        ax.set_xlabel('Lag')
        plot_pacf(df, lags=lags, ax=ax2)
        ax2.set_xlabel('Lag')
        plt.show()

    except Exception as excp:
        print(f'\n Error: {excp} \n Maybe wrong datatype for more stats or more lags, than values')

    print('\n Data description \n', df.describe())
    print('\n Data tail \n', df.tail())
    print('\n Nan values in columns \n', df.isna().sum())  # Print??


    # Moving average
    rolling_mean = df.rolling(window).mean()
    rolling_std = df.rolling(window).std()

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
    plt.suptitle("Rolling average and rolling mean", fontsize=20)
    plt.subplots_adjust(top=0.88)
    plt.show()

    # Dick Fuller test na stacionaritu
    pvalue = adfuller(array)[1]
    cutoff=0.05
    if pvalue < cutoff:
        print(f"\np-value = {pvalue} : Data series is probably stationary.\n")
    else:
        print(f"\np-value = {pvalue} : Data series is probably not stationary \n")


def analyze_correlation(data_df):
    """Plot correlation graph.

    Args:
        data (np.ndarray): Time series data
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Pairplot unfortunately very slow
    # sns.pairplot(data, diag_kind="kde")
    corr = data_df.corr()
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
        misc.traceback_warning("Number of samples is probably too low to compute.")
