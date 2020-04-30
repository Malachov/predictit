import warnings
import traceback
import textwrap
import os

_JUPYTER = 0
_GUI = 0
try:
    __IPYTHON__
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('%load_ext autoreload')
    ipython.magic('%autoreload 2')
    _JUPYTER = 1

except Exception:
    pass

# To enable colors in cmd...
os.system('')


def colorize(message):
    class X(str):
        def __repr__(self):
            return f"\033[93m {message} \033[0m"

    return X(message)


def user_warning(message):
    warnings.warn(f"\033[93m \n\n\t{message}\n\n \033[0m", stacklevel=2)


def traceback_warning(message=''):
    import pygments

    separated_traceback = pygments.highlight(traceback.format_exc(), pygments.lexers.PythonTracebackLexer(), pygments.formatters.Terminal256Formatter(style='friendly'))
    separated_traceback = textwrap.indent(text=f"\n\n{message}\n====================\n\n{separated_traceback}\n====================\n", prefix='    ')

    warnings.warn(f"\n\n\n{separated_traceback}\n\n")


def confidence_interval(data, predicts=7, confidence=0.1, p=1, d=0, q=0):
    """Function to find confidence interval of prediction for graph.

    Args:
        data (np.ndarray): Time series data
        predicts (int, optional): [description]. Defaults to 7.
        confidence (float, optional): [description]. Defaults to 0.1.
        p (int, optional): 1st order of ARIMA. Defaults to 1.
        d (int, optional): 2nd order of ARIMA. Defaults to 0.
        q (int, optional): 3rd order of ARIMA. Defaults to 0.

    Returns:
        list, list: Lower bound, upper bound.

    """

    import statsmodels

    if len(data) <= 10:
        return

    order = (p, d, q)

    try:

        model = statsmodels.tsa.arima_model.ARIMA(data, order=order)
        model_fit = model.fit(disp=0)
        predictions = model_fit.forecast(steps=predicts, alpha=confidence)

        bounds = predictions[2].T
        lower_bound = bounds[0]
        upper_bound = bounds[1]

    except Exception:

        from predictit import data_preparation

        last_value = data[-1]
        data = data_preparation.do_difference(data)

        model = statsmodels.tsa.arima_model.ARIMA(data, order=order)
        model_fit = model.fit(disp=0)
        predictions = model_fit.forecast(steps=predicts, alpha=confidence)

        bounds = predictions[2].T
        lower_bound = data_preparation.inverse_difference(bounds[0], last_value)
        upper_bound = data_preparation.inverse_difference(bounds[1], last_value)


    return lower_bound, upper_bound

    ### Pickle option was removed, add if you need it...
    # if config['pickleit']:
    #     from predictit.test_data.pickle_test_data import pickle_data_all
    #     import pickle
    #     pickle_data_all(config['data_all'], datalength=config['datalength'])

    # if config['from_pickled']:

    #     script_dir = Path(__file__).resolve().parent
    #     data_folder = script_dir / "test_data" / "pickled"

    #     for i, j in config['data_all'].items():
    #         file_name = i + '.pickle'
    #         file_path = data_folder / file_name
    #         try:
    #             with open(file_path, "rb") as input_file:
    #                 config['data_all'][i] = pickle.load(input_file)
    #         except Exception:
    #             traceback_warning(f"Test data not loaded - First in config['py'] pickleit = 1, that save the data on disk, then load from pickled.")
