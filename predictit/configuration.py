""" All config for predictit. You can setup input data, models to use, whether you want find optimal parameters etc.
Setting can be inserted as function parameters, then it has higher priority.
All values are commented and easy to understand. You can use intellisense and help in docstrings, where you can
find what options or type should variable be and what it means.

If you use predictit as downloaded script, edit, save and run function from main...

Config work for `predict()` function as well as for `predict_multiple_columns()` and `compare_models()`

Examples:
=========

    >>> import predictit
    ...
    >>> config.update({"data": "test_sin", "predicted_column": 0, "used_models": ["AR", "LNU", "Sklearn regression"]})

    This is how you can access values

    >>> config.data_input.datalength = 2000

    You can access all attributes not only from subcategories, but also from root config.

    >>> config.datalength = 1000

    You can also use dict notation

    >>> config['datalength'] = 1000

    config object from here will be used in predictit functions like predict, predict_multiple or compare_predictions

    You can also pass configuration values as dict kwargs...

    >>> predictions = predictit.main.predict(predicts=7, header=0)

    If you want to run more configurations in loop for example, you can create new instances of Config class,
    but you have to pass it then like function params

    >>> new_config = config.copy()  # or Config()
    >>> new_config.update({'predicts': 2, 'print_result_details':True})
    >>> predictions = predictit.main.predict(config=new_config)
    <BLANKLINE>
    Best model is...
"""

from typing import Union, Any, Dict, List, Set, Tuple
from pathlib import Path

from mypythontools.config import MyProperty, ConfigBase, ConfigStructured


class Config(ConfigStructured):
    """Config class. You shoud not use class itself, but created instance config."""

    def __init__(self, init_dict=None) -> None:
        self.general = self.General()
        self.data_input = self.DataInput()
        self.feature_engineering = self.FeatureEngineering()
        self.output = self.Output()
        self.prediction = self.Prediction()
        self.variable_optimization = self.VariableOptimization()
        self.hyperparameter_optimization = self.HyperparameterOptimization()
        self.models = self.Models()
        self._internal = self._Internal()

        if init_dict:
            self.update(init_dict)

    class General(ConfigBase):
        """Various config values that doesn't fit into any other category."""

        @MyProperty(options=["predict", "predict_multiple", "compare_models"])
        def used_function() -> str:
            """
            Options:
                'predict', 'predict_multiple', 'compare_models'

            Default:
                'predict'

            If running main.py script, execute this function. If import as library, ignore - just use function.
            """
            return "predict"

        @MyProperty(options=["fast", "normal", None])
        def use_config_preset() -> Union[str, None]:
            """
            Options:
                'fast', 'normal', None

            Default:
                None

            Edit some selected Config, other remains the same, check config_presets.py file."""

            return None

        @MyProperty(options=["pool", "process", None])
        def multiprocessing() -> Union[bool, str]:
            """
            Options:
                'pool', 'process', None.

            Default:
                None

            Don't use 'process' on windows. Multiprocessing beneficial mostly on bigger data and linux..."""
            return None

        @MyProperty((int, None))
        def processes_limit() -> Union[int, None]:
            """
            Types:
                str | None

            Default:
                None

            Max number of concurrent processes. If None, then (CPUs - 1) is used"""

            return None

        ### Data anlalysis
        @MyProperty(int)
        def analyzeit() -> int:
            """
            Options:
                0, 1, 2, 3

            Default:
                0

            If 0, do not analyze, if 1, analyze original data, if 2 analyze preprocessed data, if 3, then both. Statistical distribution,
            autocorrelation, seasonal decomposition etc.
            """
            return 0

        @MyProperty((dict, None))
        def analyze_seasonal_decompose() -> Union[dict, None]:
            """
            Types:
                dict | None

            Default::

                {"period": 365, "model": "additive"}

            Parameters for seasonal decompose in analyze. Find if there are periodically repeating patterns
            in data.
            """
            return {
                "period": 365,
                "model": "additive",
            }

        @MyProperty(bool)
        def return_internal_results() -> bool:
            """
            Type:
                bool

            Default:
                False

            For developers and tests. Return internal results from prediction process."""
            return False

        @MyProperty(bool)
        def trace_processes_memory() -> bool:
            """
            Type:
                bool
            Default:
                False

            Add how much memory was used by each model."""

            return False

    class DataInput(ConfigStructured):
        """Define what data you will use and how it will be preprocessed. You can use local file, url with data or database.
        Main config variable is `data` where you can find what formats are supported with some examples."""

        def __init__(self) -> None:
            self.database_subconfig = self.DatabaseSubconfig()

        @MyProperty()
        def data():
            """
            Types:
                str | pathlib.Path | np.ndarray | pandas.DataFrame | dict | list

            Default:
                'test'

            Examples::

                myarray_or_dataframe  # Numpy array or Pandas.DataFrame
                r"/home/user/my.json"  # Local file. The same with .parquet, .h5, .json or .xlsx.  On windows it's necessary to use raw string - 'r' in front of string because of escape symbols \
                "https://yoururl/your.csv"  # Web url (with suffix). Same with json.
                "https://blockchain.info/unconfirmed-transactions?format=json"  # In this case you have to specify also  'request_datatype_suffix': "json", 'data_orientation': "index", 'predicted_table': 'txs',
                [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}]  # List of records
                {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}  # Dict with colums or rows (index) - necessary to setup data_orientation!
                You can use more files in list and data will be concatenated. It can be list of paths or list of python objects. Example:
                [np.random.randn(20, 3), np.random.randn(25, 3)]  # Dataframe same way
                ["https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures2.csv"]  # List of URLs
                ["path/to/my1.csv", "path/to/my1.csv"]

            File path with suffix (string or pathlib Path). Or numpy array, pandas dataframe or series, list or dictionary.

            Supported path formats are:

            - csv
            - xlsx and xls
            - json
            - parquet
            - h5

            Data shape for numpy array and dataframe is (n_samples, n_features). Rows are samples and columns are features.

            Note:
                If you want try how it works, you can use some default test data. You can use 'get_ecg', 'test_sin', 'test_ramp' and 'test_random'."""

            return "test_ecg"

        @MyProperty(types=(list, dict, None))
        def data_all() -> Union[List, Dict]:
            """
            Types:
                list | dict | None

            Default:
                None

            Examples::

                {data_1 = (my_dataframe, 'column_name_or_index')}
                (my_data[-2000:], my_data[-1000:])
                [np.array(range(1000)), np.array(range(500))]

            Just for compare_models function. Dictionary of data names and list of it's values and predicted columns or list of
            data parts or numpy array with rows as data samples. Don't forget to setup 'predicted_column' as usually in Config.
            """
            return None

        @MyProperty((str, None))
        def predicted_table() -> Union[str, None]:
            """
            Types:
                str | None

            Default:
                None

            If using excel (xlsx) - it means what sheet to use, if json, it means what key values, if SQL, then it
            mean what table. Else it have no impact."""
            return None

        @MyProperty(options=["columns", "index", None])
        def data_orientation() -> str:
            """
            Options:
                'columns', 'index', None.

            Default:
                None

            If using json or dictionary, it describe how data are oriented. Default is 'columns' if None used.
            If orientation is records (in pandas terminology), it's detected automatically."""

            return None

        @MyProperty((int, None))
        def header() -> Union[str, bool]:
            """
            Types:
                int | None

            Default:
                'infer'

            Row index that will be used as column names."""
            return "infer"

        @MyProperty(dict)
        def csv_style() -> Dict:
            """
            Type:
                dict

            Default::

                {'sep' : ',', 'decimal': '.'}

            Examples:
                En locale usually use
                `{'sep': ',', 'decimal': '.'}`
                Some Europian country use
                `{'sep': ';', 'decimal': ','}`

            Define CSV separators."""

            return {"sep": ",", "decimal": "."}

        @MyProperty((str, None))
        def request_datatype_suffix() -> Union[str, None]:
            """
            Options:
                'csv', 'json', 'xlsx', 'xls', 'parquet', 'h5', None

            Default:
                None

            If using url with no extension, define which datatype is on this url with GET request."""

            return None

        @MyProperty((str, int))
        def predicted_column():
            """
            Types:
                str | int

            Default:
                0

            Name of predicted column (for dataframe data) or it's index - string or int. May also be pandas index"""
            return 0

        @MyProperty()
        def predicted_columns():
            """
            Types:
                list, None

            Default:
                ['*']

            For predict_multiple function only! List of names of predicted columns or it's indexes. If 'data' is
            dataframe or numpy, then you can use ['*'] or '*' to predict all the columns with numbers. If None,
            then predicted_column will be used."""

            return ["*"]

        @MyProperty((int, str, None))
        def datetime_column() -> Union[str, int, None]:
            """
            Types:
                int | str | None

            Default:
                None

            Index of dataframe datetime column or it's name if it has datetime column. If there already is index
            in input data, it will be used automatically. Data are sorted by time."""
            return None

        @MyProperty((str, None))
        def freq() -> Union[str, None]:
            """
            Types:
                str | None

            Default:
                None

            Interval for predictions 'M' - months, 'D' - Days, 'H' - Hours. Resample data if datetime column available.

            Possible values: 'D': calendar day, 'W' weekly, 'M' month end, 'Q' quarter, 'Y' year, 'H' hourly,
            'min' minutes, 'S' secondly

            You can also use another frequencies that pandas use.
            """
            return None

        @MyProperty((list, None))
        def freqs() -> List[None]:
            """
            Types:
                list | None

            Default:
                []

            For predict_multiple function only! List of intervals of predictions 'M' - months, 'D' - Days,
            'H' - Hours. Same possible values as in freq."""
            return []

        @MyProperty(options=("sum", "mean", "", None))
        def resample_function() -> Union[str, None]:
            """
            Options:
                'sum', 'mean', None

            Default:
                'sum'

            For example if current in technological process - mean, if units sold, then sum."""
            return "sum"

        @MyProperty(int)
        def datalength() -> int:
            """
            Type:
                int

            Default:
                1000

            The length of the data used for prediction (after resampling). If 0, than full length."""
            return 1000

        @MyProperty(int)
        def max_imported_length() -> int:
            """
            Type:
                int

            Default:
                100000

            Max length of imported samples (before resampling). If 0, than full length."""
            return 100000

        ### Data inputs definition
        @MyProperty(int)
        def default_n_steps_in() -> int:
            """
            Type:
                int

            Default:
                7

            How many lagged values are in vector input to model."""
            return 7

        @MyProperty(bool)
        def other_columns() -> bool:
            """
            Type:
                Bool

            Default:
                True

            If use other columns. Some models has data input settings, that already use just one column,
            but this force it on allmodels..."""
            return True

        @MyProperty(int)
        def default_other_columns_length() -> int:
            """
            Type:
                int

            Default:
                2

            Other columns vector length used for predictions. If None, lengths same as predicted columnd.
            If 0, other columns are not used for prediction."""
            return 2

        @MyProperty(options=["float32", "float64"])
        def dtype() -> str:
            """
            Type:
                str

            Default:
                'float32'

            Main dtype used in prediction. If 0, None or False, it will keep original. Eg.'float32' or 'float64'."""
            return "float32"

        @MyProperty(float)
        def unique_threshlold() -> float:
            """
            Type:
                float

            Default:
                0.1

            Remove string columns, that have to many categories. E.g 0.1 define, that has to be less that 10% of unique values.
            It will remove ids, hashes etc."""
            return 0.1  #

        @MyProperty(options=["label", "one-hot"])
        def embedding() -> str:
            """
            Options:
                'label', 'one-hot'

            Default:
                'label'

            Categorical encoding. Create numbers from strings. 'label' give each category (unique string) concrete number.
            Result will have same number of columns. 'one-hot' create for every category new column.
            """
            return "label"

        @MyProperty(float)
        def remove_nans_threshold() -> float:
            """
            Type:
                Float

            Default:
                0.2

            From 0 to 1. How much not nans (not a number) can be in column to not be deleted.
            For example if 0.9 means that columns has to have more than 90% values that are not nan to not be deleted."""
            return 0.2

        @MyProperty()
        def remove_nans_or_replace() -> Any:
            """
            Options:
                'mean', 'interpolate', 'neighbor', 'remove' | float | int

            Default:
                'mean'

            After removing columns over nan threshold, this will remove or replace rest nan values. If 'mean',
            replace with mean of column where nan is, if 'interpolate', it will return guess value based on
            neighbors. 'neighbor' use value before nan. Remove will remove rows where nans, if value set, it
            will replace all nans for example with 0.
            """
            return "mean"

        @MyProperty((None, str))
        def data_transform() -> Union[None, str]:
            """
            Options:
                'difference', None

            Default:
                None

            'difference' transform the data on differences between two neighbor values."""
            return None

        @MyProperty(options=[None, "standardize", "-11", "01", "robust"])
        def standardizeit() -> Union[str, bool]:
            """
            Options:
                'standardize', '-11', '01', 'robust', None

            Default:
                'standardize'

            How to standardize data to have similiar scopes."""

            return "standardize"

        @MyProperty((tuple, None))
        def smoothit() -> Union[Tuple[int], None]:
            """
            Type:
                tuple[int, int]

            Default:
                None

            Example:
                (11, 2)

            Smoothing data with Savitzky-Golay filter. First argument is window (must be odd!) and second is
            polynomial order. If None, not smoothing."""
            return None

        @MyProperty(bool)
        def power_transformed() -> bool:
            """
            Type:
                bool

            Default:
                False

            Whether transform results to have same standard deviation and mean as train data."""
            return False

        @MyProperty((int, float, None))
        def remove_outliers() -> Union[int, float, None]:
            """
            Types:
                int | float | None

            Default:
                None

            Remove extraordinary values. Value is threshold for ignored values. Value means how many
            standard deviations from the mean is the threshold (have to be > 3, else delete most of data).
            If predicting anomalies and want to keep the outliers, use None."""
            return None

        # @MyProperty(bool)
        # def last_row() -> bool:
        #     """
        #     Type:
        #         bool

        #     Default:
        #         False

        #     If False, erase last row (for example in day frequency, day is not complete yet)."""
        #     return False

        @MyProperty((None, int))
        def bins() -> Union[int, None]:
            """
            Types:
                int | None

            Default:
                None

            It will sort data values into limited number of bins (imagine histogram).
            It's necessary if using classifier.

            If use int, it means number of bins for every columns.

            You can define binning_type for how the intervals of bins will be evaluated.

            """
            return None

        @MyProperty(options=["cut", "qcut"])
        def binning_type() -> str:
            """
            Options:
                'cut', 'qcut'

            Default:
                'cut'

            If using a bins config variable, sort data values into limited number of bins (imagine histogram).
            This define how the intervals of bins are evaluated. "cut" for equal size of bins intervals
            (different number of members in bins) or "qcut" for equal number of members in bins and various
            size of bins. It uses pandas cut or qcut function in background. Default: 'cut'.
            """

            return "cut"

        class DatabaseSubconfig(ConfigBase):
            """If using database as data source, define server, database, authorization etc.

            Supported databases:
            - mssql (Microsoft SQL server)"""

            @MyProperty(str)
            def server():
                """
                Type:
                    str

                Default:
                    '.'

                If using SQL, sql credentials"""
                return "."

            @MyProperty(str)
            def database():
                """
                Type:
                    str

                Default:
                    ''

                Database name."""
                return ""

            @MyProperty(str)
            def username():
                """
                Type:
                    str

                Default:
                    ''

                Used username. 'sa' for mssql root."""
                return ""

            @MyProperty(str)
            def password():
                """
                Type:
                    str

                Default:
                    ''

                Used password."""
                return ""

            @MyProperty(bool)
            def trusted_connection():
                """
                Type:
                    bool

                Default:
                    False

                If using windows credentials login in mssql. Use this or username and password."""
                return False

    class FeatureEngineering(ConfigBase):
        """This confis is for extension of original data with new derived values that can help to better
        prediction. It's able to add rolling mean and std, fast fourier results on running window, first and
        second derivation or multiplications of columns."""

        @MyProperty((float, int, None))
        def correlation_threshold():
            """
            Types:
                float | int | None

            Default:
                None

            If evaluating from more collumns, use only collumns that are correlated. From 0 (All columns included)
            to 1 (only column itself)"""
            return None

        ### Data extension = add derived columns
        @MyProperty((int, None))
        def add_fft_columns(self):
            """
            Type:
                int | None

            Default:
                None

            Example:
                64

            Whether add fast fourier results on rolling window - maximum and maximum of shift. If None, no columns will be added.
            If int value, it means used rolling window. It will add collumns with maximum value and maximum shift on used window."""
            return None

        @MyProperty((dict, None))
        def data_extension() -> Union[dict, None]:
            """
            Types:
                dict | None

            Default:
                None

            Example::

                {
                    "differences": True,  # Eg. from [1, 2, 3] create [1, 1]
                    "second_differences": True,  # Add second ddifferences
                    "multiplications": True,  # Add all combinations of multiplicated columns
                    "rolling_means": 10,  # int define used window length or None
                    "rolling_stds": 10,  # int define used window length or None
                    "mean_distances": True  # Distance from average
                }

            Add new derived column to data that can help to better predictions. Check examples for what columns you can derive."""
            return None

    class Output(ConfigStructured):
        """Setup outputs of main functions like plot, logger or what should be printed."""

        def __init__(self) -> None:
            self.logger_subconfig = self.LoggerSubconfig()
            self.plot_subconfig = self.PlotSubconfig()
            self.print_subconfig = self.PrintSubconfig()
            self.print_subconfig_compare_models = self.PrintSubconfigCompareModels()

        class LoggerSubconfig(ConfigBase):
            """Predictit uses mylogging library on background. Check it's documentation for more info."""

            @MyProperty(options=["DEBUG", "INFO", "WARNING", "ERROR", "FATAL"])
            def logger_level() -> str:
                """
                Options:
                    'DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'

                Default:
                    'WARNING'

                You can filter out logs based on level you configure.
                """

                return "WARNING"

            @MyProperty(str, options=["once", "ignore", "always", "error"])
            def logger_filter():
                """
                Options:
                    'once', 'ignore', 'always', 'error'

                Default:
                    'once'

                For debug reasons - use proper level. If 'error', stops on first warning (log)."""

                return "once"

            @MyProperty([str, Path])
            def logger_output() -> Union[str, Path]:
                """
                Types:
                    str | pathlib.Path

                Default:
                    'console'

                Where logger messages are stored / printed. 'console' or path to file."""

                return "console"

            @MyProperty(bool)
            def logger_color() -> bool:
                """
                Type:
                    bool

                Default:
                    True

                Whether log output should be colored or not. Some terminals (eg. pytest log or CI/CD log) cannot displays
                colors and long symbols are displayed that are bad for readability."""

                return True

            @MyProperty(list)
            def ignored_warnings():
                """
                Type:
                    list

                Default::

                    [
                        "the imp module is deprecated in favour of importlib",
                        "Using or importing the ABCs from 'collections' instead of from",

                        # Statsmodels
                        "The parameter names will change after 0.12 is released",  # To be removed one day
                        "AR coefficients are not stationary.",
                        "statsmodels.tsa.AR has been deprecated",
                        "divide by zero encountered in true_divide",
                        "pandas.util.testing is deprecated",
                        "statsmodels.tsa.arima_model.ARIMA have been deprecated",
                        "Using or importing the ABCs from 'collections'",
                        # Tensorflow
                        "numpy.ufunc size changed",
                        # Autoregressive neuron
                        "overflow encountered in multiply",
                        # Sklearn
                        "The default value of n_estimators will change",
                        "lbfgs failed to converge",
                        "Pass neg_label=-1",
                        "invalid value encountered in sqrt",
                        "encountered in double_scalars",
                        "Inverting hessian failed",
                        "unclosed file <_io.TextIOWrapper name",
                        "Mean of empty slice",
                    ]

                There can be many warnings from imported libraries. It's possible to filter it.
                At the end, original warnings filters will be restored.

                Just a part of string is enough.
                """
                return [
                    # Internal
                    "sys.settrace() should not be used",
                    # Pandas
                    "A value is trying to be set on a copy of a slice from a DataFrame",
                    # Tensorflow
                    "numpy.ufunc size changed",
                    "Call to deprecated create function EnumValueDescriptor()",
                    # Autoregressive neuron
                    "overflow encountered in multiply",
                    # Matplotlib
                    "Matplotlib is currently using agg",
                    # Statsmodels
                    "The parameter names will change after 0.12 is released",  # To be removed one day
                    "AR coefficients are not stationary.",
                    "statsmodels.tsa.AR has been deprecated",
                    "divide by zero encountered in true_divide",
                    "pandas.util.testing is deprecated",
                    "statsmodels.tsa.arima_model.ARIMA have been deprecated",
                    "Using or importing the ABCs from 'collections'",
                    # Sklearn
                    "The default value of n_estimators will change",
                    "lbfgs failed to converge",
                    "Pass neg_label=-1",
                    "invalid value encountered in sqrt",
                    "encountered in double_scalars",
                    "Inverting hessian failed",
                    "unclosed file <_io.TextIOWrapper name",
                    "Mean of empty slice",
                ]

            @MyProperty(list)
            def ignored_warnings_class_type():
                """
                Type:
                    list

                Default:
                    [("statsmodels.tsa.arima_model", FutureWarning)]

                Sometimes only message does not work, then ignore it with class and warning type
                works only for warnings not for logs...
                """

                return [("statsmodels.tsa.arima_model", FutureWarning)]

        @MyProperty(int)
        def predicts() -> int:
            """
            Type:
                int
            Default:
                7

            Number of predicted values."""
            return 7

        class PlotSubconfig(ConfigBase):
            """Plot settings. You can choose plotly or matplotlib, what plots you want to show, whether you
            want to save it or plotted number of models.

            Whenever you use more plots, jupyter notebook mode can be optimal for displaying plots."""

            @MyProperty(bool)
            def show_plot() -> bool:
                """
                Type:
                    bool

                Default:
                    True

                Whether display plot or not. If in jupyter dislplay in jupyter, else in default browser."""
                return True

            @MyProperty((None, str, Path))
            def save_plot() -> Union[None, str, Path]:
                """
                Types:
                    None | str | pathlib.Path

                Default:
                    None

                If None, do not save, if path as str, save to defined path, if "DESKTOP" save to desktop."""
                return None

            @MyProperty(options=["plotly", "matplotlib"])
            def plot_library() -> str:
                """
                Options:
                    'plotly', 'matplotlib'

                Default:
                    'plotly'

                'plotly' (interactive) or matplotlib (static)."""
                return "plotly"

            @MyProperty(options=["with_history", "just_results"])
            def plot_type() -> str:
                """
                Options:
                    'with_history', 'just_results'

                Default:
                    'with_history'

                'with_history' also plot history as a context for predicitions. 'just_results' plot only predicted data."""
                return "with_history"

            @MyProperty(bool)
            def plot_legend() -> bool:
                """
                Type:
                    bool

                Default:
                    False

                Whether to show description of lines in chart. Even if turned off, it's still displayed on mouseover."""
                return False

            @MyProperty(str)
            def plot_name() -> str:
                """
                Type:
                    str

                Default:
                    'Predictions'

                Used name in plot."""
                return "Predictions"

            @MyProperty(int)
            def plot_history_length() -> int:
                """
                Type:
                    int

                Default:
                    200

                How many historic values will be plotted."""
                return 200

            @MyProperty((None, int))
            def plot_number_of_models() -> Union[None, int]:
                """
                Types:
                    int, None

                Default:
                    12

                Number of plotted models. If None, than all models will be plotted. 1 if only the best one."""
                return 12

        class PrintSubconfig(ConfigBase):
            """What should be printed in `predict` function. Usually table with results is printed. it can be configured."""

            @MyProperty(options=(None, "simple", "detailed"))
            def print_table() -> Union[None, str]:
                """
                Options:
                    None, 'simple', 'detailed'

                Default:
                    'detailed'

                If 'simple', print simple table, option 'detailed' will print detailed
                table with time, optimized Config value etc. None do not print."""
                return "detailed"

            @MyProperty((int, None))
            def print_number_of_models() -> Union[None, int]:
                """
                Types:
                    int | None

                Default:
                    12

                How many models will be printed in results table. If None, than all models will be plotted.
                1 if only the best one."""
                return 12

            @MyProperty(bool)
            def print_time_table() -> bool:
                """
                Type:
                    bool

                Default:
                    True

                Whether print table with models errors and time to compute. In function `compare_models` it's turned off automatically."""
                return True

            @MyProperty(bool)
            def print_result_details() -> bool:
                """
                Type:
                    bool

                Default:
                    True

                Whether print best model results and model details. In function `compare_models` it's turned off automatically."""
                return True

        class PrintSubconfigCompareModels(ConfigBase):
            """What should be printed in `predict` function. Usually table with results is printed. it can be configured."""

            @MyProperty(options=(None, "simple", "detailed"))
            def print_comparison_table() -> Union[None, str]:
                """
                Options:
                    None, 'simple', 'detailed'

                Default:
                    'simple'

                For printed table in `compare_models`. If 'simple', print simple table, option 'detailed' will
                print detailed table. None do not print."""
                return "simple"

            @MyProperty((int, None))
            def print_number_of_comparison_models() -> Union[None, int]:
                """
                Types:
                    int | None

                Default:
                    12

                How many models will be printed in results table. If None, than all models will be plotted.
                1 if only the best one."""
                return 12

            @MyProperty(bool)
            def print_comparison_result_details() -> bool:
                """
                Type:
                    bool

                Default:
                    True

                Whether print best model results and model details. In function `compare_models` it's turned off automatically."""
                return True

        @MyProperty(options=["error", "name"])
        def sort_results_by() -> str:
            """
            Options:
                'error', 'name'

            Default:
                'error'

            Sort results  by 'error' or 'name'"""
            return "error"

        @MyProperty(dict)
        def table_settigs() -> Dict:
            """
            Type:
                dict

            Default::

                {
                    "tablefmt": "grid",
                    "floatfmt": ".3f",
                    "numalign": "center",
                    "stralign": "center",
                }

            Configure table outputs. Check Default value for what keys are possible. Check tabulate for what values can be.
            Options for `tablefmt` are for example `'grid', 'simple', 'pretty', 'psql'` or any other from tabulate library."""
            return {
                "tablefmt": "grid",
                "floatfmt": ".3f",
                "numalign": "center",
                "stralign": "center",
            }

    class Prediction(ConfigBase):
        """Various configs for prediction like number of predicted values, used error criterion etc."""

        @MyProperty((None, float))
        def confidence_interval() -> Union[None, float]:
            """
            Types:
                None | float

            Default:
                0.6

            Area of confidence in result plot (grey area where we suppose values) 0 - 100 % probability area with 0.0 - 1.0 option value.
            Bigger value, narrower area. If None - not plotted."""
            return 0.6

        @MyProperty(options=["mse", "max_error", "mape", "rmse", "dtw"])
        def error_criterion() -> str:
            """
            Options:
                'mse', 'max_error', 'mape', 'rmse', 'dtw'

            Default:
                'mse'

            Equation used to calculate error value. dtw means dynamic time warping.
            """
            return "mse"

        @MyProperty(options=["original", "preprocessed"])
        def evaluate_type() -> str:
            """
            Options:
                'original', 'preprocessed'
            Default:
                'original'

            Define whether error criterion (e.g. RMSE) is evaluated on preprocessed data or on original data."""

            return "original"

        @MyProperty(int)
        def repeatit() -> int:
            """
            Type:
                int

            Default:
                5

            How many times is computation repeated for error criterion evaluation."""

            return 5

        @MyProperty(options=["validate", "predict"])
        def mode() -> str:
            """
            Options:
                'validate', 'predict'

            Default:
                'predict'

            If 'validate', put apart last few 'predicts' values and evaluate on test data
            that was not in train set. Do not setup - use compare_models function, it will use it automattically."""

            return "predict"

    class VariableOptimization(ConfigBase):
        """Any variable from this configuration can be evaluated for more values in loop and best can be
        automatically choosed.

        Note: Can be time-consuming."""

        @MyProperty(bool)
        def optimization() -> bool:
            """
            Type:
                bool

            Default:
                False

            Optimization of some outer Config value e.g. datalength. It will automatically choose the best
            value for each model. Can be time-consuming (every models are computed several times).
            If 0 do not optimize. If 1, compute on various option defined values."""

            return False

        @MyProperty((str, None))
        def optimization_variable() -> str:
            """
            Type:
                str

            Default:
                'default_n_steps_in'

            Some value from config that will be optimized. Unlike hyperparameters only defined values will be computed."""

            return "default_n_steps_in"

        @MyProperty()
        def optimization_values() -> Any:
            """
            Default:
                [4, 8, 12]

            If optimizing some variable, define what particular values should be used for prediction.
            Results for each optimized value are only in detailed table."""

            return [4, 8, 12]

        @MyProperty(bool)
        def plot_all_optimized_models() -> bool:
            """
            Type:
                bool

            Default:
                True
            """
            return True

    class HyperparameterOptimization(ConfigBase):
        """This is for optimization of models parameters for train function e.g. params `n_estimators`,
        `alpha_1`, `lambda_1` in sklearn model. If you want to optimize values from this config (like number
        of lagged models), use VariableOptimization.

        If you want to know how it works, check `optimize` function docstrings from `best_params` module."""

        @MyProperty(bool)
        def optimizeit() -> bool:
            """
            Type:
                bool

            Default:
                False

            Models parameters (hyperparameters) optimization Optimization of inner model hyperparameter e.g.
            number of lags Optimization means, that all the process is evaluated several times. Avoid too much
            combinations. If you optimize for example 8 parameters and divide it into 5 intervals it means
            hundreads of combinations (optimize only parameters worth of it) or do it by parts"""
            return False

        @MyProperty(options=[1, 2, 3])
        def optimizeit_details() -> int:
            """
            Options:
                1, 2, 3

            Default:
                2

            1 print best parameters of models, 2 print every new best parameters achieved, 3 prints all results."""

            return 1

        @MyProperty(bool)
        def optimizeit_plot() -> bool:
            """
            Type:
                bool

            Default:
                False

            Plot every optimized combinations (recommended in interactive way (jupyter) and only if have few
            parameters, otherwise hundreds of plots!!!). Use if you want to know what impact model parameters has."""
            return False

        @MyProperty((float, int, None))
        def optimizeit_limit() -> Union[float, int]:
            """
            Types:
                float | int | None

            Default:
                10

            How many seconds can take one model optimization."""
            return 10

        @MyProperty(int)
        def fragments() -> int:
            """
            Type:
                int

            Default:
                4

            How many values will be evaluated on define interval."""
            return 4

        @MyProperty(int)
        def iterations() -> int:
            """
            Type:
                int

            Default:
                2

            How many times new smaller interval will be created and evaluated."""
            return 2

        @MyProperty(dict)
        def limits_constants(self) -> Dict:
            """
            Type:
                dict

            Default::

                {
                    "models": ["LinearRegression", "BayesianRidge"],
                    "alpha": [0.0, 1.0],
                    "epochs": [1, 300],
                    "units": [1, 100],
                    "order": [0, 20],
                    "maxorder": 20,
                }

            This boundaries repeat across models. Define once here and then use in models_parameters_limits again and again."""
            return {
                "models": ["LinearRegression", "BayesianRidge"],
                "alpha": [0.0, 1.0],
                "epochs": [1, 300],
                "units": [1, 100],
                "order": [0, 20],
                "maxorder": 20,
            }

        @MyProperty(dict)
        def models_parameters_limits(self) -> Dict:
            """
            Type:
                dict

            Default:
                Check returned value (very long dict)

            Example how it works. Function predict_with_my_model(param1, param2).
            If param1 limits are [0, 10], and 'fragments': 5, it will be evaluated for [0, 2, 4, 6, 8, 10].
            Then it finds best value and make new interval that is again divided in 5 parts...
            This is done as many times as iteration value is.

            Note:
                If you need integers, type just number, if you need float, type dot, e.g. [2.0, 6.0].
                If you use list of strings or more than 2 values (e.g. [5, 4, 7]), then only this defined values will be executed
                and no new generated

            Some models can be very computationaly hard - use optimizeit_limit or already_trained!
            If models here are commented, they are not optimized !
            You can optmimize as much parameters as you want - for example just one (much faster).
            """
            return {
                "AR": {
                    "trend": ["c", "nc"],
                },
                # 'ARMA': {'p': [1, self.limits_constants["maxorder"]], 'q': self.limits_constants["order"], 'trend': ['c', 'nc']},
                "ARIMA": {
                    "p": [1, 25],
                    "d": [0, 1, 2],
                    "q": [0, 1, 2],
                    "trend": ["c", "nc"],
                },
                "autoreg": {"cov_type": ["nonrobust", "HC0", "HC1", "HC3"]},
                # 'SARIMAX': {'p': [1, self.limits_constants["maxorder"]], 'd': self.limits_constants["order"], 'q': self.limits_constants["order"], 'pp': self.limits_constants["order"], 'dd': self.limits_constants["order"], 'qq': self.limits_constants["order"],
                # 'season': self.limits_constants["order"], 'method': ['lbfgs', 'bfgs', 'newton', 'nm', 'cg', 'ncg', 'powell'], 'trend': ['n', 'c', 't', 'ct'], 'forecast_type': ['in_sample', 'out_of_sample']},
                "Ridge regression": {"lmbda": [1e-8, 1e6]},
                "Levenberg-Marquardt": {"learning_rate": [0.001, 1]},
                # 'LNU': {'mi': [1e-8, 10.0], 'normalize_learning_rate': [0, 1], 'damping': [0.0, 100.0]},
                # 'LNU with weights predicts': {'mi': [1e-8, 10.0], 'normalize_learning_rate': [0, 1], 'damping': [0.0, 100.0]},
                # 'Conjugate gradient': {'epochs': self.limits_constants["epochs"]},
                ### 'Tensorflow LSTM': {'loses':["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "logcosh",
                ### "kullback_leibler_divergence"], 'activations':['softmax', 'elu', 'selu', 'softplus', 'tanh', 'sigmoid', 'exponential', 'linear']},
                ### 'Tensorflow MLP': {'loses':["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "logcosh",
                ### "kullback_leibler_divergence"], 'activations':['softmax', 'elu', 'selu', 'softplus', 'tanh', 'sigmoid', 'exponential', 'linear']},
                # 'Sklearn regression': {'model': self.limits_constants["models"]},  # 'alpha': self.limits_constants["alpha"], 'n_iter': [100, 500], 'epsilon': [1.01, 5.0], 'alphas': [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]], 'gcv_mode': ['auto', 'svd', 'eigen'], 'solver': ['auto', 'svd', 'eigen']},
                "Extra trees": {"n_estimators": [1.0, 500.0]},
                "Bayes ridge regression": {
                    "alpha_1": [1e-5, 1e-7],
                    "alpha_2": [1e-5, 1e-7],
                    "lambda_1": [1e-5, 1e-7],
                    "lambda_2": [1e-5, 1e-7],
                },
                "Hubber regression": {
                    "epsilon": [1.01, 5.0],
                    "alpha": self.limits_constants["alpha"],
                },
                # 'Extreme learning machine': {'n_hidden': [2, 300], 'alpha': self.limits_constants["alpha"], 'rbf_width': [0.0, 10.0], 'activation_func': ['tanh', 'sine', 'tribas', 'inv_tribase', 'sigmoid', 'hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric']},
                "Gen Extreme learning machine": {
                    "alpha": self.limits_constants["alpha"]
                },
            }

    class Models(ConfigBase):
        """Configuration for models like what models should be used or what data input model has.

        Note:
            Most of variables are very long dicts by default and usually should not be edited. Only
            interesting variable for common users that should be changed is `used_columns`"""

        @MyProperty(bool)
        def already_trained() -> bool:
            """
            Type:
                bool

            Default:
                False

            Computationaly hard models (LSTM) load from disk."""

            return False

        @MyProperty((list, set))
        def used_models() -> Union[List, Set]:
            """
            Types:
                list | tuple

            Options::

                [
                    ### predictit.models.statsmodels_autoregressive
                    "AR", "ARIMA", "autoreg", 'ARMA', 'SARIMAX',

                    ### predictit.models.autoreg_LNU
                    "LNU", 'LNU with weights predicts', 'LNU normalized',

                    ### predictit.models.regression
                    "Regression", "Ridge regression",

                    ### predictit.models.levenberg_marquardt
                    "Levenberg-Marquardt",

                    ### predictit.models.conjugate_gradient
                    "Conjugate gradient",

                    # predictit.models.tensorflow
                    "Tensorflow LSTM", "Tensorflow MLP",

                    ### predictit.models.sklearn_regression
                    "Sklearn regression", "Bayes ridge regression", "KNeighbors regression", "Decision tree regression", "Hubber regression",
                    "Bagging regression", "Stochastic gradient regression", "Extreme learning machine", "Gen Extreme learning machine",
                    "Extra trees regression", "Random forest regression", "Passive aggressive regression", "Gradient boosting",
                    "Sklearn regression one column one step", "Bayes ridge regression one column one step", "Decision tree regression one column one step",
                    "Hubber regression one column one step",

                    # predictit.models.average
                    "Average short",
                    "Average long",
                ]

            Default::

                [
                    ### predictit.models.statsmodels_autoregressive
                    "AR",
                    "ARIMA",
                    "autoreg",

                    ### predictit.models.autoreg_LNU
                    "LNU",

                    ### predictit.models.regression
                    "Ridge regression",

                    ### predictit.models.levenberg_marquardt
                    "Levenberg-Marquardt",

                    ### predictit.models.conjugate_gradient
                    "Conjugate gradient",

                    ### predictit.models.sklearn_regression
                    "Sklearn regression",
                    "Bayes ridge regression",
                    "KNeighbors regression",
                    "Decision tree regression",
                    "Hubber regression",
                    "Sklearn regression one column one step",
                    "Bayes ridge regression one column one step",

                    ### predictit.models.average
                    "Average short",
                    "Average long",
                ]

            Models, that will be used in prediction.

            Verbose documentation is in models submodule `__init__.py` and it's files.
            """

            return [
                ### predictit.models.statsmodels_autoregressive
                "AR",
                "ARIMA",
                "autoreg",
                # 'ARMA', 'SARIMAX',
                ### predictit.models.autoreg_LNU
                "LNU",
                # 'LNU with weights predicts',
                # 'LNU normalized',
                ### predictit.models.regression
                # "Regression",
                "Ridge regression",
                ### predictit.models.levenberg_marquardt
                "Levenberg-Marquardt",
                ### predictit.models.conjugate_gradient
                "Conjugate gradient",
                # predictit.models.tensorflow
                # "Tensorflow LSTM",
                # "Tensorflow MLP",
                ## predictit.models.sklearn_regression
                "Sklearn regression",
                "Bayes ridge regression",
                "KNeighbors regression",
                "Decision tree regression",
                "Hubber regression",
                # 'Bagging regression', 'Stochastic gradient regression', 'Extreme learning machine', 'Gen Extreme learning machine',  'Extra trees regression', 'Random forest regression',
                # , 'Passive aggressive regression', 'Gradient boosting',
                "Sklearn regression one column one step",
                "Bayes ridge regression one column one step",
                #  'Decision tree regression one column one step', 'Hubber regression one column one step',
                # predictit.models.average
                "Average short",
                "Average long",
            ]

        @MyProperty(dict)
        def data_inputs(self) -> Dict:
            """
            Type:
                dict

            Example::

                {
                    "one_step_constant": {
                        "n_steps_in": self.default_n_steps_in,
                        "n_steps_out": 1,
                        "default_other_columns_length": self.default_other_columns_length,
                        "constant": 1,
                    }
                }

            For default value, go to source and check return.

            Some models needs input to output mapping as an input. Usually running window is used.

            You can create new data inputs with some name and then assign it to models you want with this name
            in models_input.

            Input types are dynamically defined based on 'default_n_steps_in' value. Change this value and
            models_input where you define which models what inputs use. It use mydatapreprocessing
            `create_inputs` function. Check it for more info if you want to know how it works or check visual
            test. interested how the inputs are created.
            """

            return {
                "data": None,
                "data_one_column": None,
                "one_step_constant": {
                    "n_steps_in": self.default_n_steps_in,
                    "n_steps_out": 1,
                    "default_other_columns_length": self.default_other_columns_length,
                    "constant": 1,
                },
                "one_step": {
                    "n_steps_in": self.default_n_steps_in,
                    "n_steps_out": 1,
                    "default_other_columns_length": self.default_other_columns_length,
                    "constant": 0,
                },
                "multi_step_constant": {
                    "n_steps_in": self.default_n_steps_in,
                    "n_steps_out": self.predicts,
                    "default_other_columns_length": self.default_other_columns_length,
                    "constant": 1,
                },
                "multi_step": {
                    "n_steps_in": self.default_n_steps_in,
                    "n_steps_out": self.predicts,
                    "default_other_columns_length": self.default_other_columns_length,
                    "constant": 0,
                },
                "one_in_one_out_constant": {
                    "n_steps_in": self.default_n_steps_in,
                    "n_steps_out": 1,
                    "constant": 1,
                },
                "one_in_one_out": {
                    "n_steps_in": self.default_n_steps_in,
                    "n_steps_out": 1,
                    "constant": 0,
                },
                "one_in_multi_step_out": {
                    "n_steps_in": self.default_n_steps_in,
                    "n_steps_out": self.predicts,
                    "default_other_columns_length": self.default_other_columns_length,
                    "constant": 0,
                },
                "not_serialized": {
                    "n_steps_in": self.default_n_steps_in,
                    "n_steps_out": self.predicts,
                    "constant": 0,
                    "serialize_columns": 0,
                },
            }

        @MyProperty(dict)
        def models_input(self) -> Dict:
            """
            Type:
                dict

            Example::

                {
                    "Stochastic gradient regression": "one_in_multi_step_out",
                    "Tensorflow LSTM": "not_serialized",
                }

            There are more types of data input. Define what models use what data. For default value, go to source and check return.

            Note:
                There are not only data inputs from data_inputs but also `data` and `data_one_column` which
                are not numpy.ndarray of inputs and outputs but original not transformed data."""

            return {
                **{
                    model_name: "data_one_column"
                    for model_name in [
                        "AR",
                        "ARMA",
                        "ARIMA",
                        "autoreg",
                        "SARIMAX",
                        "Average short",
                        "Average long",
                    ]
                },
                **{
                    model_name: "one_in_one_out_constant"
                    for model_name in [
                        "LNU",
                        "LNU normalized",
                        "LNU with weights predicts",
                        "Conjugate gradient",
                        "Regression",
                        "Levenberg-Marquardt",
                        "Ridge regression",
                    ]
                },
                **{
                    model_name: "one_in_one_out"
                    for model_name in [
                        "Sklearn regression one column one step",
                        "Bayes ridge regression one column one step",
                        "Decision tree regression one column one step",
                        "Hubber regression one column one step",
                    ]
                },
                # This is only for one step prediction on multivariate data
                # **{
                #     model_name: "one_step_constant"
                #     for model_name in []
                # },
                **{
                    model_name: "multi_step"
                    for model_name in [
                        "Sklearn regression",
                        "Bayes ridge regression",
                        "Hubber regression",
                        "Extra trees regression",
                        "Decision tree regression",
                        "KNeighbors regression",
                        "Random forest regression",
                        "Bagging regression",
                        "Passive aggressive regression",
                        "Extreme learning machine",
                        "Gen Extreme learning machine",
                        "Gradient boosting",
                        "Tensorflow MLP",
                    ]
                },
                "Stochastic gradient regression": "one_in_multi_step_out",
                "Tensorflow LSTM": "not_serialized",
            }

        @MyProperty(dict)
        def models_parameters(self) -> Dict:
            """
            Type:
                dict

            Example::

                {
                    "AR": {
                        "used_model": "ar",
                        "method": "cmle",
                        "trend": "nc",
                        "solver": "lbfgs",
                    },
                }

            If using presets - overwriten. If no value for model, then default values are used
            For default value, go to source and check return."""

            models_parameters = {
                "Average short": {
                    "length": 7,
                },
                "Average length": {
                    "length": 100,
                },
                "AR": {
                    "used_model": "ar",
                    "method": "cmle",
                    "trend": "nc",
                    "solver": "lbfgs",
                },
                "ARMA": {"used_model": "arima", "p": 4, "d": 0, "q": 1},
                "ARIMA": {
                    "used_model": "arima",
                    "p": 6,
                    "d": 1,
                    "q": 1,
                },
                "autoreg": {
                    "used_model": "autoreg",
                    "maxlag": 13,
                    "cov_type": "nonrobust",
                },
                "SARIMAX": {
                    "used_model": "sarimax",
                    "p": 3,
                    "d": 0,
                    "q": 0,
                    "seasonal": (1, 0, 0, 4),
                    "method": "lbfgs",
                    "trend": "nc",
                },
                "LNU": {
                    "learning_rate": "infer",
                    "epochs": 50,
                    "predict_w": False,
                    "early_stopping": True,
                    "learning_rate_decay": 0.9,
                    "normalize_learning_rate": 0,
                },
                "LNU normalized": {
                    "learning_rate": "infer",
                    "epochs": 50,
                    "predict_w": False,
                    "early_stopping": True,
                    "learning_rate_decay": 0.9,
                    "normalize_learning_rate": 1,
                },
                "LNU with weights predicts": {
                    "learning_rate": "infer",
                    "epochs": 10,
                    "predict_w": True,
                    "early_stopping": True,
                    "learning_rate_decay": 0.9,
                    "predicted_w_number": self.predicts,
                    "normalize_learning_rate": 0,
                },
                "Conjugate gradient": {"epochs": 100},
                "Regression": {"model": "linear"},
                "Ridge regression": {"model": "ridge", "lmbda": 0.1},
                "Levenberg-Marquardt": {"learning_rate": 0.1, "epochs": 50},
                # Tensorflow
                "Tensorflow LSTM": {
                    "layers": "lstm",
                    "epochs": 100,
                    "load_trained_model": 0,
                    "update_trained_model": 0,
                    "save_model": 1,
                    "saved_model_path_string": "stored_models",
                    "optimizer": "adam",
                    "loss": "mse",
                    "summary": False,
                    "used_metrics": "accuracy",
                    "timedistributed": 0,
                },
                "Tensorflow MLP": {
                    "layers": "mlp",
                    "epochs": 100,
                    "load_trained_model": 0,
                    "update_trained_model": 0,
                    "save_model": 1,
                    "saved_model_path_string": "stored_models",
                    "optimizer": "adam",
                    "loss": "mse",
                    "summary": False,
                    "used_metrics": "accuracy",
                    "timedistributed": 0,
                },
                # Sklearn
                "Sklearn regression": {
                    "model": "BayesianRidge",
                    "alpha": 0.0001,
                    "n_iter": 100,
                    "epsilon": 1.35,
                    "alphas": [0.1, 0.5, 1],
                    "gcv_mode": "auto",
                    "solver": "auto",
                    "alpha_1": 1.0e-6,
                    "alpha_2": 1.0e-6,
                    "lambda_1": 1.0e-6,
                    "lambda_2": 1.0e-6,
                    "n_hidden": 20,
                    "rbf_width": 0,
                    "activation_func": "selu",
                },
                "Bayes ridge regression": {
                    "model": "BayesianRidge",
                    "n_iter": 300,
                    "alpha_1": 1.0e-6,
                    "alpha_2": 1.0e-6,
                    "lambda_1": 1.0e-6,
                    "lambda_2": 1.0e-6,
                },
                "Hubber regression": {
                    "model": "HuberRegressor",
                    "epsilon": 1.35,
                    "alpha": 0.0001,
                },
                "Extra trees regression": {"model": "ExtraTreesRegressor"},
                "Decision tree regression": {"model": "DecisionTreeRegressor"},
                "KNeighbors regression": {"model": "KNeighborsRegressor"},
                "Random forest regression": {"model": "RandomForestRegressor"},
                "Bagging regression": {"model": "BaggingRegressor"},
                "Sklearn regression one column one step": {"model": "LinearRegression"},
                "Bayes ridge regression one column one step": {
                    "model": "BayesianRidge"
                },
                "Decision tree regression one column one step": {
                    "model": "DecisionTreeRegressor"
                },
                "Hubber regression one column one step": {"model": "HuberRegressor"},
                "Extreme learning machine": {
                    "model": "ELMRegressor",
                    "n_hidden": 50,
                    "alpha": 0.3,
                    "rbf_width": 0,
                    "activation_func": "tanh",
                },
            }

            return models_parameters

    class _Internal(ConfigBase):
        """Internal settings"""

        @MyProperty(bool)
        def is_tested() -> bool:
            """
            Type:
                'bool'

            Configured automatically from conftest (used in multiprocess).
            """
            return False

    ###############
    ### Presets ###
    ###############

    # Edit if you want, but it's not necessary from here - Mostly for GUI.
    ###!!! overwrite defined settings !!!###
    presets = {
        "fast": {
            "optimizeit": False,
            "default_n_steps_in": 6,
            "repeatit": 20,
            "optimization": False,
            "datalength": 400,
            "other_columns": False,
            "data_transform": None,
            "remove_outliers": False,
            "analyzeit": 0,
            "standardizeit": None,
            # If editing or adding new models, name of the models have to be the same as in models module
            "used_models": [
                "AR",
                "Conjugate gradient",
                "Sklearn regression",
                "Average short",
            ],
        },
        "normal": {
            "optimizeit": False,
            "default_n_steps_in": 12,
            "repeatit": 50,
            "optimization": False,
            "datalength": 3000,
            "other_columns": True,
            "remove_outliers": False,
            "analyzeit": 0,
            "standardizeit": "standardize",
            "used_models": [
                "AR",
                "ARIMA",
                "autoreg",
                "SARIMAX",
                "LNU",
                "Conjugate gradient",
                "Sklearn regression",
                "Bayes ridge regression",
                "Hubber regression",
                "Decision tree regression",
                "KNeighbors regression",
                "Random forest regression",
                "Bagging regression",
                "Average short",
                "Average long",
            ],
        },
    }


config = Config()
