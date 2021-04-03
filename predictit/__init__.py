# -*- coding: utf-8 -*-

"""
predictit
=========

.. image:: https://img.shields.io/pypi/pyversions/predictit.svg
    :target: https://pypi.python.org/pypi/predictit/
    :alt: Python versions

.. image:: https://badge.fury.io/py/predictit.svg
    :target: https://badge.fury.io/py/predictit
    :alt: PyPI pyversion

.. image:: https://img.shields.io/lgtm/grade/python/g/Malachov/predictit.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/Malachov/predictit/context:python
    :alt: Language grade: Python

.. image:: https://travis-ci.com/Malachov/predictit.svg?branch=master
    :target: https://travis-ci.com/Malachov/predictit
    :alt: Build Status

.. image:: https://readthedocs.org/projects/predictit/badge/?version=master
    :target: https://predictit.readthedocs.io/en/master/?badge=master
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License: MIT

.. image:: https://codecov.io/gh/Malachov/predictit/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Malachov/predictit
    :alt: codecov


Library/framework for making time series predictions. Choose the data, choose the models (ARIMA, regressions, LSTM...)
from libraries like statsmodels, scikit-learn, tensorflow. Do the setup (it's not necessary of course - you can use some preset)
and predict.

Library contain model hyperparameters optimization as well as option variable optimization.
That means, that library can find optimal preprocessing (smoothing, dropping non correlated columns,
standardization) and on top of that it can find optimal hyperparameters such as number of neuron layers.


Output
------

Most common output is plotly interactive graph, numpy array of results or detailed dataframe results.

.. image:: img/output_example.png
  :width: 620
  :alt: Output example

.. image:: img/table_of_results.png
  :width: 620
  :alt: Table of results


Return type of main predict function depends on *configuration.py*. It can return best prediction
as array or all predictions as dataframe. Interactive html plot is also created.

Oficial repo and documentation links
------------------------------------

Repo on github - https://github.com/Malachov/predictit

Readthedocs documentation - https://predictit.readthedocs.io

Installation
------------

Python >=3.6. Python 2 is not supported. Install just with::

    $ pip install predictit

Sometime you can have issues with installing some libraries from requirements
(e.g. numpy because not BLAS / LAPACK). There are also two libraries - Tensorflow
and pyodbc not in requirements, because not necessary, but troublesome. If library
not installed with pip, check which library don't work, install manually with stackoverflow and repeat...

Versions troubleshooting => Software is build in way, that it should be the best using latest versions of dependencies. In most cases older versions works well as well. Only exception can be author's library mydatapreprocessing, which is new and under development (API is not stable) and some version of predictit has dependency on particular version of mydatapreprocessing. Clean install of latest versions fix issues.

Library was developed during 2020 and structure and even API (configuration) changed a lot. From version 1.60 it's considered to be stable and code made for library will work till 2.0.0.

Examples
--------

    Software can be used in three ways. As a python library or with command line arguments
    or as normal python scripts.
    Main function is predict in *main.py* script.
    There is also *predict_multiple_columns* function if you want to predict more at once
    (columns or time frequentions) and also *compare_models* function that tell you which models are best.
    It evaluate error criterion on out of sample test data instead of predict (use as much as possible)
    so more reliable errors (for example decision trees just assign input from learning set, so error
    in predict is 0, in compare_models it's accurate). Then you can use only good models in predict function.

    Simple example of using predictit as a python library and function arguments
    ----------------------------------------------------------------------------

    >>> import predictit
    >>> import numpy as np
    ...
    >>> predictions_1 = predictit.main.predict(data=np.random.randn(100, 2), predicted_column=1, predicts=3, return_type='best')

    There are only two positional arguments (because, there is more than hundred configurable values).
    data and predicted_column, so you can use also

    >>> mydata = pd.DataFrame(np.random.randn(100, 2), columns=['a', 'b'])
    >>> predictions_1_positional = predictit.main.predict(mydata, 'b')

    Simple example of using as a python library and editing Config
    --------------------------------------------------------------

    >>> import predictit
    >>> from predictit.configuration import Config
    ...
    >>> # You can edit Config in two ways
    >>> Config.data = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'  # You can use local path on pc as well... "/home/name/mycsv.csv" !
    >>> Config.predicted_column = 'Temp'  # You can use index as well
    >>> Config.datetime_column = 'Date'  # Will be used for resampling and result plot description
    >>> Config.freq = "D"  # One day - one value resampling
    >>> Config.resample_function = "mean"  # If more values in one day - use mean (more sources)
    >>> Config.return_type = 'detailed_dictionary'
    >>> Config.debug = 0  # Ignore warnings
    ...
    >>> # Or
    >>> Config.update({
    ...     'datalength': 300,  # Used datalength
    ...     'predicts': 14,  # Number of predicted values
    ...     'default_n_steps_in': 12  # Value of recursive inputs in model (do not use too high - slower and worse predictions)
    ... })
    ...
    >>> predictions_2 = predictit.main.predict()


    Simple example of using *main.py* as a script
    ---------------------------------------------

    Open *configuration.py* (only script you need to edit (very simple)), do the setup. Mainly used_function and data or data_source and path. Then just run *main.py*.

    Simple example of using command line arguments
    ----------------------------------------------

    Run code below in terminal in predictit folder.
    Use *python main.py --help* for more parameters info.

    >>> python main.py --used_function predict --data 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv' --predicted_column "'Temp'"


    Explore Config
    --------------

    Type *Config.*, then, if not autamatically, use ctrl + spacebar to see all posible values. To see what option means, type for example *Config.return_type*, than do mouseover with pressed ctrl. It will reveal comment that describe the option (at least at VS Code)

    To see all the possible values in *configuration.py*, use

    >>> predictit.configuration.print_config()

    Example of compare_models function
    ----------------------------------

    >>> import predictit
    >>> from predictit.configuration import Config
    ...
    >>> my_data_array = np.random.randn(200, 2)  # Define your data here
    ...
    >>> # You can compare it on same data in various parts or on different data (check configuration on how to insert dictionary with data names)
    >>> Config.update({
    ...     'data_all': {'First part': (my_data_array[:100], 0), 'Second part': (my_data_array[100:], 1)}
    >>> })
    ...
    >>> compared_models = predictit.main.compare_models()

    Example of predict_multiple function
    ------------------------------------

    >>> import predictit
    >>> from predictit.configuration import Config
    ...
    >>> Config.data = np.random.randn(120, 3)
    >>> Config.predicted_columns = ['*']  # Define list of columns or '*' for predicting all of the numeric columns
    >>> Config.used_models = ['Conjugate gradient', 'Decision tree regression']  # Use just few models to be faster
    ...
    >>> multiple_columns_prediction = predictit.main.predict_multiple_columns()


    Example of Config variable optimization
    ---------------------------------------

    >>> Config.update({
    ...     'data': "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    ...     'predicted_column': 'Temp',
    ...     'return_type': 'all_dataframe',
    ...     'datalength': 120,
    ...     'optimization': 1,
    ...     'optimization_variable': 'default_n_steps_in',
    ...     'optimization_values': [4, 6, 8],
    ...     'plot_all_optimized_models': 0,
    ...     'print_table': 2,  # Print detailed table
    ...     'used_models': ['AR (Autoregression)', 'Sklearn regression']
    ... })
    ...
    >>> predictions_optimized_config = predictit.main.predict()


    Hyperparameters tuning
    ----------------------

    To optmize hyperparameters, just set *optimizeit: 1,* and model parameters limits. It is commented in *Config.py* how to use it. It's not grid bruteforce. Heuristic method based on halving interval is used, but still it can be time consuming. It is recomend only to tune parameters worth of it. Or tune it by parts.

    GUI
    ---

    It is possible to use basic GUI. But only with CSV data source.
    Just run *gui_start.py* if you have downloaded software or call *predictit.gui_start.run_gui()* if you are importing via PyPI.

    Screenshot of such a GUI

    <p align="center">
    <img src="./img/GUI.png" width="620" alt="Table of results"/>
    </p>

    .. image:: img/GUI.png
    :width: 620
    :alt: GUI


    Better GUI with fully customizable settings will be shipped next year hopefully.

    Feature derivation
    ------------------

    It is possible to add new data that is derived from original. It can be running fourier transform maximum or two columns multiplication or rolling standard deviation.

    Categorical embedings
    ---------------------

    It is also possible to use string values in predictions. You can choose Config values 'embedding' 'label' and every unique string will be assigned unique number, 'one-hot' create new column for every unique string (can be time consuming).

    Feature extraction
    ------------------

    Under development right now :[

    Data preprocessing, plotting and other Functions
    ------------------------------------------------

    You can use any library functions separately for your needs of course. mydatapreprocessing, mylogging and myplottling are my other projects, which are used heavily. Example is here

    >>> from mydatapreprocessing import load_data, data_consolidation, preprocess_data
    >>> from myplotting import plot
    >>> from predictit.analyze import analyze_column
    ...
    >>> data = "https://blockchain.info/unconfirmed-transactions?format=json"
    ...
    >>> # Load data from file or URL
    >>> data_loaded = load_data(data, request_datatype_suffix=".json", predicted_table='txs', data_orientation="index")
    ...
    >>> # Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
    >>> # only numeric data and resample ifg configured. It return array, dataframe
    >>> data_consolidated = data_consolidation(
    ...     data_loaded, predicted_column="weight", remove_nans_threshold=0.9, remove_nans_or_replace='interpolate')
    ...
    >>> # Predicted column is on index 0 after consolidation)
    >>> analyze_column(data_consolidated.iloc[:, 0])
    ...
    >>> # Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
    >>> # transformation, so unpack it with _
    >>> data_preprocessed, _, _ = preprocess_data(data_consolidated, remove_outliers=True, smoothit=False,
    ...                                         correlation_threshold=False, data_transform=False, standardizeit='standardize')
    ...
    >>> # Plot inserted data
    >>> plot(data_preprocessed)


    Using just one model apart main function
    ----------------------------------------

    Main benefit is performance boost. You can have code inder the controll much simpler (much less code), but no features from configuration available.

    >>> import mydatapreprocessing as mdp
    ...
    >>> data = mdp.generatedata.gen_sin(1000)
    >>> test = data[-7:]
    >>> data = data[: -7]
    >>> data = mdp.preprocessing.data_consolidation(data)
    >>> (X, y), x_input, _ = mdp.inputs.create_inputs(data.values, 'batch', input_type_params={'n_steps_in': 6})  # First tuple, because some models use raw data - one argument, e.g. [1, 2, 3...]
    ...
    >>> trained_model = predictit.models.sklearn_regression.train((X, y), regressor='bayesianridge')
    >>> predictions_one_model = predictit.models.sklearn_regression.predict(x_input, trained_model, predicts=7)
    ...
    >>> predictions_one_model_error = predictit.evaluate_predictions.compare_predicted_to_test(predictions_one_model, test, error_criterion='mape')  # , plot=1


    Example of using library as a pro with deeper editting Config
    -------------------------------------------------------------

    >>> import predictit
    >>> from predictit.configuration import Config
    ...
    >>> Config.update({
    ...     'data': r'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv',  # Full CSV path with suffix
    ...     'predicted_column': 'Temp',  # Column name that we want to predict
    ...     'datalength': 200,
    ...     'predicts': 7,  # Number of predicted values - 7 by default
    ...     'print_number_of_models': 6,  # Visualize 6 best models
    ...     'repeatit': 50,  # Repeat calculation times on shifted data to evaluate error criterion
    ...     'other_columns': 0,  # Whether use other columns or not
    ...     'debug': 1,  # Whether print details and warnings
    ...
    ...     # Chose models that will be computed - remove if you want to use all the models
    ...     'used_models': [
    ...         "AR (Autoregression)",
    ...         "ARIMA (Autoregression integrated moving average)",
    ...         "Autoregressive Linear neural unit",
    ...         "Conjugate gradient",
    ...         "Sklearn regression",
    ...         "Bayes ridge regression one column one step",
    ...         "Decision tree regression",
    ...     ],
    ...
    ...     # Define parameters of models
    ...
    ...     'models_parameters': {
    ...
    ...         "AR (Autoregression)": {'used_model': 'ar', 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
    ...         "ARIMA (Autoregression integrated moving average)": {'used_model': 'arima', 'p': 6, 'd': 0, 'q': 0},
    ...
    ...         "Autoregressive Linear neural unit": {'mi_multiple': 1, 'mi_linspace': (1e-5, 1e-4, 3), 'epochs': 10, 'w_predict': 0, 'minormit': 0},
    ...         "Conjugate gradient": {'epochs': 80},
    ...
    ...         "Bayes ridge regression": {'regressor': 'bayesianridge', 'n_iter': 300, 'alpha_1': 1.e-6, 'alpha_2': 1.e-6, 'lambda_1': 1.e-6, 'lambda_2': 1.e-6},
    ...     }
    ... })
    ...
    >>> predictions_configured = predictit.main.predict()

Performance - How to scale
--------------------------

Time series prediction is very different than image recognition and more data doesn't necesarilly means better prediction. If you're issuing performance problems, try fast preset (turn off optimizations, make less recurent values, choose only few models, threshold datalength etc.) you can edit preset if you need. If you still have performance troubles and you have too much data, use resampling and select only valuable columns - for example correlation_threshold and do not derive extra columns. If you are interested mostly in predictions and not in the plot, turn the plot of.

Future work
-----------

It's planned to do real GUI and possibility to serve web app as well as desktop. Scalability can be solved two ways. First is incremental learning (not every model supports today). Second is virtualisation (processes running in cluster separately).

There is very big todo list on root called *TODO.md.*

For developers
--------------

Any help from other developers very appreciated... :D
Dont be shy to create Issue or text on <malachovd@seznam.cz>

"""

__version__ = "1.61.3"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = [
    "analyze",
    "best_params",
    "configuration",
    "database",
    "evaluate_predictions",
    "main",
    "misc",
    "models",
    "plots",
    "test_data",
    "main_loop",
    "gui_start",
]


from . import configuration
import mylogging
import warnings


# Define whether to print warnings or not or stop on warnings as on error (mute warnings from imports)
with warnings.catch_warnings():
    mylogging.set_warnings(
        configuration.Config.debug,
        configuration.Config.ignored_warnings,
        configuration.Config.ignored_warnings_class_type,
    )

    from . import analyze
    from . import best_params
    from . import models
    from . import evaluate_predictions
    from . import misc
    from . import plots
    from . import test_data
    from . import main_loop
    from . import gui_start

    from . import main
