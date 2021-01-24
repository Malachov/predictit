# -*- coding: utf-8 -*-

"""
Library/framework for making predictions. Choose best of 20 models (ARIMA, regressions, LSTM...)
from libraries like statsmodels, scikit-learn, tensorflow and some own models. There are hundreds of
customizable options (it's not necessary of course) as well as some config presets.

Library contain model hyperparameters optimization as well as option variable optimization.
That means, that library can find optimal preprocessing (smoothing, dropping non correlated columns,
standardization) and on top of that it can find optimal models inner parameters such as number of neuron layers.

## Output

Most common output is plotly interactive graph, numpy array of results or deploying to database.

<p align="center">
<img src="https://raw.githubusercontent.com/Malachov/predictit/master/output_example.png" width="620" alt="Plot of results"/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/Malachov/predictit/master/table_of_results.png" width="620" alt="Table of results"/>
</p>

Return type of main predict function depends on `configation.py`. It can return best prediction as array
or all predictions as dataframe. Interactive html plot is also created.

## Oficial repo and documentation links

[Repo on github](https://github.com/Malachov/predictit)

[Official readthedocs documentation](https://predictit.readthedocs.io)

## Installation

Python >=3.6. Python 2 is not supported. Install just with

    pip install predictit

Sometime you can have issues with installing some libraries from requirements
(e.g. numpy because not BLAS / LAPACK). There are also two libraries - Tensorflow
and pyodbc not in requirements, because not necessary, but troublesome. If library
not installed with pip, check which library don't work, install manually with stackoverflow and repeat...

## How to

Software can be used in three ways. As a python library or with command line arguments
or as normal python scripts.
Main function is predict in `main.py` script.
There is also `predict_multiple_columns` function if you want to predict more at once
(columns or time frequentions) and also `compare_models` function that tell you which models are best.
It evaluate error criterion on out of sample test data instead of predict (use as much as possible)
so more reliable errors (for example decision trees just assign input from learning set, so error
in predict is 0, in compare_models it's accurate). Then you can use only good models in predict function.

### Simple example of using predictit as a python library and function arguments

```Python
import predictit
import numpy as np

predictions_1 = predictit.main.predict(data=np.random.randn(100, 2), predicted_column=1, predicts=3, return_type='best')
```

### Simple example of using as a python library and editing Config

```Python
import predictit
from predictit.configuration import Config

# You can edit Config in two ways
Config.data = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'  # You can use local path on pc as well... "/home/name/mycsv.csv" !
Config.predicted_column = 'Temp'  # You can use index as well
Config.datetime_column = 'Date'  # Will be used in result
Config.freq = "D"  # One day - one value
Config.resample_function = "mean"  # If more values in one day - use mean (more sources)
Config.return_type = 'detailed_dictionary'
Config.debug = 0  # Ignore warnings

# Or
Config.update({
    'predicts': 14,  # Number of predicted values
    'default_n_steps_in': 12  # Value of recursive inputs in model (do not use too high - slower and worse predictions)
})

predictions_2 = predictit.main.predict()
```

### Simple example of using `main.py` as a script

Open `configuration.py` (only script you need to edit (very simple)), do the setup. Mainly used_function and data or data_source and path. Then just run `main.py`.

### Simple example of using command line arguments

Run code below in terminal in predictit folder.
Use `python main.py --help` for more parameters info.

```
python main.py --used_function predict --data 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv' --predicted_column "'Temp'"

```

### Explore Config

Type `Config.`, then, if not autamatically, use ctrl + spacebar to see all posible values. To see what option means, type for example `Config.return_type`, than do mouseover with pressed ctrl. It will reveal comment that describe the option (at least at VS Code)

To see all the possible values in `configuration.py`, use

```Python
predictit.configuration.print_config()
```

### Example of compare_models function

```Python
import predictit
from predictit.configuration import Config

my_data_array = np.random.randn(2000, 4)  # Define your data here

# You can compare it on same data in various parts or on different data (check configuration on how to insert dictionary with data names)
Config.update({
    'data_all': (my_data_array[-2000:], my_data_array[-1500:], my_data_array[-1000:])
})

compared_models = predictit.main.compare_models()
```

### Example of predict_multiple function

```Python
import predictit
from predictit.configuration import Config

Config.data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")

# Define list of columns or '*' for predicting all of the columns
Config.predicted_columns = ['*']

multiple_columns_prediction = predictit.main.predict_multiple_columns()

```

### Example of Config variable optimization

```Python

Config.update({
    'data': "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    'predicted_column': 'Temp',
    'return_type': 'all_dataframe',
    'optimization': 1,
    'optimization_variable': 'default_n_steps_in',
    'optimization_values': [4, 8, 10],
    'plot_all_optimized_models': 1,
    'print_table': 2  # Print detailed table
})

predictions_optimized_config = predictit.main.predict()

```

### Hyperparameters tuning

To optmize hyperparameters, just set `optimizeit: 1,` and model parameters limits. It is commented in `Config.py` how to use it. It's not grid bruteforce. Heuristic method based on halving interval is used, but still it can be time consuming. It is recomend only to tune parameters worth of it. Or tune it by parts.

## GUI

It is possible to use basic GUI. But only with CSV data source.
Just run `gui_start.py` if you have downloaded software or call `predictit.gui_start.run_gui()` if you are importing via PyPI.

## Data preprocessing, plotting and other Functions

You can use any library function separately for your needs of course. Example is here

```Python

from predictit.analyze import analyze_column
from predictit.data_preprocessing import load_data, data_consolidation, preprocess_data
from predictit.plots import plot

data = "https://blockchain.info/unconfirmed-transactions?format=json"

# Load data from file or URL
data_loaded = load_data(data, request_datatype_suffix=".json", predicted_table='txs')

# Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
# only numeric data and resample ifg configured. It return array, dataframe
data_consolidated = data_consolidation(
    data_loaded, predicted_column="weight", data_orientation="index", remove_nans_threshold=0.9, remove_nans_or_replace='interpolate')

# Predicted column is on index 0 after consolidation)
analyze_column(data_consolidated.iloc[:, 0])

# Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
# transformation, so unpack it with _
data_preprocessed, _, _ = preprocess_data(data_consolidated, remove_outliers=True, smoothit=False,
                                        correlation_threshold=False, data_transform=False, standardizeit='standardize')

# Plot inserted data
plot(data_preprocessed)


"""

__version__ = "1.61.0"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = ['analyze', 'best_params', 'configuration', 'database',
           'evaluate_predictions', 'main', 'misc', 'models', 'plots', 'test_data',
           'main_loop', 'gui_start']


from . import configuration
import mylogging
import warnings


# Define whether to print warnings or not or stop on warnings as on error (mute warnings from imports)
with warnings.catch_warnings():
    mylogging.set_warnings(configuration.Config.debug, configuration.Config.ignored_warnings, configuration.Config.ignored_warnings_class_type)


    from . import analyze
    from . import best_params
    from . import models
    from . import database
    from . import evaluate_predictions
    from . import misc
    from . import plots
    from . import test_data
    from . import main_loop
    from . import gui_start

    from . import main
