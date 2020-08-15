# predictit

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/predictit.svg)](https://pypi.python.org/pypi/predictit/) [![PyPI version](https://badge.fury.io/py/predictit.svg)](https://badge.fury.io/py/predictit) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Malachov/predictit.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Malachov/predictit/context:python) [![Build Status](https://travis-ci.com/Malachov/predictit.svg?branch=master)](https://travis-ci.com/Malachov/predictit) [![Documentation Status](https://readthedocs.org/projects/predictit/badge/?version=master)](https://predictit.readthedocs.io/en/master/?badge=master) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Library/framework for making predictions. Choose best of 20 models (ARIMA, regressions, LSTM...) from libraries like statsmodels, scikit-learn, tensorflow and some own models. There are hundreds of customizable options (it's not necessary of course) as well as some config presets.

Library contain model hyperparameters optimization as well as option variable optimization. That means, that library can find optimal preprocessing (smoothing, dropping non correlated columns, standardization) and on top of that it can find optimal models inner parameters such as number of neuron layers.

## Output

Most common output is plotly interactive graph, numpy array of results or deploying to database.

<p align="center">
<img src="https://raw.githubusercontent.com/Malachov/predictit/master/output_example.png" width="620" alt="Plot of results"/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/Malachov/predictit/master/table_of_results.png" width="620" alt="Table of results"/>
</p>

Return type of main predict function depends on `configation.py`. It can return best prediction as array or all predictions as dataframe. Interactive html plot is also created.

## Oficial repo and documentation links

[Repo on github](https://github.com/Malachov/predictit)

[Official readthedocs documentation](https://predictit.readthedocs.io)

## Installation

Python >=3.6. Python 2 is not supported. Install just with

    pip install predictit

Sometime you can have issues with installing some libraries from requirements (e.g. numpy because not BLAS / LAPACK). There are also two libraries - Tensorflow and pyodbc not in requirements, because not necessary, but troublesome. If library not installed with pip, check which library don't work, install manually with stackoverflow and repeat...

## How to

Software can be used in three ways. As a python library or with command line arguments or as normal python scripts.
Main function is predict in `main.py` script.
There is also predict_multiple_columns if you want to predict more at once (columns or time frequentions) and also compare_models function that evaluate defined test data and can tell you which models are best. Then you can use only such a models.

### Simple example of using predictit as a python library and function arguments

```Python
import predictit
import numpy as np

predictions = predictit.main.predict(data=np.random.randn(1, 100), predicts=3, plotit=1)
```

### Simple example of using as a python library and editing config

```Python
import predictit
from predictit.configuration import config

# You can edit config in two ways
config.data_source = 'csv'
config.csv_full_path = 'https://datahub.io/core/global-temp/r/monthly.csv'  # You can use local path on pc as well... "/home/dan/..."
config.predicted_column = 'Mean'
config.datetime_index = 'Date'

# Or
config.update({
    'predicts': 3,
    'default_n_steps_in': 15
})

predictions = predictit.main.predict()
```

### Simple example of using `main.py` as a script

Open `configuration.py` (only script you need to edit (very simple)), do the setup. Mainly used_function and data or data_source and path. Then just run `main.py`.

### Simple example of using command line arguments

Run code below in terminal in predictit folder.
Use `python main.py --help` for more parameters info.

```
python main.py --used_function predict --data_source 'csv' --csv_full_path 'https://datahub.io/core/global-temp/r/monthly.csv' --predicted_column "'Mean'"

```

### Explore config

To see all the possible values in `configuration.py` from your IDE, use

```Python
predictit.configuration.print_config()
```

### Example of compare_models function

```Python
import predictit
from predictit.configuration import config

my_data_array = np.random.randn(2000, 4)  # Define your data here

# You can compare it on same data in various parts or on different data (check configuration on how to insert dictionary with data names)
config.update({
    'data_all': (my_data_array[-2000:], my_data_array[-1500:], my_data_array[-1000:])
})

predictit.main.compare_models()
```

### Example of predict_multiple function

```Python
import predictit
from predictit.configuration import config

config.data = pd.read_csv("https://datahub.io/core/global-temp/r/monthly.csv")

# Define list of columns or '*' for predicting all of the columns
config.predicted_columns = ['*']

predictit.main.predict_multiple_columns()
```

### Example of config variable optimization

```Python

config.update({
    'data_source': 'csv',
    'csv_full_path': "https://datahub.io/core/global-temp/r/monthly.csv",
    'predicted_column': 'Mean',
    'return_type': 'all_dataframe',
    'optimization': 1,
    'optimization_variable': 'default_n_steps_in',
    'optimization_values': [12, 20, 40],
    'plot_all_optimized_models': 1,
    'print_detailed_result': 1
})

predictions = predictit.main.predict()
```

### Hyperparameters tuning

To optmize hyperparameters, just set `optimizeit: 1,` and model parameters limits. It is commented in `config.py` how to use it. It's not grid bruteforce. Heuristic method based on halving interval is used, but still it can be time consuming. It is recomend only to tune parameters worth of it. Or tune it by parts.

## GUI

It is possible to use basic GUI. But only with CSV data source.
Just run `gui_start.py` if you have downloaded software or call `predictit.gui_start.run_gui()` if you are importing via PyPI.

## Example of using library as a pro with deeper editting config

```Python

import predictit
from predictit.configuration import config

config.update({
    'data_source': 'test',  # Data source. ('csv' or 'sql' or 'test')
    'csv_full_path': r'C:\Users\truton\ownCloud\Github\predictit_library\predictit\test_data\5000 Sales Records.csv',  # Full CSV path with suffix
    'predicted_column': '',  # Column name that we want to predict

    'predicts': 7,  # Number of predicted values - 7 by default
    'print_number_of_models': 6,  # Visualize 6 best models
    'repeatit': 50,  # Repeat calculation times on shifted data to evaluate error criterion
    'other_columns': 0,  # Whether use other columns or not
    'debug': 1,  # Whether print details and warnings

    # Chose models that will be computed - remove if you want to use all the models
    'used_models': [
        "AR (Autoregression)",
        "ARIMA (Autoregression integrated moving average)",
        "Autoregressive Linear neural unit",
        "Conjugate gradient",
        "Sklearn regression",
        'Bayes ridge regression one step',
        'Decision tree regression',
    ],

    # Define parameters of models

    'models_parameters': {

        'AR (Autoregression)': {'used_model': 'ar', 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        'ARIMA (Autoregression integrated moving average)': {'used_model': 'arima', 'p': 6, 'd': 0, 'q': 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm'},

        'Autoregressive Linear neural unit': {'mi_multiple': 1, 'mi_linspace': (1e-5, 1e-4, 3), 'epochs': 10, 'w_predict': 0, 'minormit': 0},
        'Conjugate gradient': {'epochs': 200},

        'Bayes ridge regression': {'regressor': 'bayesianridge', 'n_iter': 300, 'alpha_1': 1.e-6, 'alpha_2': 1.e-6, 'lambda_1': 1.e-6, 'lambda_2': 1.e-6},

})

predictions = predictit.main.predict()

```
