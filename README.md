# predictit
Library/framework for making predictions. Choose best of 20 models (ARIMA, regressions, LSTM...) from libraries like statsmodels, sci-kit, tensorflow and some own models. Library also automatically preprocess data and chose optimal parameters of predictions.

## Output
Most common output is plotly interactive graph, numpy array of results or deploying to database.

<p align="center">
<img src="https://raw.githubusercontent.com/Malachov/predictit/master/output_example.png" width="620" alt="Plot of results"/>
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/Malachov/predictit/master/table_of_results.png" width="620" alt="Table of results"/>
</p>

Return type of main predict function depends on config. It can return best prediction as array or all predictions array or plot as div string or dictionary or detailed results.

## Oficial repo and documentation links

[Repo on github](https://github.com/Malachov/predictit)

[Official readthedocs documentation](https://predictit.readthedocs.io)

## Installation
    pip install predictit

Sometime you can have issues with installing some libraries from requirements (e.g. numpy because not BLAS / LAPACK). There are also two libraries - Tensorflow and pyodbc not in requirements, because not necessary, but troublesome. If library not installed with pip, check which library don't work, install manually with stackoverflow and repeat...

## How to
Software can be used in three ways. As a python library or with command line arguments or as normal python scripts.
Main function is predict in `main.py` script.
There is also predict_multiple_columns if you want to predict more at once (columns or time frequentions) and also compare_models function that evaluate test data and can tell you which models are best. Then you can use only such a models. It's recommended also to use arguments optimization just once, change initial parameters in config and turn optimization off for performance reasons.

Command line arguments as well as functions arguments overwrite default `config.py` values. Not all the config options are in function arguments or command line arguments.

### Simple example of predict function with Pypi and function arguments
```Python
import predictit
import numpy as np

predictions = predictit.main.predict(np.random.randn(1, 100), predicts=3, plot=1)
```

### Simple example of using `main.py` script
Open `config.py` (only script you need to edit (very simple)), do the setup. Mainly used_function and data or data_source and path. Then just run `main.py`.

### Simple example of using command line arguments
Run code below in terminal in predictit folder and change csv path (test data are not included in library because of size!). Use `main.py` --help for more parameters info.

```
python main.py --used_function predict --data_source 'csv' --csv_full_path 'test_data/daily-minimum-temperatures.csv' --predicted_column 1
```

### Example of using as a library as a pro with editting `config.py`
```Python

import predictit

config = predictit.config.config

config.update({
    'data_source': 'csv',  # Data source. ('csv' or 'sql' or 'test')
    'csv_full_path': r'C:\Users\truton\ownCloud\Github\predictit_library\predictit\test_data\5000 Sales Records.csv',  # Full CSV path with suffix
    'predicted_column': 'Temp',  # Column name that we want to predict

    'predicts': 7,  # Number of predicted values - 7 by default
    'datalength': 1000,  # The length of the data used for prediction
    'print_number_of_models': 6,  # Visualize 6 best models
    'repeatit': 50,  # Repeat calculation times on shifted data to evaluate error criterion
    'other_columns': 0,  # Whether use other columns or not
    'debug': 1,  # Whether print details and warnings

    # Chose models that will be computed - remove if you want to use all the models
    'used_models': {
        "AR (Autoregression)": predictit.models.statsmodels_autoregressive,

        "ARIMA (Autoregression integrated moving average)": predictit.models.statsmodels_autoregressive,

        "Autoregressive Linear neural unit": predictit.models.autoreg_LNU,
        "Conjugate gradient": predictit.models.conjugate_gradient,

        "Sklearn regression": predictit.models.sklearn_regression,
    },

    # Define parameters of models

    'models_parameters': {

        'AR (Autoregression)': {'model': 'ar', 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        'ARIMA (Autoregression integrated moving average)': {'model': 'arima', 'p': 3, 'd': 0, 'q': 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm'},

        'Autoregressive Linear neural unit': {'mi_multiple': 1, 'mi_linspace': (1e-7, 1e-2, 200), 'epochs': 100, 'w_predict': 0, 'minormit': 0},
        'Conjugate gradient': {'epochs': 200},

        'Bayes ridge regression': {'regressor': 'bayesianridge', 'n_iter': 300, 'alpha_1': 1.e-6, 'alpha_2': 1.e-6, 'lambda_1': 1.e-6, 'lambda_2': 1.e-6},
        'Sklearn regression': {'regressor': 'linear', 'alpha': 0.0001, 'n_iter': 100, 'epsilon': 1.35, 'alphas': [0.1, 0.5, 1], 'gcv_mode': 'auto', 'solver': 'auto', 'alpha_1': 1.e-6, 'alpha_2': 1.e-6, 'lambda_1': 1.e-6, 'lambda_2': 1.e-6, 'n_hidden': 20, 'rbf_width': 0, 'activation_func': 'selu'},
    }

})

predictions = predictit.main.predict()
```


Or if you downloaded it from github and not via pypi, just edit config as you need and run `main.py`

To see all the possible values in `config.py` from your IDE, use

```Python
predictit.config.print_config()
```
