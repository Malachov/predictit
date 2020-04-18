# predictit
Library/framework for making predictions. Choose best of 20 models (ARIMA, regressions, LSTM...) from libraries like statsmodels, sci-kit, tensorflow and some own models. Library also automatically preprocess data and chose optimal parameters of predictions.

## Output
Most common output is plotly interactive graph, numpy array of results or deploying to database.

<p align="center">
<img src="https://raw.githubusercontent.com/Malachov/predictit/master/output_example.png" width="620" alt="Table of results"/>
</p>

It will also print the table of models errors.

<p align="center">
<img src="https://raw.githubusercontent.com/Malachov/predictit/master/table_of_results.png" width="700" alt="Table of results"/>
</p>

## Oficial repo and documentation links

[Repo on github](https://github.com/Malachov/predictit)

[Official readthedocs documentation](https://predictit.readthedocs.io)

## Installation
    pip install predictit

Sometime you can have issues with installing some libraries from requirements (e.g. numpy because not BLAS / LAPACK). There are also two libraries - Tensorflow and pyodbc not in requirements, because not necessary, but troublesome. If library not installed with pip, check which library don't work, install manually with stackoverflow and repeat...

## How to
Software can be used in three ways. As a python library or with command line arguments or as normal python scripts.
Main function is predict in main.py script.
There is also predict_multiple_columns if you want to predict more at once (columns or time frequentions) and also compare_models function that evaluate test data and can tell you which models are best. Then you can use only such a models. It's recommended also to use arguments optimization just once, change initial parameters in config and turn optimization off for performance reasons.

Command line arguments as well as functions arguments overwrite default config.py values. Not all the config options are in function arguments or command line arguments.

### Simple example of predict function with Pypi and function arguments
```Python
import predictit
import numpy

predictions = predictit.main.predict(np.random.randn(1, 100), predicts=3, plot=1)
```

### Simple example of using main.py script
Open config.py (only script you need to edit (very simple)), do the setup. Mainly used_function and data or data_source and path. Then just run main.py.

### Simple example of using command line arguments
Run code below in terminal in predictit folder and change csv path (test data are not included in library because of size!). Use main.py --help for more parameters info.

```
python main.py --function predict --data_source 'csv' --csv_path 'test_data/daily-minimum-temperatures.csv' --predicted_column 1
```

### Example of using as a library as a pro with editting config.py
```Python
import predictit

predictit.config.predicts = 12  # Create 12 predictions
predictit.config.data_source = 'csv'  # Define that we load data from CSV
predictit.config.csv_full_path = r'E:\VSCODE\Diplomka\test_data\daily-minimum-temperatures.csv'  # Load CSV file with data
predictit.config.save_plot_adress = r'C:\Users\TruTonton\Documents\GitHub'  # Where to save HTML plot
predictit.config.datalength = 1000  # Consider only last 1000 data points  
predictit.config.predicted_columns_names = 'Temp'  # Column name that we want to predict
predictit.config.optimizeit = 0  # Find or not best parameters for models
predictit.config.compareit = 6  # Visualize 6 best models
predictit.config.repeatit = 4  # Repeat calculation 4x times on shifted data to reduce chance
predictit.config.other_columns_length = 0  # Whether use other columns or not

# Chose models that will be computed
used_models = {
            "AR (Autoregression)": predictit.models.ar,

            "ARIMA (Autoregression integrated moving average)": predictit.models.arima,

            "Autoregressive Linear neural unit": predictit.models.autoreg_LNU,
            "Conjugate gradient": predictit.models.cg,

            "Extreme learning machine": predictit.models.regression,

            "Sklearn regression": predictit.models.regression,

           }
           
# Define parameters of models

n_steps_in = 50  # How many lagged values in models
output_shape = 'batch'  # Whether batch or one-step models

models_parameters = {

        "AR (Autoregression)": {"plot": 0, 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        "ARIMA (Autoregression integrated moving average)": {"p": 12, "d": 0, "q": 1, "plot": 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm', 'forecast_type': 'out_of_sample'},

        "Autoregressive Linear neural unit": {"plot": 0, "lags": n_steps_in, "mi": 1, "minormit": 0, "tlumenimi": 1},
        "Conjugate gradient": {"n_steps_in": 30, "epochs": 5, "constant": 1, "other_columns_lenght": None, "constant": None},

        "Extreme learning machine": {"n_steps_in": 20, "output_shape": 'one_step', "other_columns_lenght": None, "constant": None, "n_hidden": 20, "alpha": 0.3, "rbf_width": 0, "activation_func": 'selu'},

        "Sklearn regression": {"regressor": 'linear', "n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alpha": 0.0001, "n_iter": 100, "epsilon": 1.35, "alphas": [0.1, 0.5, 1], "gcv_mode": 'auto', "solver": 'auto'}

        }

predictions = predictit.main.predict()
```

Or if you downloaded it from github and not via pypi, just edit config as you need and run main.py
