# predictit
Library/framework for making predictions. Choose best of 20 models (ARIMA, regressions, LSTM...) from libraries like statsmodels, sci-kit, tensorflow and some own models. Library also automaticcaly preprocess data and chose optimal parameters of predictions.
## Output
Most common output is plotly interactive graph, deploying to database and list of results.
Printscreen of graph
![Printscreen of output HTML graph](https://raw.githubusercontent.com/Malachov/predictit/master/output_example.png)

## How to
    pip install predictit

### Simple example with Pypi
```Python
import predictit

predictions = predictit.main.predict()  # Make prediction on test data
```

### Example with own data from CSV and config
```Python
import predictit

predictit.config.predicts = 12  # Create 12 predictions
predictit.config.data_source = 'csv'  # Define that we load data from CSV
predictit.config.csv_adress = r'E:\VSCODE\Diplomka\test_data\daily-minimum-temperatures.csv'  # Load CSV file with data
predictit.config.save_plot_adress = r'C:\Users\TruTonton\Documents\GitHub'  # Where to save HTML plot
predictit.config.datalength = 1000  # Consider only last 1000 data points  
predictit.config.predicted_columns_names = 'Temp'  # Column name that we want to predict
predictit.config.optimizeit = 0  # Find or not best parameters for models
predictit.config.compareit = 6  # Visualize 6 best models
predictit.config.repeatit = 4  # Repeat calculation 4x times on shifted data to reduce chance
predictit.config.other_columns = 0  # Whether use other columns or not

# Chose models that will be computed
used_models = {
            "AR (Autoregression)": predictit.models.ar,

            "ARIMA (Autoregression integrated moving average)": predictit.models.arima,

            "Autoregressive Linear neural unit": predictit.models.autoreg_LNU,
            "Conjugate gradient": predictit.models.cg,

            "Extreme learning machine": predictit.models.elm,

            "Sklearn universal": predictit.models.sklearn_universal,

            "Bayes Ridge Regression": predictit.models.regression_bayes_ridge,
            "Hubber regression": predictit.models.regression_hubber,
            "Lasso Regression": predictit.models.regression_lasso,
           }
           
# Define parameters of models

n_steps_in = 50  # How many lagged values in models
output_shape = 'batch'  # Whether batch or one-step models

models_parameters = {


        "AR (Autoregression)": {"plot": 0, 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        "ARMA": {"plot": 0, "p": 3, "q": 0, 'method': 'mle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs', 'forecast_type': 'in_sample'},
        "ARIMA (Autoregression integrated moving average)": {"p": 12, "d": 0, "q": 1, "plot": 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm', 'forecast_type': 'out_of_sample'},

        "Autoregressive Linear neural unit": {"plot": 0, "lags": n_steps_in, "mi": 1, "minormit": 0, "tlumenimi": 1},
        "Conjugate gradient": {"n_steps_in": 30, "epochs": 5, "constant": 1, "other_columns_lenght": None, "constant": None},

        "Extreme learning machine": {"n_steps_in": 20, "output_shape": 'one_step', "other_columns_lenght": None, "constant": None, "n_hidden": 20, "alpha": 0.3, "rbf_width": 0, "activation_func": 'selu'},

        "Sklearn universal": {"n_steps_in": n_steps_in, "output_shape": "one_step", "model": predictit.models.default_regressor, "constant": None},

        "Bayes Ridge Regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6},
        "Hubber regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "epsilon": 1.35, "alpha": 0.0001},
        "Lasso Regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alpha": 0.6}
        
        }
        
predictions = predictit.main.predict()
```
