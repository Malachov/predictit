# predictit
Library/framework for making predictions. Choose best of 20 models (ARIMA, regressions, LSTM...). Preprocess data and chose optimal parameters of predictions.

## Output
Output is plotly interactive graph or deploying to database.
![Printscreen of output HTML graph](/output_example.png)

## How to use
Download it. Open config.py. Two possible inputs. CSV or Database. Config is quite clear. Setup usually 1 or 0. Finally run main.py. With config.optimize_it it's pretty time consuming. So firs turn optimize_it off. Find best three models and then optimize. Do not try to optimize LSTM...

You can also download it from Pypi with

```Python 
pip install predictit
```

### Simple example with Pypi
```Python
import predictit

predictit.main.predict()  # Make prediction od test data
```
### Example with config
```Python
predictit.config.predicts = 30  # Create 30 predictions
predictit.config.csv_adress = r'E:\VSCODE\Diplomka\test_data\daily-minimum-temperatures.csv'  # Load CSV file with data
predictit.config.datalength = 1000  # Consider only last 1000 data points  
predictit.config.predicted_columns_names = 'Temp'  # Column name that we want to predict
predictit.config.optimizeit = 0  # Find or not best parameters for models
predictit.config.compareit = 6  # Visualize 6 best models
predictit.config.repeatit = 4  # Repeat calculation 4x times on shifted data to reduce chance
predictit.config.other_columns = 0  # Whether use other columns or not

# Chose models that will be computed
used_models = {
            "AR (Autoregression)": models.ar,

            "ARIMA (Autoregression integrated moving average)": models.arima,

            "Autoregressive Linear neural unit": models.autoreg_LNU,
            "Conjugate gradient": models.cg,

            "Extreme learning machine": models.elm,

            "LSTM": models.lstm,

            "Sklearn universal": models.sklearn_universal,

            "Bayes Ridge Regression": models.regression_bayes_ridge,
            "Hubber regression": models.regression_hubber,
            "Lasso Regression": models.regression_lasso,
           }
           
# Define parameters of models
models_parameters = {


        "AR (Autoregression)": {"plot": 0, 'method': 'cmle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs'},
        "ARMA": {"plot": 0, "p": 3, "q": 0, 'method': 'mle', 'ic': 'aic', 'trend': 'nc', 'solver': 'lbfgs', 'forecast_type': 'in_sample'},
        "ARIMA (Autoregression integrated moving average)": {"p": 12, "d": 0, "q": 1, "plot": 0, 'method': 'css', 'ic': 'aic', 'trend': 'nc', 'solver': 'nm', 'forecast_type': 'out_of_sample'},

        "Autoregressive Linear neural unit": {"plot": 0, "lags": n_steps_in, "mi": 1, "minormit": 0, "tlumenimi": 1},
        "Conjugate gradient": {"n_steps_in": 30, "epochs": 5, "constant": 1, "other_columns_lenght": None, "constant": None},

        "Extreme learning machine": {"n_steps_in": 20, "output_shape": 'one_step', "other_columns_lenght": None, "constant": None, "n_hidden": 20, "alpha": 0.3, "rbf_width": 0, "activation_func": 'selu'},


        "LSTM": {"n_steps_in": 50, "save": saveit, "already_trained": 0, "epochs": 70, "units":50, "optimizer":'adam', "loss":'mse', "verbose": 1, "activation": 'relu', "timedistributed": 0, "metrics": ['mape']},

        "Sklearn universal": {"n_steps_in": n_steps_in, "output_shape": "one_step", "model": models.default_regressor, "constant": None},

        "Bayes Ridge Regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alpha_1": 1.e-6, "alpha_2": 1.e-6, "lambda_1": 1.e-6, "lambda_2": 1.e-6},
        "Hubber regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "epsilon": 1.35, "alpha": 0.0001},
        "Lasso Regression": {"n_steps_in": n_steps_in, "output_shape": output_shape, "other_columns_lenght": None, "constant": None, "alpha": 0.6}
        
        }
        
predictit.main.predict()
```
