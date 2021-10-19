import subprocess

import numpy as np
import pandas as pd

import mydatapreprocessing
import mypythontools

mypythontools.paths.PROJECT_PATHS.add_ROOT_PATH_to_sys_path()

from conftest import validate_result
import predictit
from predictit import config


def test_default_config_and_outputs():
    config.reset()

    results_default = predictit.predict()

    # Validate tables
    validate_tables = True

    for i in [
        results_default.tables.detailed,
        results_default.tables.simple,
        results_default.tables.time,
    ]:
        if not isinstance(i, str) or len(i) < 20:
            validate_tables = False

    assert validate_result(results_default.predictions)
    assert validate_result(results_default.best_prediction)
    assert validate_result(results_default.results_df)
    assert validate_tables
    assert np.isnan(results_default.with_history["Predicted column"].iloc[-1])
    assert not np.isnan((results_default.with_history.iloc[-1, -1]))


def test_config_inputs():
    predict_with_params = predictit.predict(predicts=3)

    data = [
        pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"]),
        pd.DataFrame(np.random.randn(80, 3), columns=["e", "b", "c"]),
    ]
    predict_with_positional_params = predictit.predict(data, "b")

    config_test = config.copy()
    config_test.output.predicts = 4
    predict_config_as_param = predictit.predict(config=config_test)

    cli_args_str = (
        "python predictit/main.py --used_function predict --data "
        "'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv' "
        "--predicted_column 'Temp' "
    )

    result_cli = subprocess.check_output(cli_args_str.split(" "))

    assert len(predict_with_params.best_prediction) == 3
    assert len(predict_config_as_param.best_prediction) == 4
    assert validate_result(predict_with_positional_params.predictions)
    assert "Best model is" in str(result_cli)


def test_config_variations_1():

    config.update(
        {
            "embedding": "label",
            "plot_library": "plotly",
            "data": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
            "predicted_column": "Temp",
            "datetime_column": "Date",
            "freq": "D",
            "error_criterion": "dtw",
            "max_imported_length": 300,
            "remove_nans_threshold": 0.85,
            "remove_nans_or_replace": "neighbor",
            "trace_processes_memory": False,
            "print_number_of_models": 10,
            "add_fft_columns": 16,
            "data_extension": {
                "differences": True,
                "second_differences": True,
                "multiplications": True,
                "rolling_means": 10,
                "rolling_stds": 10,
                "mean_distances": True,
            },
            "analyzeit": 3,
            "correlation_threshold": 0.2,
            "optimizeit": False,
            "standardizeit": "01",
            "multiprocessing": "process",
            "smoothit": (19, 2),
            "power_transformed": True,
            "analyze_seasonal_decompose": {"period": 32, "model": "additive"},
            "confidence_interval": 0.6,
        }
    )

    assert validate_result(predictit.predict().predictions)


def test_config_variations_2():

    config.update(
        {
            "data": "test_random",
            "predicted_column": "",
            "predicts": 4,
            "analyzeit": 1,
            "datetime_column": "",
            "freq": "M",
            "datalength": 100,
            "default_n_steps_in": 3,
            "data_transform": "difference",
            "error_criterion": "rmse",
            "remove_outliers": 3,
            "print_number_of_models": None,
            "print_table": "detailed",
            "plot_library": "matplotlib",
            "show_plot": True,
            "print_time_table": True,
            "correlation_threshold": 0.2,
            "data_extension": None,
            "optimizeit": True,
            "optimizeit_limit": 0.1,
            "optimizeit_details": 2,
            "optimizeit_plot": True,
            "standardizeit": None,
            "multiprocessing": "pool",
            "confidence_interval": None,
            "trace_processes_memory": True,
            "used_models": ["Bayes ridge regression"],
            "models_parameters": {
                "Bayes ridge regression": {
                    "model": "BayesianRidge",
                    "n_iter": 300,
                    "alpha_1": 1.0e-6,
                    "alpha_2": 1.0e-6,
                    "lambda_1": 1.0e-6,
                    "lambda_2": 1.0e-6,
                }
            },
            "fragments": 4,
            "iterations": 2,
            "models_parameters_limits": {
                "Bayes ridge regression": {
                    "alpha_1": [0.1e-6, 3e-6],
                    "model": ["BayesianRidge", "LassoRegression", "LinearRegression"],
                }
            },
        }
    )
    config.hyperparameter_optimization.optimizeit_limit = 10
    assert validate_result(predictit.predict().predictions)


def test_config_variations_3():

    df = pd.DataFrame([range(300), range(1000, 1300)]).T
    df["time"] = pd.date_range("2018-01-01", periods=len(df), freq="H")

    config.update(
        {
            "data": df,
            "datetime_column": "time",
            "freq": "",
            "predicted_column": False,
            "datalength": 100,
            "other_columns": True,
            "analyzeit": 2,
            "default_other_columns_length": 5,
            "data_transform": None,
            "repeatit": 10,
            "error_criterion": "max_error",
            "multiprocessing": None,
            "remove_outliers": None,
            "remove_nans_threshold": 0.8,
            "remove_nans_or_replace": "mean",
            "optimization": True,
            "optimization_variable": "default_n_steps_in",
            "optimization_values": [3, 6, 9],
            "plot_all_optimized_models": True,
            "correlation_threshold": 0.4,
            "standardizeit": "standardize",
            "predicts": 7,
            "smoothit": None,
        }
    )

    assert validate_result(predictit.predict().predictions)


def test_most_models():

    a = np.array(range(200)) + np.random.rand(200)
    b = np.array(range(200)) + mydatapreprocessing.generate_data.sin(200)
    df = pd.DataFrame([a, b]).T

    config.update(
        {
            "data": df,
            "predicts": 7,
            "datalength": 200,
            "default_n_steps_in": 3,
            "error_criterion": "mape",
            "used_models": [
                "ARMA",
                "ARIMA",
                "autoreg",
                "SARIMAX",
                "LNU",
                "LNU normalized",
                "LNU with weights predicts",
                "Sklearn regression",
                "Bayes ridge regression",
                "Passive aggressive regression",
                "Gradient boosting",
                "KNeighbors regression",
                "Decision tree regression",
                "Hubber regression",
                "Bagging regression",
                "Stochastic gradient regression",
                "Extreme learning machine",
                "Gen Extreme learning machine",
                "Extra trees regression",
                "Random forest regression",
                "Tensorflow LSTM",
                "Tensorflow MLP",
                "Average short",
                "Average long",
                "Regression",
                "Ridge regression",
            ],
        }
    )

    result = predictit.predict()
    assert result.predictions.isnull().sum().sum() <= 2


def test_presets():
    config.update(
        {
            "data": "https://www.ncdc.noaa.gov/cag/global/time-series/globe/land_ocean/ytd/12/1880-2016.json",
            "request_datatype_suffix": ".json",
            "predicted_table": "data",
            "data_orientation": "index",
            "predicted_column": 0,
            "datalength": 100,
        }
    )

    config.update({"use_config_preset": "fast"})
    preset_result = predictit.predict().predictions
    assert validate_result(preset_result)

    config.update({"use_config_preset": "normal"})
    assert validate_result(preset_result)


# def test_GUI():
#     predictit.gui_start.run_gui()


# For deeper debug, uncomment problematic test
if __name__ == "__main__":

    # test_default_config_and_outputs()
    # test_config_inputs()
    # test_config_variations_1()
    # test_config_variations_2()
    # test_config_variations_3()
    # test_most_models()
    # test_other_models_functions()
    # test_presets()

    pass

if __name__ == "__main__":
    test_most_models()
    pass
