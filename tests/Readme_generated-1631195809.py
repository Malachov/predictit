"""pytest file built from C:/Users/Malac/ownCloud/Github/predictit_library/README.md"""
import pytest

from phmdoctest.fixture import managenamespace


@pytest.fixture(scope="module")
def _phm_setup_teardown(managenamespace):
    # setup code line 102.
    import predictit
    import numpy as np
    import pandas as pd

    from predictit import config

    managenamespace(operation="update", additions=locals())
    yield
    # <teardown code here>

    managenamespace(operation="clear")


pytestmark = pytest.mark.usefixtures("_phm_setup_teardown")


def test_code_74():
    config.update(
        {
            "datalength": 300,  # Used datalength
            "predicts": 14,  # Number of predicted values
            "default_n_steps_in": 12,  # Value of recursive inputs in model (do not use too high - slower and worse predictions)
        }
    )

    # After if you setup prediction as needed, it's simple

    predictions = predictit.predict()

    # Caution- no assertions.


def test_code_110():
    predictions_1 = predictit.predict(
        data=np.random.randn(100, 2), predicted_column=1, predicts=3
    )

    # Caution- no assertions.


def test_code_116():
    my_data = pd.DataFrame(np.random.randn(100, 2), columns=["a", "b"])
    predictions_1_positional = predictit.predict(my_data, "b")

    # Caution- no assertions.


def test_code_136():
    my_data_array = np.random.randn(200, 2)  # Define your data here

    # You can compare it on same data in various parts or on different data (check configuration on how to insert dictionary with data names)
    config.update(
        {
            "data_all": {
                "First part": (my_data_array[:100], 0),
                "Second part": (my_data_array[100:], 1),
            },
            "predicted_column": 0,
        }
    )
    compared_models = predictit.compare_models()

    # Caution- no assertions.


def test_code_149():
    config.data = np.random.randn(120, 3)
    config.predicted_columns = [
        "*"
    ]  # Define list of columns or '*' for predicting all of the numeric columns
    config.used_models = [
        "Conjugate gradient",
        "Decision tree regression",
    ]  # Use just few models to be faster

    multiple_columns_prediction = predictit.predict_multiple_columns()

    # Caution- no assertions.


def test_code_159():
    config.update(
        {
            "data": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
            "predicted_column": "Temp",
            "datalength": 120,
            "optimization": True,
            "optimization_variable": "default_n_steps_in",
            "optimization_values": [4, 6, 8],
            "plot_all_optimized_models": False,
            "print_table": "detailed",  # Print detailed table
            "print_result_details": True,
            "used_models": ["AR", "Sklearn regression"],
        }
    )

    predictions_optimized_config = predictit.predict()

    # Caution- no assertions.


def test_code_204():

    import mydatapreprocessing as mdp
    from mypythontools.plots import plot
    from predictit.analyze import analyze_column

    data = "https://blockchain.info/unconfirmed-transactions?format=json"

    # Load data from file or URL
    data_loaded = mdp.load_data.load_data(
        data,
        request_datatype_suffix=".json",
        predicted_table="txs",
        data_orientation="index",
    )

    # Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
    # only numeric data and resample ifg configured.
    data_consolidated = mdp.preprocessing.data_consolidation(
        data_loaded,
        predicted_column="weight",
        remove_nans_threshold=0.9,
        remove_nans_or_replace="interpolate",
    )

    # Predicted column is on index 0 after consolidation)
    analyze_column(data_consolidated.iloc[:, 0])

    # Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
    # transformation, so unpack it with _
    data_preprocessed, _, _ = mdp.preprocessing.preprocess_data(
        data_consolidated,
        remove_outliers=True,
        smoothit=False,
        correlation_threshold=False,
        data_transform=False,
        standardizeit="standardize",
    )

    # Plot inserted data
    plot(data_preprocessed)

    # Caution- no assertions.


def test_code_237():

    import mydatapreprocessing as mdp

    data = mdp.generate_data.sin(1000)
    test = data[-7:]
    data = data[:-7]
    data = mdp.preprocessing.data_consolidation(data)
    (X, y), x_input, _ = mdp.create_model_inputs.create_inputs(
        data.values, "batch", input_type_params={"n_steps_in": 6}
    )  # First tuple, because some models use raw data - one argument, e.g. [1, 2, 3...]

    trained_model = predictit.models.sklearn_regression.train(
        (X, y), model="BayesianRidge"
    )
    predictions_one_model = predictit.models.sklearn_regression.predict(
        x_input, trained_model, predicts=7
    )

    predictions_one_model_error = (
        predictit.evaluate_predictions.compare_predicted_to_test(
            predictions_one_model, test, error_criterion="mape"
        )
    )  # , plot=1

    # Caution- no assertions.


def test_code_255():
    config.update(
        {
            "data": r"https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",  # Full CSV path with suffix
            "predicted_column": 0,  # Column name that we want **to** predict
            "predicted_column": "Temp",  # Column name that we want to predict
            "datalength": 200,
            "predicts": 7,  # Number of predicted values - 7 by default
            "repeatit": 50,  # Repeat calculation times on shifted data to evaluate error criterion
            "other_columns": False,  # Whether use other columns or not
            # Chose models that will be computed - remove if you want to use all the models
            "used_models": [
                "AR",
                "ARIMA",
                "LNU",
                "Conjugate gradient",
                "Sklearn regression",
                "Bayes ridge regression one column one step",
                "Decision tree regression",
            ],
            # Define parameters of models
            "models_parameters": {
                "AR": {
                    "used_model": "ar",
                    "method": "cmle",
                    "trend": "nc",
                    "solver": "lbfgs",
                },
                "ARIMA": {
                    "used_model": "arima",
                    "p": 6,
                    "d": 0,
                    "q": 0,
                },
                "LNU": {
                    "mi_multiple": 1,
                    "mi_linspace": (1e-5, 1e-4, 3),
                    "epochs": 10,
                    "w_predict": 0,
                    "minormit": 0,
                },
                "Conjugate gradient": {"epochs": 200},
                "Bayes ridge regression": {
                    "model": "BayesianRidge",
                    "n_iter": 300,
                    "alpha_1": 1.0e-6,
                    "alpha_2": 1.0e-6,
                    "lambda_1": 1.0e-6,
                    "lambda_2": 1.0e-6,
                },
            },
        }
    )

    predictions_configured = predictit.predict()

    # Caution- no assertions.
