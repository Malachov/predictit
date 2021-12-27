# import subprocess

# import numpy as np
# import pandas as pd

# import mydatapreprocessing
import mypythontools

mypythontools.tests.setup_tests()

import predictit
from predictit import config
from predictit import optimization


def test_config_optimization():
    config.update(
        {
            "multiprocessing": "process",
            "used_models": ["LNU", "Sklearn regression"],
            "optimization": {"default_n_steps_in": [3, 6, 9]},
            "plot_all": True,
        }
    )

    result = optimization.config_optimization(config=config)

    assert result.best_values


def test_hyperparameter_optimization():
    config.update(
        {
            "used_models": ["LNU", "Sklearn regression"],
            "optimizeit": True,
            "optimizeit_limit": 0.1,
            "optimizeit_details": 2,
            "optimizeit_plot": True,
            "iterations": 2,
            "fragments": 3,
            "models_parameters_limits": {
                "LNU": {"learning_rate": [0.0001, 0.1]},
                "Sklearn regression": ["DecisionTreeRegressor", "LinearRegression"],
            },
        }
    )

    result = predictit.predict(config=config)

    assert result.hyperparameter_optimization["LNU"].best_params["learning_rate"] > 0.0001 < 0.1


def test_input_optimization():
    result = predictit.input_optimization()
    assert result.best_data_dict
    assert result.results
    assert result.tables


# For deeper debug, uncomment problematic test
if __name__ == "__main__":

    # test_hyperparameter_optimization()
    # test_config_optimization()
    # test_input_optimization()

    pass
