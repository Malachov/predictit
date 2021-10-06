import pytest
import numpy as np
import matplotlib

import mylogging
import mypythontools

mypythontools.tests.setup_tests(matplotlib_test_backend=True)

import predictit
from predictit.configuration import config


config_for_tests = {
    "is_tested": True,
    "predicted_column": 0,
    "logger_level": "ERROR",
    "logger_color": False,
    "show_plot": False,
    "data": "test_sin",
    "datalength": 120,
    "default_n_steps_in": 3,
    "analyzeit": 0,
    "optimization": False,
    "optimizeit": False,
    "used_models": [
        "AR",
        "Conjugate gradient",
        "Sklearn regression",
    ],
    "print_time_table": False,
    "print_result_details": False,
    "print_comparison_result_details": False,
    "print_table": None,
    "print_comparison_table": None,
    "analyze_seasonal_decompose": None,
}


@pytest.fixture(autouse=True)
def setup_tests(doctest_namespace):

    doctest_namespace["predictit"] = predictit
    doctest_namespace["mylogging"] = mylogging
    doctest_namespace["matplotlib"] = matplotlib

    # Config reset to default for each test
    config.reset()
    config.update(config_for_tests)


def validate_result(data):
    """Return true if result is valid. It means no np.nan in results.

    Args:
        data (np.array, pd.DataFrame): Tested data.
    """
    return not np.isnan(np.array(data).min())
