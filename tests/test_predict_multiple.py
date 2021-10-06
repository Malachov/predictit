import pandas as pd

import mypythontools

mypythontools.paths.PROJECT_PATHS.add_ROOT_PATH_to_sys_path()

from conftest import validate_result
import predictit
from predictit import config


def test_main_multiple():

    data = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    ).iloc[:120]
    data["Another col"] = data["Temp"] + 16
    config_test = config.copy()
    config_test.update(
        {
            "datetime_column": "Date",
            "freqs": ["D", "H"],
            "data": data,
            "predicted_columns": ["*"],
            "remove_nans_threshold": 0.9,
            "remove_nans_or_replace": 2,
            "optimization": False,
        }
    )

    result = predictit.predict_multiple_columns(config=config_test)
    result_dataframe = next(iter(result.best_predictions_dataframes.values()))
    assert (
        result.results
        and len(result.results) == 4
        and validate_result(result_dataframe)
        and len(result_dataframe.columns) == 2
    )


if __name__ == "__main__":

    # test_main_multiple()

    pass
