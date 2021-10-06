import mypythontools

mypythontools.paths.PROJECT_PATHS.add_ROOT_PATH_to_sys_path()

import predictit


def test_find_optimal_input_for_models():
    result = predictit.find_optimal_input_for_models()
    assert result.best_data_dict
    assert result.results
    assert result.tables


# For deeper debug, uncomment problematic test
if __name__ == "__main__":
    pass
