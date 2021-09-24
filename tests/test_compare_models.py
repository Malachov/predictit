import numpy as np

import predictit
from predictit import config


def test_compare_models():
    data_all = None  # Means default sin, random, sign
    result = predictit.compare_models(data_all=data_all)

    assert result


def test_compare_models_list():
    dummy_data = np.random.randn(300)
    data_all = [dummy_data[:100], dummy_data[100:200], dummy_data[200:]]

    result = predictit.compare_models(data_all=data_all)

    assert result


def test_compare_models_with_optimization():
    config.update(
        {
            "data_all": None,
            "optimization": True,
            "optimization_variable": "data_transform",
            "optimization_values": [None, "difference"],
        }
    )

    result = predictit.compare_models()

    assert result


if __name__ == "__main__":

    # test_compare_models()
    # test_compare_models_list()
    # test_compare_models_with_optimization()

    pass
