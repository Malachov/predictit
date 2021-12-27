""" Test module. Auto pytest that can be started in IDE or with::

    python -m pytest . --cov predictit --cov-report xml:.coverage.xml

in terminal in tests folder.
"""

from . import (
    test_compare_models,
    test_misc,
    test_predict,
    test_predict_multiple,
    test_visual,
)

__all__ = [
    "test_compare_models",
    "test_misc",
    "test_predict",
    "test_predict_multiple",
    "test_visual",
]
