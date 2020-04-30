""" Test data definition. Data can be pickled and saved on disk. User can create own test data and use compare_models to find best models."""

import numpy as np


def gen_sin(n=1000):
    """Generate test data of length n in sinus shape.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Sinus shaped data.
    """

    fs = 8000  # Sample rate
    f = 50
    x = np.arange(n)
    y = np.sin(2 * np.pi * f * x / fs)

    return y


def gen_sign(n=1000, periods=220):
    """Generate test data of length n in signum shape.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Signum shaped data.
    """

    sin = gen_sin(n=n)
    data = np.sign(sin)

    return data


# Random
def gen_random(n=1000):
    """Generate random test data of length.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Random test data.
    """

    data = np.random.randn(n) * 5 + 10

    return data


# Range
def gen_slope(n=1000):
    """Generate random test data of length.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Slope test data.
    """

    data = np.array(range(n))

    return data
