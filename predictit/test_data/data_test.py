""" Test data definition. Data can be pickled and saved on disk. User can create own test data and use compare_models to find best models."""

import numpy as np


# Sin
def gen_sin(n):
    """Generate test data of length n in sinus shape.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Sinus shaped data.
    """

    t = np.linspace(0, 8 * np.pi, n)
    data1 = np.sin(t)
    return data1


# Sign
def gen_sign(n):
    """Generate test data of length n in signum shape.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Signum shaped data.
    """

    data1 = gen_sin(n)
    data2 = np.sign(data1)
    return data2


# Random
def gen_random(n):
    """Generate random test data of length.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Random test data.
    """

    data3 = np.random.randn(n) * 5 + 10
    return data3
