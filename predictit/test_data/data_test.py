""" Test data definition. Data can be pickled and saved on disk"""

import numpy as np


# Sin
def gen_sin(n):
    t = np.linspace(0, 8 * np.pi, n)
    data1 = np.sin(t)
    return data1


# Sign
def gen_sign(n):
    data1 = gen_sin(n)
    data2 = np.sign(data1)
    return data2


# Random
def gen_random(n):
    data3 = np.random.randn(n) * 5 + 10
    return data3
