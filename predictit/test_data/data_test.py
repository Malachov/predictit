import numpy as np
from numpy.random import randn
from .. import config


# TODO - custom data on demand - csv, txt
# Sin
def gen_sin(n):
    t = np.linspace(0, 8 * np.pi, n)
    data1 = np.sin(t)
    return data1

# Sign
def gen_sign(n):
    data1 = data_1(n)
    data2 = np.sign(data1)
    return data2

# Random
def gen_random(n):
    data3 = np.random.randn(n)*5 + 10
    return data3
