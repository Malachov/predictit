#%%
import sys
from pathlib import Path
import numpy as np
import inspect
import os
import pandas as pd

# oo = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]).T
oo = np.random.randn(10000, 10000)
arr = np.array(oo)
df = pd.DataFrame(oo)



def profiled_function():

    ee = df.values

    return ee


def profiled_function2():

    ee = arr.diff(axis=0)

    return ee


if __name__ == "__main__":
    print(profiled_function())
    print(profiled_function2())


