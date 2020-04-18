"""Module that profile time and memory usage of used libraries.
Results are: avoid cufflinks if possible and lazy load plotly only if ploting.
All the rest libraries have no impact"""

from memory_profiler import profile
from prettytable import PrettyTable
import time as time_lib

time_parts_table = PrettyTable()

@profile(precision=8)
def profiling_func():

    # Definition of the table for spent time on code parts
    time_parts_table.field_names = ["Part", "Time"]

    def update_time_table(time_last):
        time_parts_table.add_row([part, time_lib.time() - time_last])
        return time_lib.time()

    time_point = time_lib.time()

    part = 'sys'
    import sys
    time_point = update_time_table(time_point)

    part = 'Path'
    from pathlib import Path
    time_point = update_time_table(time_point)

    part = 'np'
    import numpy as np
    time_point = update_time_table(time_point)

    part = 'PrettyTable'
    from prettytable import PrettyTable
    time_point = update_time_table(time_point)

    part = 'time'
    import time
    time_point = update_time_table(time_point)

    part = 'os'
    import os
    time_point = update_time_table(time_point)

    part = 'pl'
    import plotly as pl
    time_point = update_time_table(time_point)

    part = 'cf'
    import cufflinks as cf
    time_point = update_time_table(time_point)

    part = 'pd'
    import pandas as pd
    time_point = update_time_table(time_point)

    part = 'warnings'
    import warnings
    time_point = update_time_table(time_point)

    part = 'traceback'
    import traceback
    time_point = update_time_table(time_point)

    part = 'argparse'
    import argparse
    time_point = update_time_table(time_point)

    part = 'inspect'
    import inspect
    time_point = update_time_table(time_point)

    return time_parts_table

if __name__ == "__main__":
    print(profiling_func())
