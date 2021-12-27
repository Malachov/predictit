#%%

# This file is for profiling some arbitrary code. If profile some functions from package,
# use profiled.ipynb

import numpy as np
import pandas as pd

from memory_profiler import profile

if __name__ == "__main__":

    import line_profiler
    import atexit

    profile = line_profiler.LineProfiler()
    atexit.register(profile.print_stats)

    # (precision=8)  # Just memory profiler
    @profile
    def profiled_function():

        pass

    profiled_function()
