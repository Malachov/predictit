#%%
# import numpy as np
# import pandas as pd

# from pathlib import Path
# import sys
# import inspect
# import os

# oo = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]).T
# oo = np.random.randn(10000, 10000)
# arr = np.array(oo)
# df = pd.DataFrame(oo)

# from memory_profiler import profile
if __name__ == "__main__":

    import line_profiler
    import atexit

    profile = line_profiler.LineProfiler()
    atexit.register(profile.print_stats)

    # (precision=8)  # Just memory profiler
    @profile
    def profiled_function():
        pass
