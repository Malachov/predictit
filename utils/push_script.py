import os
import inspect
from pathlib import Path
import sys

# Find paths and add to sys.path to be able to import local modules
test_path = Path(
    os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
).parent
root_path = test_path.parent

if root_path not in sys.path:
    sys.path.insert(0, root_path.as_posix())

import mypythontools

if __name__ == "__main__":
    mypythontools.utils.push_pipeline(deploy=False)
