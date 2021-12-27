import mypythontools

if __name__ == "__main__":
    # All the parameters can be overwritten via CLI args
    mypythontools.utils.push_pipeline(deploy=True, test=False)
