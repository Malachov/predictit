from . import test_data
from . import config
import os
from pathlib import Path
import pickle

def pickle_data_all():

    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / "test_data"

    #data_folder = Path("test_data")
    for i, j in config.data_all_pickle.items():
        file_name = i + '.pickle'
        file_adress = data_folder / file_name
        with open(file_adress, "wb") as output_file: pickle.dump((j)[-config.datalength:], output_file)