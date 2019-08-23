from data_test import *
from config import *
import os
from pathlib import Path

def pickle_data_all(datalength=1000):

    data_folder = Path("test_data")
    for i, j in data_all_pickle.items():
        file_name = i + '.pickle'
        file_adress = data_folder / file_name
        with open(file_adress, "wb") as output_file: pickle.dump(np.array(eval(j)[-datalength:]), output_file)