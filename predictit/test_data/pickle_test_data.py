"""Module containing function for pickling test data if set up in config.py."""

from pathlib import Path
import pickle


def pickle_data_all(data_all_pickle, datalength=1000000):
    """Pickle test data on disk for faster loading."""
    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / "pickled"

    #data_folder = Path("test_data")
    for i, j in data_all_pickle.items():
        file_name = i + '.pickle'
        file_adress = data_folder / file_name

        with open(file_adress, "wb") as output_file:
            pickle.dump((j)[-datalength:], output_file)
