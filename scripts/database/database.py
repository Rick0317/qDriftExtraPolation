from scripts.database.data_interface import *
import pickle
import os


class DataManager:
    """Class that provides methods for saving and loading Hamiltonian data
    """
    data_path = "../data/"
    def __init__(self, data_path):
        self.data_path = data_path

    def mksubdir(self, subdir_name: str):
        """
        Creates subdirectory under the data folder
        Args:
            subdir_name: the name of the subdirectory to create
        """
        subdir_path = os.path.join(DataManager.data_path, subdir_name)
        os.mkdir(subdir_path)
        print(f"Created subdirectory {subdir_name}")

    def save(self, subdir, name, data):
        """
        Saves data with the name to the directory specified by dir
        Args:
            data: the data to be saved
            subdir: the name of the subdirectory under the data folder in which the data is to be saved
            name: the name of the file to be saved
        """
        path = os.path.join(DataManager.data_path, subdir)
        if not os.path.exists(path):
            self.mksubdir(subdir)
        assert os.path.exists(path), 'the path "{}" cannot be found'.format(path)
        path = os.path.join(path, name)
        with open(path + ".pkl", 'wb') as f:
            pickle.dump(data, f)
        print("Data saved at " + path + ".pkl")

    def load(self, subdir, name) -> Hamiltonian:
        """
        Loads Hamiltonian data from the file specified by dir.
        Args:
            subdir: the subdirectory of the file to be loaded
            name: the name of the file to be loaded
        Returns:
            the Hamiltonian object loaded from the file
        """
        path = os.path.join(DataManager.data_path, subdir, name + '.pkl')
        assert os.path.isfile(path), 'the file "{}" cannot be found'.format(path)

        with open(path, "rb") as f:
            loaded_data = pickle.load(f)

        print("Data loaded from " + path)
        return loaded_data




