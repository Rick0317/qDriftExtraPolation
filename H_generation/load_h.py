import pickle

import numpy as np
from typing import cast
from H_data_storage.data_interface import TensorData


def get_obt_and_tbt(directory_path, file_name):
    """
    Load the one-body tensor and the two-body tensor of the Hamiltonian stored in the given path
    :param directory_path:
    :param file_name:
    :return:
    """
    with open(directory_path + file_name + ".pkl", "rb") as f:
        loaded_data = pickle.load(f)

    return loaded_data["obt_data"], loaded_data["tbt_data"]


def obt_from_1d(obt_1d):
    """
    Convert a 1d version of the one-body tensor into the original one-body tensor
    :param obt_1d:
    :return:
    """
    assert type(obt_1d) == TensorData, "obt_1d should be a TensorData"

    obt_data: TensorData = cast(TensorData, obt_1d)
    n = obt_data.spatial_orb
    obt = np.zeros((2 * n, 2 * n))
    for elem in obt_data.tensor_1:
        obt[elem[0], elem[1]] = elem[2]

    return obt


def tbt_from_1d(tbt_1d):
    """
    Convert a 1d version of the two-body tensor into the original two-body tensor
    :param tbt_1d:
    :return:
    """
    assert type(tbt_1d) == TensorData, "tbt_1d should be a TensorData"

    tbt_data: TensorData = cast(TensorData, tbt_1d)
    n = tbt_data.spatial_orb
    tbt = np.zeros((2 * n, 2 * n, 2 * n, 2 * n))
    for elem in tbt_data.tensor_1:
        tbt[elem[0], elem[1], elem[2], elem[3]] = elem[4]
    return tbt
