import typing

class TensorData:
    """
    The tensor data class is used to store the tensor data of Hamiltonians.

    spatial_orb: int, spatial orbital
    tensor_1d: list[tuple], 1d list of the sparse tensor data.
    """
    def __init__(self, spatial_orb: int, tensor_1d: list[tuple]):
        self.spatial_orb = spatial_orb
        self.tensor_1 = tensor_1d

class Hamiltonian(TensorData):
    """
    The unitary hamiltonian class to store its relevant properties.
    
    lcu: Dict[str, tuple(int, Hamiltonian)], a mapping from the name of each term to the strength and the Hamiltonian that decomposes the Hamiltonian
    """
    def set_lcu(names:list[str], strengths:list[float], h_list:list[Hamiltonian]):
        d = {}
        for i, name in enumerate(names):
            assert name not in d, "duplicate name error"
            d[name] = (strengths[i], h_list[i])