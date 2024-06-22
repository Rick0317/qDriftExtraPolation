class TensorData:
    """
    The tensor data class is used to store the tensor data of Hamiltonians.

    spatial_orb: int, spatial orbital
    tensor_1d: list[tuple], 1d list of the sparse tensor data.
    """
    def __init__(self, spatial_orb: int, tensor_1d: list[tuple]):
        self.spatial_orb = spatial_orb
        self.tensor_1 = tensor_1d

