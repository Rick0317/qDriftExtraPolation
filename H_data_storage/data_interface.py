import typing


class DecomposedHamiltonian:
    """
    Hamiltonian object for decomposed Hamiltonian

    Attributes:
        sum_coeff (float): the sum of the coefficients
        coeff_list (list): the list of coefficients
        term_list (list): the list of terms
    """
    def __init__(self, sum_coeff: float,
                 coeff_list: list[float], term_list: list[tuple]):
        self.sum_coeff = sum_coeff
        self.coeff_list = coeff_list
        self.term_list = term_list


class TensorData:
    """
    The tensor data class is used to store the tensor data of Hamiltonians.

    Attributes:
        spatial_orb (int): int, spatial orbital
        tensor_1d (list[tuple]): list[tuple], 1d list of the sparse tensor data.
        each tuple is of the form (index 0, ... , index 2n, coefficient)
        where n is spatial_orb.
    """
    def __init__(self, spatial_orb: int, tensor_1d: list[tuple]):
        self.spatial_orb = spatial_orb
        self.tensor_1 = tensor_1d

    def decompose(self) -> DecomposedHamiltonian:
        '''
        The decomposed Hamiltonian class to store its relevant properties.
        By its format, the coefficient and the tensor data already satisfy
        the definition of the decomposed Hamiltonian.
        :return: DecomposedHamiltonian object
        '''
        sum_coeff = 0
        coeff_list = []
        term_list = []
        for term in self.tensor_1:
            sum_coeff += term[-1]
            coeff_list.append(term[-1])
            term_list.append(term[:-1])

        decomposed_hamiltonian = DecomposedHamiltonian(
            sum_coeff, coeff_list, term_list)

        return decomposed_hamiltonian


class Hamiltonian(TensorData):
    """
    The unitary hamiltonian class to store its relevant properties.

    lcu: Dict[str, tuple(int, Hamiltonian)], a mapping from the name of each term to the strength and the Hamiltonian that decomposes the Hamiltonian
    """
    def set_lcu(names:list[str], strengths:list[float], h_list:list[float]):
        d = {}
        for i, name in enumerate(names):
            assert name not in d, "duplicate name error"
            d[name] = (strengths[i], h_list[i])

