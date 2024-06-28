import typing
import numpy as np

class DecomposedHamiltonian:
    """
    Hamiltonian object for decomposed Hamiltonian

    Attributes:
        sum_coeff (float): the sum of the coefficients
        lst_coeff: the parallel list of coefficients
        lst_Hamil: the parallel list of Hermitians
        decomp (Dict[str, tuple[float, TensorData]]): mapping from the name of a term to the tuple of a coefficient and a Hamiltonian
    """
    def __init__(self, names: list[str],
                 coeff_list: list[float], term_list: list):
        self.lst_coeff = coeff_list
        self.lst_Hamil = term_list
        self.decomp = {}
        for i, name in enumerate(names):
            assert name not in self.decomp, "duplicate name error"
            self.decomp[name] = (coeff_list[i], term_list[i])
        self.sum_coeff = np.sum(coeff_list)


class Hamiltonian:
    """
    The unitary hamiltonian class to store its relevant properties.

    decomp Dict[str, tuple(int, Hamiltonian)]: a mapping from the name of each term to the strength and the Hamiltonian that decomposes the Hamiltonian
    """
    decomp = None
    
    def __init__(self, spatial_orb: int, tensor_1d: list[tuple]):
        super().__init__(spatial_orb, tensor_1d)
        
    
    def get_decomp(self) -> DecomposedHamiltonian:
        assert self.decomp is not None, "no decomposition found (call .decompose first)"
        return self.decomp
    


class Hubbard(Hamiltonian):
    """The Hubbard model

    Attributes:
        spatial_orb (int): int, spatial orbital
        tensor_1d (list[tuple]): list[tuple], 1d list of the sparse tensor data.
        each tuple is of the form (index 0, ... , index 2n, coefficient)
        where n is spatial_orb.
    """
    def __init__(self, spatial_orb: int, tensor_1d: list[tuple]):
        self.spatial_orb = spatial_orb
        self.tensor_1 = tensor_1d
        
        
    def decompose(self, names: list[str]) -> None:
        '''
        Decompose this Hamiltonian object to store its relevant properties.
        By its format, the coefficient and the tensor data already satisfy
        the definition of the decomposed Hamiltonian.
        '''
        sum_coeff = 0
        coeff_list = []
        term_list = []
        for term in self.tensor_1:
            sum_coeff += term[-1]
            coeff_list.append(term[-1])
            term_list.append(term[:-1])

        decomposed_hamiltonian = DecomposedHamiltonian(names, coeff_list, term_list)

        self.decomp = decomposed_hamiltonian
    
    