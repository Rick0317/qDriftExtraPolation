import math

from numpy import ndarray
from scipy import linalg  # for exponentials of matrices

from scripts.database import DataManager
from scripts.database.data_interface import *

t = 0.1


class HamiltonianSampling:
    """This class contains methods for sampling protocol given Hamiltonian
    Attributes:
        h: Hamiltonian object
        pk: probability distribution
    """

    def __init__(self, h: Hamiltonian):
        self.h = h
        decomp = h.get_decomp()
        lm = decomp.sum_abs_coeff
        lst_term = decomp.lst_Hamil
        self.pk = [np.abs(lst_term[i].coefficient) / lm for i in range(len(lst_term))]

    def sample(self) -> Tensor:
        return np.random.choice(np.array(self.h.get_decomp().lst_Hamil), p=self.pk)


class QDrift:
    def __init__(self, h: Hamiltonian, t):
        self.h = h
        self.t = t
        self.sample = HamiltonianSampling(h).sample

    def qdrift(self, N:int) -> tuple[list[ndarray], list[ndarray]]:
        """The qDrift protocol. The variable names follow the definition in the "Random Compiler for Fast Hamiltonian Simulation" paper.

        :param hubbard: A Hubbard hamiltonian
        :param sample: the classical oracle function SAMPLE()
        :param epsilon: target precision
        :return: v_list: a list of sampled unitaries of the exponential form
        """
        lm = self.h.get_decomp().sum_coeff
        # N = math.ceil(2 * (lm ** 2) * (t ** 2) / epsilon)
        i = 0
        v_list = []
        h_list = []
        while i < N:
            i = i + 1
            j = self.sample()
            h_list.append(j.matrix * j.coefficient)
            v_list.append(linalg.expm(1j * lm * self.t * j.matrix * j.coefficient / N))

        return h_list, v_list

