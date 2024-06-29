import math
from scipy import linalg  # for exponentials of matrices
from scripts.database.data_interface import *

t = 6000


class HamiltonianSampling:
    """This class contains methods for sampling protocol given Hamiltonian
    Attributes:
        h: Hamiltonian object
        pk: probability distribution
    """

    def __init__(self, h: Hamiltonian):
        self.h = h
        decomp = h.get_decomp()
        lm = decomp.sum_coeff
        lst_term = decomp.lst_Hamil
        self.pk = [lst_term[i].coefficient / lm for i in range(len(lst_term))]

    def sample(self) -> Tensor:
        return np.random.choice(np.array(self.h.get_decomp().lst_Hamil), p=self.pk)


def qdrift(hubbard: Hubbard, sample, epsilon: float):
    """The qDrift protocol. The variable names follow the definition in the "Random Compiler for Fast Hamiltonian Simulation" paper.
    
    :param hubbard: A Hubbard hamiltonian
    :param sample: the classicaloracle function SAMPLE() 
    :param epsilon: target precision
    :return: v_list: a list of sampled unitaries of the exponential form
    """
    hubbard.decompose(['obt', 'tbt'])
    sample = HamiltonianSampling(hubbard).sample
    lm = hubbard.get_decomp().sum_coeff
    N = math.ceil(2 * (lm ** 2) * (t ** 2) / epsilon)
    i = 0
    v_list = []
    while i < N:
        i = i + 1
        j = sample()
        v_list.append(linalg.expm(1j * lm * t * j.matrix / N))

    return v_list
