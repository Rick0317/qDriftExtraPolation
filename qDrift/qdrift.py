import numpy as np
import math
from scipy import linalg # for exponentials of matrices
from scipy.stats import rv_discrete
from H_data_storage.data_interface import *

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
        self.pk = [decomp.lst_coeff[i] / lm for i in range(len(decomp.lst_coeff))]

    def sample(self):
        return np.random.choice(self.h.get_decomp().lst_Hamil, p=self.pk)




def qDrift(hubbard: Hubbard, sample, epsilon: float):
    """The qDrift protocol. The variable names follow the definition in the "Random Compiler for Fast Hamiltonian Simulation" paper.
    
    :param hubbard: A Hubbard hamiltonian
    :param sample: the classicaloracle function SAMPLE() 
    :param epsilon: target precision
    :return: v_list: a list of sampled unitaries of the exponential form
    """
    hubbard.decompose(['obt', 'tbt'])
    sample = HamiltonianSampling(hubbard).sample
    lm = hubbard.get_decomp().sum_coeff
    N = math.ceil( 2 * (lm ** 2) * (t ** 2) / epsilon)
    i = 0
    v_list = []
    while i < N:
        i = i + 1
        j = sample()
        v_list.append(linalg.expm( 1j * lm * t * hubbard.get_decomp().lst_Hamil[j] / N))
    
    return v_list