import numpy as np
import math
import random # for random sampling
from scipy import linalg # for exponentials of matrices
from functools import reduce # for pi-produc
from "..\H_data_storage\data_interface.py"

t = 6000

def sample()

def qDrift(hubbard, sample, epsilon: float):
    """The qDrift protocol. The variable names follow the definition in the "Random Compiler for Fast Hamiltonian Simulation" paper.
    
    :param hubbard: A Hubbard hamiltonian
    :param sample: the classicaloracle function SAMPLE() 
    :param epsilon: target precision
    :return: v_list: a list of sampled unitaries of the exponential form
    """
    lm = sum(term[0] for term in hamiltonian_terms)
    N = math.ceil( 2 * (lm ** 2) * (t ** 2) / epsilon)
    i = 0
    v_list = []
    while i < N:
        i = i + 1
        j = sample()
        v_list.append(linalg.expm( 1j * lm * t * hamiltonian_terms[j][1] / N))
    
    return v_list