import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pytest
import numpy as np
from qiskit.quantum_info import Pauli, Operator
from scripts.algo.qft_qpe_qdrift import qdrift_sample
from scipy.linalg import expm

def test_exponential():
    """Test basic functionality of qdrift_sample"""
    # Create a simple 2-qubit Hamiltonian with known terms
    J = 0.6
    z1z2 = Pauli('ZZ').to_matrix()

    H = J * z1z2


    expm()
