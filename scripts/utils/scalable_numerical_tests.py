from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, circuit_drawer
from typing import Optional, Union
import random
import numpy as np
from functools import reduce
import scipy.linalg
import pandas as pd

# Type aliases
coefficient = float
Hamiltonian = list[tuple[coefficient, Pauli]]

# Function to generate a random Hamiltonian
def generate_random_hamiltonian(num_qubits, num_terms) -> Hamiltonian:
    pauli_matrices = [Pauli('X'), Pauli('Z'), Pauli('I')]
    hamiltonian_terms = []
    for _ in range(num_terms):
        pauli_string = reduce(lambda x, y: x.tensor(y), random.choices(pauli_matrices, k=num_qubits))
        hamiltonian_terms.append((random.uniform(0, 1), pauli_string)) # The first element is the coefficient
    return hamiltonian_terms


def construct_hamiltonian(hamiltonian_terms: Hamiltonian) -> SparsePauliOp:
    """Constructs the full Hamiltonian as a SparsePauliOp."""
    return SparsePauliOp([term[1] for term in hamiltonian_terms], coeffs=[term[0] for term in hamiltonian_terms])

def get_eigenvalues_vectors(hamiltonian: SparsePauliOp):
    """Computes eigenvalues and eigenvectors of the Hamiltonian."""
    matrix = hamiltonian.to_matrix()  # Convert to a dense matrix
    eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
    return eigenvalues, eigenvectors

# Function to calculate the eigenstate associated with the smallest eigenvalue (NOT optimized)
def calculate_smallest_eigenstate(hamiltonian_terms):
    # Construct the full Hamiltonian matrix
    H = sum(coeff * op.to_matrix() for coeff, op in hamiltonian_terms)
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    # Find the smallest eigenvalue and corresponding eigenstate
    smallest_eigenvalue = eigenvalues[0]
    smallest_eigenstate = eigenvectors[:, 0]
    return smallest_eigenvalue, smallest_eigenstate
