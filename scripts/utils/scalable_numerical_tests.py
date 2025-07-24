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
import datetime
import json
import hashlib


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

def export_run_parameters(
    output_path: str,
    params: Dict[str, Any],
    hamiltonian_coeffs: Optional[np.ndarray] = None,
    hamiltonian_paulis: Optional[list] = None,
    experiment_id: Optional[str] = None
):
    """
    Export the full dictionary of sweep/test parameters to a JSON file.

    Args:
        output_path (str): Filepath to save JSON.
        params (dict): Dictionary of all run parameters.
        hamiltonian_coeffs (np.ndarray, optional): Hamiltonian coefficients.
        hamiltonian_paulis (list, optional): Pauli strings.
        experiment_id (str, optional): Optional experiment group identifier.
    """
    # Try to estimate the seed hash from numpy state
    try:
        rng_state = np.random.get_state()
        seed_bytes = rng_state[1][0].tobytes()
        approximate_seed_hash = hashlib.sha256(seed_bytes).hexdigest()
    except Exception:
        approximate_seed_hash = "Unavailable"

    export_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "parameters": params,
        "hamiltonian_coeffs": hamiltonian_coeffs.tolist() if hamiltonian_coeffs is not None else None,
        "hamiltonian_paulis": hamiltonian_paulis,
        "numpy_rng_seed_sha256": approximate_seed_hash,
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=4)