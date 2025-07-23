import os
import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import PhaseGate
from scripts.parameter_sweep.algos import generate_ising_hamiltonian, prepare_eigenstate_circuit, qdrift_qpe_extra_random, qdrift_qpe, qdrift_qpe_chat_gpts_take, generate_random_hamiltonian_with_pauli_tensor_structure, deterministic_qpe_qdrift_with_error_budget
from scripts.parameter_sweep.kitaev_qpe import get_kitaev_result, get_qDrift_kitaev_result
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
from algo.chebyshev import chebyshev_nodes
import pandas as pd
import csv
from typing import Callable, Tuple
import inspect
import datetime
# memory_guard
import psutil
import time
import os
import statistics

# Parameters for the Ising model
NUM_QUBITS = 2
J = 1.0
G = 0.8

CSV_FILE_QDRIFT_QPE_ALL = f"qdrift_ising_model_6_nodes_{datetime.datetime.today().strftime('%Y-%m-%d')}.csv"  # New CSV file for QDRIFT tests

QDRIFT_IMPLEMENTATIONS = [(qdrift_qpe, "exponential invocations of qdrift channel")]


chebyshev_nodes12 = np.array(chebyshev_nodes(10))
TIME_VALUES = 0.01 + (0.1 - 0.01) * np.array(chebyshev_nodes12[:5])
HAMILTONIANS = [("Ising", generate_ising_hamiltonian(NUM_QUBITS, 0.5 * J, 0.5 * G))]
NUM_MEASUREMENTS = 1000
NUM_SHOTS = 1024 * 100

@pytest.mark.parametrize("qdrift_impl", QDRIFT_IMPLEMENTATIONS)
@pytest.mark.parametrize("calculate_ground_state", [True])
@pytest.mark.parametrize("H", HAMILTONIANS)
@pytest.mark.parametrize("n_measurements", [NUM_MEASUREMENTS])
@pytest.mark.parametrize("num_shots_per_circuit", [NUM_SHOTS])
def test_kitaev_qpe_general_case(
        qdrift_impl: Tuple[Callable, str],
        calculate_ground_state: bool, H, n_measurements: int,
        num_shots_per_circuit: int):

    type_of_hamiltonian, H = H

    matrix = H.to_matrix()

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    alpha = sum(abs(H.coeffs))

    # compute the exact target eigenvalue and eigenvector
    if calculate_ground_state:
        # Let's do the ground state
        exact_eigenvalue = min(eigenvalues[eigenvalues > 0])
        eigenvector_index = np.where(eigenvalues == exact_eigenvalue)[0][0]
        eigenstate = eigenvectors[:, eigenvector_index]
    else:
        # Let's do the largest eigenvalue
        exact_eigenvalue = max(eigenvalues)
        eigenvector_index = np.where(eigenvalues == exact_eigenvalue)[0][0]
        eigenstate = eigenvectors[:, eigenvector_index]

    # Prepare the eigenstate circuit
    eigenstate_circuit = prepare_eigenstate_circuit(eigenstate)

    # Expected phase calculation
    expected_phase = (exact_eigenvalue.real / (2 * np.pi)) % 1
    print(f"Expected phase: {expected_phase}")
    print(f"Exact eigenvalue: {exact_eigenvalue}")

    m = 5
    estimated_phase = get_kitaev_result(H, eigenstate_circuit, m, shots_per_estimation=num_shots_per_circuit)

    print(f"Estimated phase: {estimated_phase}")
    assert abs(expected_phase - estimated_phase) < 1 / (2 ** (m + 2))



@pytest.mark.parametrize("qdrift_impl", QDRIFT_IMPLEMENTATIONS)
@pytest.mark.parametrize("calculate_ground_state", [True])
@pytest.mark.parametrize("H", HAMILTONIANS)
@pytest.mark.parametrize("n_measurements", [NUM_MEASUREMENTS])
@pytest.mark.parametrize("num_shots_per_circuit", [NUM_SHOTS])
def test_qdrift_kitaev_qpe_general_case(
        qdrift_impl: Tuple[Callable, str],
        calculate_ground_state: bool, H, n_measurements: int,
        num_shots_per_circuit: int):

    type_of_hamiltonian, H = H

    matrix = H.to_matrix()

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    alpha = sum(abs(H.coeffs))

    # compute the exact target eigenvalue and eigenvector
    if calculate_ground_state:
        # Let's do the ground state
        exact_eigenvalue = min(eigenvalues[eigenvalues > 0])
        eigenvector_index = np.where(eigenvalues == exact_eigenvalue)[0][0]
        eigenstate = eigenvectors[:, eigenvector_index]
    else:
        # Let's do the largest eigenvalue
        exact_eigenvalue = max(eigenvalues)
        eigenvector_index = np.where(eigenvalues == exact_eigenvalue)[0][0]
        eigenstate = eigenvectors[:, eigenvector_index]

    # Prepare the eigenstate circuit
    eigenstate_circuit = prepare_eigenstate_circuit(eigenstate)

    # Expected phase calculation
    expected_phase = (exact_eigenvalue.real / (2 * np.pi)) % 1
    print(f"Expected phase: {expected_phase}")
    print(f"Exact eigenvalue: {exact_eigenvalue}")

    m = 9
    estimated_phase = get_qDrift_kitaev_result(H, eigenstate_circuit, m, 10)

    print(f"Estimated phase: {estimated_phase}")
    assert abs(expected_phase - estimated_phase) < 1 / (2 ** (m + 2))
