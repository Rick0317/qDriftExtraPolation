import os
import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import PhaseGate
from scripts.parameter_sweep.algos import generate_ising_hamiltonian, prepare_eigenstate_circuit, qdrift_qpe_extra_random, qdrift_qpe, generate_random_hamiltonian_with_pauli_tensor_structure
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
import pandas as pd
import csv
from typing import Callable, Tuple
import inspect
import datetime
# memory_guard
import psutil
import time
import os
import gc

def log_memory_usage(note=""):
    """Logs memory usage with an optional note."""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    print(f"[MEMORY] {datetime.datetime.now()} | {note} | RSS: {mem_gb:.2f} GB")


def process_qpe_results(time, num_ancilla, counts):
    # this accidentally implements statistical amplification
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = estimated_decimal
    eigenval_is_negative = estimated_phase >= 0.5
    if eigenval_is_negative:
        estimated_phase = estimated_phase - 1
    estimated_energy = 2 * np.pi * estimated_phase / time
    return most_probable,estimated_phase,estimated_energy


# Parameters for the Ising model
NUM_QUBITS = 2
J = 1
G = 0.8

#Log file for QDRIFT tests
CSV_FILE_QDRIFT_QPE_ALL = f"qdrift_ising_model_sweep_ultimate_2025-07-02.csv"  # New CSV file for QDRIFT tests

QDRIFT_IMPLEMENTATIONS = [(qdrift_qpe, "exponential invocations of qdrift channel")]
# HAMILTONIANS = [("Ising", generate_ising_hamiltonian(NUM_QUBITS, J, G)), ("Simple Z",     SparsePauliOp(["Z"*NUM_QUBITS], coeffs=[1.0]))]
HAMILTONIANS = [("Ising", generate_ising_hamiltonian(NUM_QUBITS, J, G))]
RANDOMNESS = [(1000, 1)]  # (num_random_circuits, num_shots_per_circuit)
ANCILLA_VALUES = [10, 12, 14, 16]  # Number of ancilla qubits
TIME_VALUES = list(np.logspace(-4, 1, num=100))
NUM_QDRIFT_SAMPLES_PER_CHANNEL_INVOCATION = [1, 10, 100]

@pytest.mark.parametrize("qdrift_impl", QDRIFT_IMPLEMENTATIONS)
@pytest.mark.parametrize("calculate_ground_state", [False])
@pytest.mark.parametrize("H", HAMILTONIANS)
@pytest.mark.parametrize("num_random_circuits_and_num_shots_per_circuit", RANDOMNESS,
                         ids=lambda p: f"{p[0]}circ_{p[1]}shots")
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES, ids=lambda v: f"qubit{v}")
@pytest.mark.parametrize("total_simulation_time", TIME_VALUES)
@pytest.mark.parametrize("N", NUM_QDRIFT_SAMPLES_PER_CHANNEL_INVOCATION)


def test_qdrift_qpe_general_case(total_simulation_time, num_ancilla, qdrift_impl: Tuple[Callable, str], num_random_circuits_and_num_shots_per_circuit, calculate_ground_state: bool, H: SparsePauliOp, N):
    """Test QPE with Ising Hamiltonian (General Case) and log Hamiltonian representations."""
    # Generate Ising Hamiltonian
    log_memory_usage("Start of test case")
    type_of_hamiltonian, H = H
    num_random_circuits, num_shots_per_circuit = num_random_circuits_and_num_shots_per_circuit
    matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    alpha = sum(abs(H.coeffs))

    # compute the exact target eigenvalue and eigenvector
    if calculate_ground_state:
        # Let's do the ground state
        first_positive_eigenvalue = min(eigenvalues[eigenvalues > 0])
        eigenvector_index = np.where(eigenvalues == first_positive_eigenvalue)[0][0]
        eigenstate = eigenvectors[:, eigenvector_index]
    else:
        # Let's do the largest eigenvalue
        first_positive_eigenvalue = max(eigenvalues)
        eigenvector_index = np.where(eigenvalues == first_positive_eigenvalue)[0][0]
        eigenstate = eigenvectors[:, eigenvector_index]

    # Prepare the eigenstate circuit
    eigenstate_circuit = prepare_eigenstate_circuit(eigenstate)

    # Expected phase calculation
    expected_phase = (first_positive_eigenvalue.real * total_simulation_time) / (2 * np.pi) % 1
    expected_bitstring = bin(round(expected_phase * (2 ** num_ancilla)))[2:].zfill(num_ancilla)

    generate_qdrift_circuit, impl_name = qdrift_impl
    results = []
    simulator = AerSimulator()
    log_memory_usage("Before simulation run")
    for rand_circuit in range(num_random_circuits):
        qc = generate_qdrift_circuit(H, eigenstate=eigenstate_circuit, time=total_simulation_time, num_qubits=NUM_QUBITS, num_ancilla=num_ancilla, num_samples_per_channel_invocation=N)
        # Simulate the circuit
        compiled_circuit = transpile(qc, simulator)
        result = simulator.run(compiled_circuit, shots=num_shots_per_circuit).result()
        results.append(result)
    # Aggregate counts across all random circuits
    counts = {}
    for result in results:
        result_counts = result.get_counts()
        for bitstring, count in result_counts.items():
            if bitstring in counts:
                counts[bitstring] += count
            else:
                counts[bitstring] = count

    log_memory_usage("After simulation run")
    # Determine the most probable bitstring
    most_probable, estimated_phase, estimated_energy = process_qpe_results(total_simulation_time, num_ancilla, counts)

    # Calculate error
    eigenvalue_error = np.abs(estimated_energy - first_positive_eigenvalue)

    # Append directly to CSV
    header = [
        "Num Qubits", "Time", "Shots", "Num Ancilla",
        "Exact Eigenvalue", "Expected Phase",
        "Most Probable Bitstring", "Estimated Phase",
        "Estimated Eigenvalue", "Eigenvalue Error", "Alpha", "QDRIFT Implementation",
        "type of Hamiltonian", "Num Random Circuits", "Num Shots per Circuit", "Circuit Depth", "qDRIFT samples per invocation of qDRIFT channel"
        "Raw results"
    ]
    row = [
        NUM_QUBITS, total_simulation_time, num_random_circuits * num_shots_per_circuit, num_ancilla,
        first_positive_eigenvalue, expected_phase,
        most_probable, estimated_phase,
        estimated_energy, eigenvalue_error, alpha, impl_name, type_of_hamiltonian,
        num_random_circuits, num_shots_per_circuit, qc.depth() if isinstance(qc, QuantumCircuit) else "N/A", N,
        str(counts)  # Store raw results as a string
    ]

    file_exists = os.path.isfile(CSV_FILE_QDRIFT_QPE_ALL)
    with open(CSV_FILE_QDRIFT_QPE_ALL, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    gc.collect()
    log_memory_usage("After gc.collect()")

    # Assert that the most probable bitstring starts with the expected prefix
    assert most_probable.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {most_probable}"

