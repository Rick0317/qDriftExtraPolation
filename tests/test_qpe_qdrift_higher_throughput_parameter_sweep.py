import os
import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import PhaseGate
from scripts.parameter_sweep.algos import generate_ising_hamiltonian, prepare_eigenstate_circuit, qdrift_qpe_extra_random, qdrift_qpe, qdrift_qpe_chat_gpts_take, generate_random_hamiltonian_with_pauli_tensor_structure, deterministic_qpe_qdrift_with_error_budget
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

class QPE_Result:
    def __init__(self, time, num_ancilla, counts):
        self.time = time
        self.num_ancilla = num_ancilla
        self.bitstring_counts = counts
        self.phase_counts = None
        self.energies_counts = None
        self.most_likely_bitstring = None
        self.median_energy = None
        self.min_energy = None
        self.max_energy = None
        self.stdev_energy = None
        self.mean_energy = None
        self.estimated_phase = None
        self.estimated_energy = None


    def __str__(self):
        return f"QPE_Result(time={self.time}, num_ancilla={self.num_ancilla}, most_probable={self.most_likely_bitstring}, estimated_phase={self.estimated_phase}, estimated_energy={self.estimated_energy})"

def process_qpe_results(time, num_ancilla, counts) -> QPE_Result:
    # this accidentally implements statistical amplification
    phase_counts_naive = {int(k, 2) / (2 ** num_ancilla): v for k, v in counts.items()}
    phase_counts_corrected = {k if k < 0.5 else k - 1: v for k, v in phase_counts_naive.items()}
    estimated_energies = {2 * np.pi * k / time: v for k, v in phase_counts_corrected.items()}
    # turn into flat list of energies
    energies_flat = np.repeat(list(estimated_energies.keys()), list(estimated_energies.values()))
    phases_flat = np.repeat(list(phase_counts_corrected.keys()), list(phase_counts_corrected.values()))
    # calculate statistics
    qpe_results = QPE_Result(time, num_ancilla, counts)
    qpe_results.phase_counts = phase_counts_corrected
    qpe_results.energies_counts = estimated_energies
    qpe_results.most_likely_bitstring = max(counts, key=counts.get)
    qpe_results.median_energy = statistics.median(energies_flat)
    qpe_results.min_energy = min(energies_flat)
    qpe_results.max_energy = max(energies_flat)
    qpe_results.stdev_energy = statistics.stdev(energies_flat)
    qpe_results.mean_energy = statistics.mean(energies_flat)
    qpe_results.estimated_phase = statistics.median(phases_flat)
    qpe_results.estimated_energy = statistics.median(energies_flat)
    return qpe_results

# Parameters for the Ising model
NUM_QUBITS = 2
J = 1.0
G = 0.8

#Log file for QDRIFT tests
CSV_FILE_QDRIFT_QPE_ALL = f"qdrift_ising_model_6_nodes_{datetime.datetime.today().strftime('%Y-%m-%d')}.csv"  # New CSV file for QDRIFT tests

QDRIFT_IMPLEMENTATIONS = [(qdrift_qpe, "exponential invocations of qdrift channel")]


chebyshev_nodes12 = np.array(chebyshev_nodes(10))
scaled_pos = 0.01 + (0.1 - 0.01) * np.array(chebyshev_nodes12[:5])
TIME_VALUES = scaled_pos # list(np.logspace(-3, 1.5, num=10))
HAMILTONIANS = [("Ising", generate_ising_hamiltonian(NUM_QUBITS, 0.5 * J, 0.5 * G))]
# HAMILTONIANS = [("Simple Z", SparsePauliOp(["Z"*NUM_QUBITS], coeffs=[np.pi / (4 * t)])) for t in TIME_VALUES ]
RANDOMNESS = [(1024, 1)]  # (num_random_circuits, num_shots_per_circuit)
ANCILLA_VALUES = [10]  # Number of ancilla qubits
NUM_SEGMENTS_PER_INVOCATION = [1]
@pytest.mark.parametrize("qdrift_impl", QDRIFT_IMPLEMENTATIONS)
@pytest.mark.parametrize("calculate_ground_state", [False])
@pytest.mark.parametrize("H", HAMILTONIANS)
@pytest.mark.parametrize("num_random_circuits_and_num_shots_per_circuit", RANDOMNESS,
                         ids=lambda p: f"{p[0]}circ_{p[1]}shots")
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES, ids=lambda v: f"qubit{v}")
@pytest.mark.parametrize("total_simulation_time", TIME_VALUES)
@pytest.mark.parametrize("num_segments_per_invocation", NUM_SEGMENTS_PER_INVOCATION, ids=lambda v: f"segments{v}")


def test_qdrift_qpe_general_case(total_simulation_time, num_ancilla, qdrift_impl: Tuple[Callable, str], num_random_circuits_and_num_shots_per_circuit, calculate_ground_state: bool, H, num_segments_per_invocation):
    """Test QPE with Ising Hamiltonian (General Case) and log Hamiltonian representations."""
    # Generate Ising Hamiltonian
    # Jt = np.sqrt((np.pi / (4 * total_simulation_time)) ** 2 - 4 * G ** 2) / 2
    # print(f"Jt: {Jt}")

    type_of_hamiltonian, H = H
    num_random_circuits, num_shots_per_circuit = num_random_circuits_and_num_shots_per_circuit
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
    expected_phase = (exact_eigenvalue.real * total_simulation_time) / (2 * np.pi) % 1
    expected_bitstring = bin(round(expected_phase * (2 ** num_ancilla)))[2:].zfill(num_ancilla)

    generate_qdrift_circuit, impl_name = qdrift_impl
    results = []
    simulator = AerSimulator()
    for rand_circuit in range(num_random_circuits):
        qc = generate_qdrift_circuit(H, eigenstate=eigenstate_circuit, time=total_simulation_time, num_qubits=NUM_QUBITS, num_ancilla=num_ancilla, num_samples_per_channel_invocation=num_segments_per_invocation)
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

    # Determine the most probable bitstring
    qpe_result = process_qpe_results(total_simulation_time, num_ancilla, counts)

    # Calculate error
    eigenvalue_error = np.abs(qpe_result.estimated_energy - exact_eigenvalue)

    # Append directly to CSV
    header = [
        "Num Qubits", "Time", "Shots", "Num Ancilla",
        "Exact Eigenvalue", "Expected Phase",
        "Most Probable Bitstring", "Estimated Phase",
        "Estimated Eigenvalue", "Eigenvalue Error", "Alpha", "QDRIFT Implementation",
        "type of Hamiltonian", "Num Random Circuits", "Num Shots per Circuit", "Circuit Depth",
        "Raw results", "estimnated energies counts", "max energy", "min energy", "stdev energy", "mean energy", "num segments per invocation"
    ]
    row = [
        NUM_QUBITS, total_simulation_time, num_random_circuits * num_shots_per_circuit, num_ancilla,
        exact_eigenvalue, expected_phase,
        qpe_result.most_likely_bitstring, qpe_result.estimated_phase,
        qpe_result.estimated_energy, eigenvalue_error, alpha, impl_name, type_of_hamiltonian,
        num_random_circuits, num_shots_per_circuit, qc.depth() if isinstance(qc, QuantumCircuit) else "N/A",
        str(counts),  # Store raw results as a string
        str(qpe_result.energies_counts), qpe_result.max_energy, qpe_result.min_energy, qpe_result.stdev_energy, qpe_result.mean_energy, num_segments_per_invocation
    ]

    file_exists = os.path.isfile(CSV_FILE_QDRIFT_QPE_ALL)
    with open(CSV_FILE_QDRIFT_QPE_ALL, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    # Assert that the most probable bitstring starts with the expected prefix
    assert qpe_result.most_likely_bitstring.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {qpe_result.most_likely_bitstring}"


'''
def test_qdrift_qpe_extra_random(tot_simulation_time, num_ancilla, num_samples):
    H = generate_ising_hamiltonian(NUM_QUBITS, J, G)
    matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    alpha = sum(abs(H.coeffs))

    # let's do the largest eigenvalue
    largest_eigenvalue = max(eigenvalues)
    eigenvector_index = np.where(eigenvalues == largest_eigenvalue)[0][0]
    largest_eigenvect = eigenvectors[:, eigenvector_index]
    eigenstate_circuit = prepare_eigenstate_circuit(largest_eigenvect)

    # Expected phase calculation
    expected_phase = (largest_eigenvalue.real * tot_simulation_time) / (2 * np.pi) % 1
    expected_bitstring = bin(round(expected_phase * (2 ** num_ancilla)))[2:].zfill(num_ancilla)
    # build the actual quantum circuits

    curr_samples = 0
    circuits = []
    simulator = AerSimulator()
    while curr_samples < num_samples:
        qc = qdrift_qpe_extra_random(hamiltonian=H, eigenstate=eigenstate_circuit, num_qubits=NUM_QUBITS, num_ancilla=num_ancilla, total_simulation_time=tot_simulation_time, num_samples=num_samples)
        # Simulate the circuit
        compiled_circuit = transpile(qc, simulator)
        circuits.append(compiled_circuit)
        curr_samples += 2 ** num_ancilla - 1 # Each circuit gives us 2^num_ancilla samples
    # Aggregate counts across all random circuits
    counts = {}
    for compiled_circuit in circuits:
        result = simulator.run(compiled_circuit, shots=1).result()
        result_counts = result.get_counts()
        for bitstring, count in result_counts.items():
            if bitstring in counts:
                counts[bitstring] += count
            else:
                counts[bitstring] = count
    # Determine the most probable bitstring
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = estimated_decimal
    estimated_energy = 2 * np.pi * estimated_phase / tot_simulation_time
    # Calculate error
    eigenvalue_error = np.abs(estimated_energy - largest_eigenvalue)
    # Append directly to CSV
    header = [
        "Num Qubits", "Time", "Shots", "Num Ancilla",
        "Exact Eigenvalue", "Expected Phase",
        "Most Probable Bitstring", "Estimated Phase",
        "Estimated Eigenvalue", "Eigenvalue Error", "Alpha"
    ]
    row = [
        NUM_QUBITS, tot_simulation_time, num_samples, num_ancilla,
        largest_eigenvalue, expected_phase,
        most_probable, estimated_phase,
        estimated_energy, eigenvalue_error, alpha
    ]
    file_exists = os.path.isfile(CSV_FILE_QDRIFT_EXTRA_RANDOM)
    with open(CSV_FILE_QDRIFT_EXTRA_RANDOM, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    
    assert most_probable.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {most_probable}"
'''
