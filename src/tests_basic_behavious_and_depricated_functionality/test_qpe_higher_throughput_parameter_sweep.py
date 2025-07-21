import sys
import os
import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import PhaseGate
from sane_applications.qft_qpe.algos import standard_qpe, generate_ising_hamiltonian, exponentiate_hamiltonian, prepare_eigenstate_circuit, qdrift_qpe_extra_random, qdrift_qpe, qdrift_qpe_chat_gpts_take
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
import math
import pandas as pd
import csv
from line_profiler import profile
from typing import Callable, Tuple
import inspect
import datetime

CSV_FILE = f"ising_model_sweep_data {datetime.datetime.today()}.csv"
CSV_FILE_QDRIFT = "qdrift_ising_model_sweep_data_extra_random.csv"  # New CSV file for QDRIFT tests

# Parameters for the Ising model
NUM_QUBITS = 2
J = 1.2
G = 1.0

# Parameter sweep for time
TIME_VALUES = [0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002] + [0.002 + 0.001 * i for i in range(1, 100, 5)] + [0.01 * i for i in range(1, 100)]
ANCILLA_VALUES = [5, 10]  # Number of ancilla qubits
@pytest.mark.parametrize("time", TIME_VALUES)
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES)
#@pytest.mark.parametrize("qdrift_impl", QDRIFT_IMPLEMENTATIONS)
@pytest.mark.parametrize("num_samples", [10240, 102400, 1024000])
@pytest.mark.parametrize("tot_simulation_time", range(1, 1000, 10))



    

def test_qpe_ising_hamiltonian_general_case_positive(time, shots, num_ancilla):
    """Test QPE with Ising Hamiltonian (General Case) and log Hamiltonian representations."""
    global data_from_ising_model_tests
    # Generate Ising Hamiltonian
    H = generate_ising_hamiltonian(NUM_QUBITS, J, G)
    matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    first_positive_eigenvalue = min(eigenvalues[eigenvalues > 0])
    eigenvector_index = np.where(eigenvalues == first_positive_eigenvalue)[0][0]
    first_positive_eigenvect = eigenvectors[:, eigenvector_index]

    # Expected phase calculation
    expected_phase = (first_positive_eigenvalue.real * time) / (2 * np.pi) % 1
    expected_bitstring = bin(round(expected_phase * (2 ** num_ancilla)))[2:].zfill(num_ancilla)
    

    # Exponentiate the Hamiltonian
    U = exponentiate_hamiltonian(H, time)

    # Construct QPE circuit
    eigenstate_circuit = prepare_eigenstate_circuit(first_positive_eigenvect)
    qc = standard_qpe(U, eigenstate_circuit, num_ancilla)
    print("Depth:", qc.depth())
    print("Size:", qc.size())
    print("Width:", qc.width())


    # Simulate the circuit
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()

    # Determine the most probable bitstring
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = estimated_decimal
    estimated_energy = 2 * np.pi * estimated_phase / time

    # Calculate error
    eigenvalue_error = np.abs(estimated_energy - first_positive_eigenvalue)

     # Append directly to CSV
    header = [
        "Num Qubits", "Time", "Shots", "Num Ancilla",
        "Exact Eigenvalue", "Expected Phase",
        "Most Probable Bitstring", "Estimated Phase",
        "Estimated Eigenvalue", "Eigenvalue Error"
    ]
    row = [
        NUM_QUBITS, time, shots, num_ancilla,
        first_positive_eigenvalue, expected_phase,
        most_probable, estimated_phase,
        estimated_energy, eigenvalue_error
    ]

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


    # Assert that the most probable bitstring starts with the expected prefix
    assert most_probable.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {most_probable}"

