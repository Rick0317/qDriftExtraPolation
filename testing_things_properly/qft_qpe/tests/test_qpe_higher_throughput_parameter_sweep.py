
import sys
import os
import numpy as np
from qiskit.quantum_info import Operator
from qiskit.circuit.library import PhaseGate
from sane_applications.qft_qpe.algos import standard_qpe, generate_ising_hamiltonian, exponentiate_hamiltonian, prepare_eigenstate_circuit, calculate_ground_state_and_energy, qdrift_qpe
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
import math
import pandas as pd

# Setup for the test environment
LOG_FILE = "qpe_high_throughput_parameter_sweep.csv"

# Sanity check but with varying number of ancilla qubits
ANCILLA_VALUES = [3, 4, 5]
@pytest.mark.parametrize("phase, expected_bin, num_ancilla", [(2 * math.pi * k / 2**n, bin(k)[2:].zfill(n), n) for n in ANCILLA_VALUES for k in range(1, 2**n) ])
def test_generat_qpe_with_parametrized_phase_and_ancilla(phase, expected_bin, num_ancilla):
    """Test QPE with parametrized phase and varying number of ancilla qubits."""
    unitary = Operator(PhaseGate(phase))

    # Prepare the eigenstate circuit
    eigenstate = QuantumCircuit(1)
    eigenstate.x(0)

    # Generate the QPE circuit
    qc = standard_qpe(unitary, eigenstate, num_ancilla)

    # Simulate the circuit
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=1024)
    result = job.result()
    counts = result.get_counts(qc)

    # Determine the most probable bitstring
    most_probable = max(counts, key=counts.get)
    
    # Assert that the most probable bitstring starts with the expected prefix
    assert most_probable.startswith(expected_bin), f"Expected prefix {expected_bin}, got {most_probable}"


# Real deal: QPE to estimate eigenvalues of Ising Hamiltonian

import csv

CSV_FILE = "ising_model_sweep_data.csv"

# Parameters for the Ising model
NUM_QUBITS = 2
J = 1.2
G = 1.0

data_from_ising_model_tests = pd.DataFrame(columns=["Num Qubits", "Exact Eigenvalue", "Expected Phase", "Most Probable Bitstring", "Estimated Phase", "Estimated Eigenvalue", "Eigenvalue Error"])

# Parameter sweep for time
TIME_VALUES = [0.0001 * i for i in range(1, 100001, 20)]  # 0.0001 to 1.0 in steps of 0.0001
SHOTS_VALUES = [10000]
ANCILLA_VALUES = [ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Number of ancilla qubits

@pytest.mark.parametrize("time", TIME_VALUES)
@pytest.mark.parametrize("shots", SHOTS_VALUES)
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES)
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


CSV_FILE = "qdrift_ising_model_sweep_data.csv"  # New CSV file for QDRIFT tests

@pytest.mark.parametrize("time", TIME_VALUES)
@pytest.mark.parametrize("shots", SHOTS_VALUES)
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES)

def test_qdrift_qpe_ising_hamiltonian_general_case(time, shots, num_ancilla):
    """Test QPE with Ising Hamiltonian (General Case) and log Hamiltonian representations."""

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

    eigenstate_circuit = prepare_eigenstate_circuit(first_positive_eigenvect)

    qc = qdrift_qpe(H, eigenstate=eigenstate_circuit, time=time, num_qubits=NUM_QUBITS, num_ancilla=num_ancilla)

    # Simulate the circuit
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


# Export DataFrame to CSV after all tests have completed
@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """
    Hook to export data to CSV after all tests have completed.
    """
    output_file = "ising_model_sweep_data.csv"
    print(f"\nExporting data to {output_file}...")
    data_from_ising_model_tests.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Print the DataFrame to console
    print(data_from_ising_model_tests)
    
    # Save the DataFrame to CSV
    data_from_ising_model_tests.to_csv(LOG_FILE, index=False)
    
    print(f"Data saved to {LOG_FILE}")
