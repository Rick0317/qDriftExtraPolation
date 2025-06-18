import sys
import os
import numpy as np
from testing_things_properly.qft_qpe.algos import standard_qpe, generate_ising_hamiltonian, exponentiate_hamiltonian, prepare_eigenstate_circuit, calculate_ground_state_and_energy #, qdrift_qpe
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from testing_things_properly.qft_qpe.algos import qdrift_qpe
import pytest
import csv
from scripts.algo.chebyshev import chebyshev_nodes


CSV_FILE = "ising_model_sweep_data.csv"
CSV_FILE_QDRIFT = "qdrift_ising_model_sweep_data.csv"  # New CSV file for QDRIFT tests

# Parameters for the Ising model
NUM_QUBITS = 2
J = 1.2
G = 1.0

# Parameter sweep for time
TIME_VALUES = [1]
SHOTS_VALUES = [1]
ANCILLA_VALUES = [7]

@pytest.mark.parametrize("time", TIME_VALUES)
@pytest.mark.parametrize("shots", SHOTS_VALUES)
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES)
def test_qdrift_qpe_extrapolation_ising_hamiltonian_general_case(time, shots, num_ancilla):
    """
    Test qDirft & QPE & Extrapolation algorithm
    Args:
        time:
        shots:
        num_ancilla:

    Returns:

    """
    for n in range(2, 12, 2):  # number of the Chebyshev nodes
        # For each number of Chebyshev nodes candidates,
        # we find the estimated eigenvalue
        data_point = []
        # Storage for data points to be interpolated
        nodes = chebyshev_nodes(n)
        for i in range(n // 2):
            # node corresponds to tau, the time step.
            tau = nodes[i]
            print(f"Index {i}, tau = {tau}")


        #     # Generate Ising Hamiltonian
        #     H = generate_ising_hamiltonian(NUM_QUBITS, J, G)
        #     matrix = H.to_matrix()
        #     eigenvalues, eigenvectors = np.linalg.eig(matrix)
        #     alpha = sum(abs(H.coeffs))
        #     first_positive_eigenvalue = min(eigenvalues[eigenvalues > 0])
        #     eigenvector_index = np.where(eigenvalues == first_positive_eigenvalue)[0][0]
        #     first_positive_eigenvect = eigenvectors[:, eigenvector_index]
        #
        #     # Expected phase calculation
        #     expected_phase = (first_positive_eigenvalue.real * time) / (2 * np.pi) % 1
        #     expected_bitstring = bin(round(expected_phase * (2 ** num_ancilla)))[2:].zfill(num_ancilla)
        #
        #     eigenstate_circuit = prepare_eigenstate_circuit(first_positive_eigenvect)
        #     num_samples = 100
        #
        #     total_counts = {}
        #     for _ in range(num_samples):
        #         qc = qdrift_qpe(H, eigenstate=eigenstate_circuit, time=time, num_qubits=NUM_QUBITS, num_ancilla=num_ancilla, num_samples=num_samples)
        #
        #         # Simulate the circuit
        #         simulator = AerSimulator()
        #         compiled_circuit = transpile(qc, simulator)
        #         result = simulator.run(compiled_circuit, shots=shots).result()
        #         counts = result.get_counts()
        #
        #         # Determine the most probable bitstring
        #         most_probable = max(counts, key=counts.get)
        #         if most_probable in total_counts:
        #             total_counts[most_probable] += 1
        #         else:
        #             total_counts[most_probable] = 1
        #
        #     most_probable = max(total_counts, key=total_counts.get)
        #     estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
        #     estimated_phase = estimated_decimal
        #     estimated_energy = 2 * np.pi * estimated_phase / time
        #
        #
        # # Calculate error
        # eigenvalue_error = np.abs(estimated_energy - first_positive_eigenvalue)
        #
        # # Append directly to CSV
        # header = [
        #     "Num Qubits", "Time", "Shots", "Num Ancilla",
        #     "Exact Eigenvalue", "Expected Phase",
        #     "Most Probable Bitstring", "Estimated Phase",
        #     "Estimated Eigenvalue", "Eigenvalue Error", "Alpha"
        # ]
        # row = [
        #     NUM_QUBITS, time, shots, num_ancilla,
        #     first_positive_eigenvalue, expected_phase,
        #     most_probable, estimated_phase,
        #     estimated_energy, eigenvalue_error, alpha
        # ]
        #
        # file_exists = os.path.isfile(CSV_FILE_QDRIFT)
        # with open(CSV_FILE_QDRIFT, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     if not file_exists:
        #         writer.writerow(header)
        #     writer.writerow(row)
        # # Assert that the most probable bitstring starts with the expected prefix
        # assert most_probable.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {most_probable}"

