from sane_applications.qft_qpe.algos import qdrift_qpe, generate_ising_hamiltonian, exponentiate_hamiltonian, prepare_eigenstate_circuit, calculate_ground_state_and_energy
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
import math
from random_cool_trinkets.utils import hamiltonian_explicit, hamiltonian_simplified, format_hamiltonian_matrix_as_latex, plot_complex_unit_circle
import uuid
import numpy as np

# Setup for the test environment
LOG_FILE = "qpe_histograms_log.txt"

# Parameters for the Ising model
NUM_QUBITS = 2
J = 1.2
G = 1.0

# Parameter sweep for time
TIME_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]
SHOTS_VALUES = [10000]
ANCILLA_VALUES = [5, 6, 7]

@pytest.mark.parametrize("time", TIME_VALUES)
@pytest.mark.parametrize("shots", SHOTS_VALUES)
@pytest.mark.parametrize("num_ancilla", ANCILLA_VALUES)

def test_qdrift_qpe_ising_hamiltonian_general_case(time, shots, num_ancilla):
    """Test QPE with Ising Hamiltonian (General Case) and log Hamiltonian representations."""
    test_id = str(uuid.uuid4())[:8]  # Unique ID for the test run

    # Generate Ising Hamiltonian
    H = generate_ising_hamiltonian(NUM_QUBITS, J, G)
    matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    first_positive_eigenvalue = min(eigenvalues[eigenvalues > 0])
    eigenvector_index = np.where(eigenvalues == first_positive_eigenvalue)[0][0]
    first_positive_eigenvect = eigenvectors[:, eigenvector_index]

    # Hamiltonian representations to display in the log
    hamiltonian_tensor = hamiltonian_explicit(H)
    hamiltonian_simplified_repr = hamiltonian_simplified(H)
    hamiltonian_matrix = format_hamiltonian_matrix_as_latex(H.to_matrix())

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

    # Save histogram
    filename = f"histogram_{test_id}.png"
    plot_histogram(counts).savefig(filename)

    # Save circuit diagram
    circuit_filename = f"circuit_{test_id}.png"
    qc.draw("mpl").savefig(circuit_filename)

    # Plot the complex unit circle
    unit_circle_filename = f"unit_circle_{test_id}.png"
    plot_complex_unit_circle(estimated_phase * 2*np.pi, expected_phase * 2*np.pi, num_ancilla, time, unit_circle_filename)

    # Log data into the log file
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"qdrift general case,{filename},{time},{shots},{num_ancilla},{most_probable},{estimated_phase},{expected_phase},{estimated_energy},{first_positive_eigenvalue},{hamiltonian_tensor},{hamiltonian_simplified_repr},{hamiltonian_matrix},{expected_bitstring},{test_id}\n")

    # Assert that the most probable bitstring starts with the expected prefix
    assert most_probable.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {most_probable}"