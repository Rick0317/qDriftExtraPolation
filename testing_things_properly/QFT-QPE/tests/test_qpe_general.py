import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from qiskit.quantum_info import Operator, Pauli
from qiskit.circuit.library import PhaseGate
from scripts.algo import standard_qpe
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
import math
from scipy.linalg import expm, eigh
import matplotlib.pyplot as plt
from datetime import datetime

# Setup for the test environment
LOG_FILE = "qpe_histograms_log.txt"
if os.path.exists(LOG_FILE): # Clear log file at start of test run
    os.remove(LOG_FILE)

# Test function for Quantum Phase Estimation with a simple phase gate
@pytest.mark.parametrize("phase, expected_bin", [
    (math.pi / 4, '001'),
    (math.pi / 2,  '010'),
    (3 * math.pi / 4, '011'),
    (math.pi,   '100'),
    (5 * math.pi / 4, '101'),
    (3 * math.pi / 2,  '110'),
    (7 * math.pi / 4, '111'),
])
def test_general_qpe_with_parametrized_phase(phase, expected_bin):
    unitary = Operator(PhaseGate(phase))

    eigenstate = QuantumCircuit(1)
    eigenstate.x(0)

    num_ancilla = 3
    shots = 1024
    qc = standard_qpe(unitary, eigenstate, num_ancilla)

    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = 2 * math.pi * estimated_decimal
    assert most_probable.startswith(expected_bin), f"Expected prefix {expected_bin}, got {most_probable}"

    plot_histogram(counts).savefig(f"output_general_phase_{round(phase, 3)}.png")
    # Generate filename and save histogram
    filename = f"output_general_phase_{round(phase, 3)}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(filename)

    # save circuit diagram
    qc.draw("mpl").savefig(f"circuit_general_phase_{round(phase, 3)}_a{num_ancilla}.png")

    # Log data
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{filename},{phase},{num_ancilla},{most_probable},{estimated_phase}\n")


def test_general_qpe_with_n_fold_tensor_prod_of_paulis():
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"qpe_results_n_paulis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Parameters
    N = 4  # Number of qubits/Pauli operators
    t = 1.0  # Evolution time

    # Create a random product of N Pauli operators
    pauli_choices = ['X', 'Y', 'Z']
    pauli_string = ''.join(np.random.choice(pauli_choices) for _ in range(N))
    print(f"Generated Pauli string: {pauli_string}")

    # Create the Hamiltonian as a single Pauli operator
    H = Pauli(pauli_string).to_matrix()

    # Get exact eigenvalues and eigenvectors
    eigvals, eigvecs = eigh(H)  # Hermitian matrix decomposition
    ground_energy = eigvals[0]
    ground_state = eigvecs[:, 0]

    # Compute the time evolution unitary: exp(-iHt)
    unitary = Operator(expm(-1j * t * H))

    # Create an eigenstate using the ground state
    eigenstate = QuantumCircuit(N)
    eigenstate.initialize(ground_state, range(N))

    print(f"Ground state energy: {ground_energy}")

    # Set up QPE parameters
    num_ancilla = 6
    shots = 4096

    # Run QPE
    qc = standard_qpe(unitary, eigenstate, num_ancilla)

    # Execute the circuit
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    # Analyze the full distribution
    total_shots = sum(counts.values())
    phase_estimates = []
    probabilities = []

    for bitstring, count in counts.items():
        # Convert bitstring to decimal and then to phase
        decimal = int(bitstring, 2) / (2 ** num_ancilla)
        phase = 2 * math.pi * decimal
        probability = count / total_shots

        phase_estimates.append(phase)
        probabilities.append(probability)

    # Sort by probability for better visualization
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_phases = np.array(phase_estimates)[sorted_indices]
    sorted_probs = np.array(probabilities)[sorted_indices]

    # Calculate weighted average of phases
    weighted_phase = np.sum(sorted_phases * sorted_probs)

    # Convert phase to energy (since phase = -Et)
    estimated_energy = -weighted_phase / t

    # Save results
    filename = f"output_n_paulis_N{N}_t{t}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(os.path.join(results_dir, filename))

    # Save circuit diagram
    qc.draw("mpl").savefig(
        os.path.join(results_dir, f"circuit_n_paulis_N{N}_t{t}_a{num_ancilla}.png"))

    # Log data
    with open(os.path.join(results_dir, "results.log"), "w") as log_file:
        log_file.write(f"Parameters:\n")
        log_file.write(f"N (number of qubits) = {N}\n")
        log_file.write(f"Pauli string = {pauli_string}\n")
        log_file.write(f"t = {t}\n")
        log_file.write(f"num_ancilla = {num_ancilla}\n")
        log_file.write(f"shots = {shots}\n\n")

        log_file.write("Top 5 most probable measurements:\n")
        for i in range(min(5, len(sorted_phases))):
            log_file.write(f"Phase: {sorted_phases[i]:.4f}, Probability: {sorted_probs[i]:.4f}\n")

        log_file.write(f"\nWeighted average phase: {weighted_phase:.4f}\n")
        log_file.write(f"Estimated energy: {estimated_energy:.4f}\n")
        log_file.write(f"Exact ground state energy: {ground_energy:.4f}\n")
        log_file.write(f"Absolute error: {abs(estimated_energy - ground_energy):.4f}\n")

    # Also append to the main log file
    with open(LOG_FILE, "a") as log_file:
        log_file.write(
            f"{filename},{estimated_energy},{num_ancilla},{ground_energy},{abs(estimated_energy - ground_energy)}\n")

    # Print detailed results
    print("\nTop 5 most probable measurements:")
    for i in range(min(5, len(sorted_phases))):
        print(f"Phase: {sorted_phases[i]:.4f}, Probability: {sorted_probs[i]:.4f}")

    print(f"\nWeighted average phase: {weighted_phase:.4f}")
    print(f"Estimated energy: {estimated_energy:.4f}")
    print(f"Exact ground state energy: {ground_energy:.4f}")
    print(f"Absolute error: {abs(estimated_energy - ground_energy):.4f}")

    # Create a bar plot of the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_probs)), sorted_probs)
    plt.title(f"Phase Distribution (N={N}, t={t})")
    plt.xlabel("Measurement Index (sorted by probability)")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(results_dir, f"phase_distribution_N{N}_t{t}_a{num_ancilla}.png"))
    plt.close()

    # Save raw data
    np.savez(os.path.join(results_dir, "raw_data.npz"),
             phases=sorted_phases,
             probabilities=sorted_probs,
             estimated_energy=estimated_energy,
             ground_energy=ground_energy,
             counts=counts,
             pauli_string=pauli_string)

    print(f"\nAll results have been saved in the directory: {results_dir}")


def test_general_qpe_with_known_hamiltonian():
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"qpe_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Parameters for the Transverse Ising model
    J = 0.6  # Coupling strength
    g = 0.5  # Transverse field strength
    t = 1.0  # Evolution time

    # Create the Hamiltonian terms
    z1z2 = Pauli('ZZ').to_matrix()
    x1 = Pauli('XI').to_matrix()
    x2 = Pauli('IX').to_matrix()

    # Construct the full Hamiltonian: -J Z1Z2 + gX1 + gX2
    H = -J * z1z2 + g * x1 + g * x2

    # Get exact eigenvalues and eigenvectors
    eigvals, eigvecs = eigh(H)  # Hermitian matrix decomposition
    ground_energy = eigvals[0]
    ground_state = eigvecs[:, 0]

    # Compute the time evolution unitary: exp(-iHt)
    unitary = Operator(expm(-1j * t * H))

    # Create an eigenstate using the ground state
    eigenstate = QuantumCircuit(2)
    eigenstate.initialize(ground_state, [0, 1])

    print(f"Ground state energy: {ground_energy}")

    # Set up QPE parameters
    num_ancilla = 6
    shots = 4096 * 10

    # Run QPE
    qc = standard_qpe(unitary, eigenstate, num_ancilla)

    # Execute the circuit
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    # Analyze the full distribution
    total_shots = sum(counts.values())
    phase_estimates = []
    probabilities = []

    for bitstring, count in counts.items():
        # Convert bitstring to decimal and then to phase
        decimal = int(bitstring, 2) / (2 ** num_ancilla)
        phase = 2 * math.pi * decimal
        probability = count / total_shots

        phase_estimates.append(phase)
        probabilities.append(probability)

    # Sort by probability for better visualization
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_phases = np.array(phase_estimates)[sorted_indices]
    sorted_probs = np.array(probabilities)[sorted_indices]

    # Calculate weighted average of phases
    weighted_phase = np.sum(sorted_phases * sorted_probs)

    # Convert phase to energy (since phase = -Et)
    estimated_energy = -weighted_phase / t

    # Save results
    filename = f"output_transverse_ising_J{J}_g{g}_t{t}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(os.path.join(results_dir, filename))

    # Save circuit diagram
    qc.draw("mpl").savefig(
        os.path.join(results_dir, f"circuit_transverse_ising_J{J}_g{g}_t{t}_a{num_ancilla}.png"))

    # Log data
    with open(os.path.join(results_dir, "results.log"), "w") as log_file:
        log_file.write(f"Parameters:\n")
        log_file.write(f"J = {J}\n")
        log_file.write(f"g = {g}\n")
        log_file.write(f"t = {t}\n")
        log_file.write(f"num_ancilla = {num_ancilla}\n")
        log_file.write(f"shots = {shots}\n\n")

        log_file.write("Top 5 most probable measurements:\n")
        for i in range(min(5, len(sorted_phases))):
            log_file.write(f"Phase: {sorted_phases[i]:.4f}, Probability: {sorted_probs[i]:.4f}\n")

        log_file.write(f"\nWeighted average phase: {weighted_phase:.4f}\n")
        log_file.write(f"Estimated energy: {estimated_energy:.4f}\n")
        log_file.write(f"Exact ground state energy: {ground_energy:.4f}\n")
        log_file.write(f"Absolute error: {abs(estimated_energy - ground_energy):.4f}\n")

    # Also append to the main log file
    with open(LOG_FILE, "a") as log_file:
        log_file.write(
            f"{filename},{estimated_energy},{num_ancilla},{ground_energy},{abs(estimated_energy - ground_energy)}\n")

    # Print detailed results
    print("\nTop 5 most probable measurements:")
    for i in range(min(5, len(sorted_phases))):
        print(f"Phase: {sorted_phases[i]:.4f}, Probability: {sorted_probs[i]:.4f}")

    print(f"\nWeighted average phase: {weighted_phase:.4f}")
    print(f"Estimated energy: {estimated_energy:.4f}")
    print(f"Exact ground state energy: {ground_energy:.4f}")
    print(f"Absolute error: {abs(estimated_energy - ground_energy):.4f}")

    # Create a bar plot of the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_probs)), sorted_probs)
    plt.title(f"Phase Distribution (J={J}, g={g}, t={t})")
    plt.xlabel("Measurement Index (sorted by probability)")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(results_dir, f"phase_distribution_J{J}_g{g}_t{t}_a{num_ancilla}.png"))
    plt.close()

    # Save raw data
    np.savez(os.path.join(results_dir, "raw_data.npz"),
             phases=sorted_phases,
             probabilities=sorted_probs,
             estimated_energy=estimated_energy,
             ground_energy=ground_energy,
             counts=counts)

    print(f"\nAll results have been saved in the directory: {results_dir}")
