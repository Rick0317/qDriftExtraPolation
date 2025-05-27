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

# Setup for the tests environment
LOG_FILE = "qpe_histograms_log.txt"
if os.path.exists(LOG_FILE): # Clear log file at start of tests run
    os.remove(LOG_FILE)

# Test function for Quantum Phase Estimation with a simple phase gate
# @pytest.mark.parametrize("phase,expected_bitstring", [
#     (math.pi / 4, "001"),
#     (math.pi / 2, "010"),
#     (3 * math.pi / 4, "011"),
#     (math.pi, "100"),
#     (5 * math.pi / 4, "101"),
#     (3 * math.pi / 2, "110"),
#     (7 * math.pi / 4, "111"),
# ])
# def test_general_qpe_with_parametrized_phase(phase, expected_bitstring):
#     unitary = Operator(PhaseGate(phase))
#
#     eigenstate = QuantumCircuit(1)
#     eigenstate.x(0)
#
#     num_ancilla = 3
#     shots = 1024
#     qc = standard_qpe(unitary, eigenstate, num_ancilla)
#
#     simulator = AerSimulator()
#     job = simulator.run(transpile(qc, simulator), shots=shots)
#     result = job.result()
#     counts = result.get_counts(qc)
#     most_probable = max(counts, key=counts.get)
#     estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
#     estimated_phase = 2 * math.pi * estimated_decimal
#     assert most_probable.startswith(expected_bitstring), f"Expected prefix {expected_bitstring}, got {most_probable}"
#
#     plot_histogram(counts).savefig(f"output_general_phase_{round(phase, 3)}.png")
#     # Generate filename and save histogram
#     filename = f"output_general_phase_{round(phase, 3)}_a{num_ancilla}.png"
#     plot_histogram(counts).savefig(filename)
#
#     # save circuit diagram
#     qc.draw("mpl").savefig(f"circuit_general_phase_{round(phase, 3)}_a{num_ancilla}.png")
#
#     # Log data
#     with open(LOG_FILE, "a") as log_file:
#         log_file.write(f"{filename},{phase},{num_ancilla},{most_probable},{estimated_phase}\n")


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
    g = 0.1  # Transverse field strength
    t = 0.5  # Reduced evolution time for better phase resolution

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

    # Check eigenstate quality
    H_ground_state = H @ ground_state
    eigenvalue_estimate = np.real(np.vdot(ground_state, H_ground_state))
    eigenstate_error = np.linalg.norm(H_ground_state - ground_energy * ground_state)
    print(f"Eigenstate quality check:")
    print(f"Direct eigenvalue estimate: {eigenvalue_estimate:.6f}")
    print(f"Eigenstate error norm: {eigenstate_error:.6f}")

    # Compute the time evolution unitary: exp(-iHt)
    unitary = Operator(expm(-1j * t * H))

    # Verify eigenstate property
    evolved_state = unitary.data @ ground_state
    phase = np.angle(np.vdot(ground_state, evolved_state))
    expected_phase = - (-ground_energy * t) / (2 * np.pi) % 1
    print(f"Phase verification:")
    print(f"Expected phase: {expected_phase:.6f}")
    print(f"Actual phase: {phase:.6f}")
    print(f"Phase error: {abs(phase - expected_phase):.6f}")

    # Create an eigenstate using the ground state
    eigenstate = QuantumCircuit(2)
    eigenstate.initialize(ground_state, [0, 1])

    print(f"Ground state energy: {ground_energy}")

    # Set up QPE parameters
    num_ancilla = 7
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
    print(f"Counts: {counts}")

    for bitstring, count in counts.items():
        # Convert bitstring to decimal and then to phase
        decimal = int(bitstring, 2) / (2 ** num_ancilla)
        phase = decimal
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
    estimated_energy = - 2 * np.pi * weighted_phase / t

    # Save results
    filename = f"output_transverse_ising_J{J}_g{g}_t{t}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(os.path.join(results_dir, filename))

    # Save circuit diagram
    qc.draw("mpl").savefig(
        os.path.join(results_dir, f"circuit_transverse_ising_J{J}_g{g}_t{t}_a{num_ancilla}.png"))

    # Log data with additional diagnostics
    with open(os.path.join(results_dir, "results.log"), "w") as log_file:
        log_file.write(f"Parameters:\n")
        log_file.write(f"J = {J}\n")
        log_file.write(f"g = {g}\n")
        log_file.write(f"t = {t}\n")
        log_file.write(f"num_ancilla = {num_ancilla}\n")
        log_file.write(f"shots = {shots}\n\n")

        log_file.write("Eigenstate quality check:\n")
        log_file.write(f"Direct eigenvalue estimate: {eigenvalue_estimate:.6f}\n")
        log_file.write(f"Eigenstate error norm: {eigenstate_error:.6f}\n\n")

        log_file.write("Phase verification:\n")
        log_file.write(f"Expected phase: {expected_phase:.6f}\n")
        log_file.write(f"Actual phase: {phase:.6f}\n")
        log_file.write(f"Phase error: {abs(phase - expected_phase):.6f}\n\n")

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
             counts=counts,
             eigenstate_error=eigenstate_error,
             phase_error=abs(phase - expected_phase))

    print(f"\nAll results have been saved in the directory: {results_dir}")


def test_general_qpe_with_4site_ising():
    """
    Test QPE on a 4-site Transverse Ising model with periodic boundary conditions.
    The Hamiltonian is: H = -J Σᵢ ZᵢZᵢ₊₁ + g Σᵢ Xᵢ
    where i+1 wraps around to 0 for the last site (periodic boundary).
    """
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"qpe_results_4site_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Parameters for the 4-site Transverse Ising model
    J = 0.2  # Coupling strength
    g = 0.5  # Transverse field strength
    t = 0.1  # Evolution time (reduced for better phase resolution)
    N = 4    # Number of sites

    # Create the Hamiltonian terms
    # First, create all the ZZ terms for nearest neighbors (including periodic boundary)
    H = np.zeros((2**N, 2**N), dtype=complex)

    # Add ZZ terms for nearest neighbors
    for i in range(N):
        # Create the ZZ term for sites i and (i+1) mod N
        pauli_string = ['I'] * N
        pauli_string[i] = 'Z'
        pauli_string[(i + 1) % N] = 'Z'
        H -= J * Pauli(''.join(pauli_string)).to_matrix()

    # Add X terms for each site
    for i in range(N):
        pauli_string = ['I'] * N
        pauli_string[i] = 'X'
        H += g * Pauli(''.join(pauli_string)).to_matrix()

    # Get exact eigenvalues and eigenvectors
    eigvals, eigvecs = eigh(H)  # Hermitian matrix decomposition
    ground_energy = eigvals[0]
    ground_state = eigvecs[:, 0]

    # Check eigenstate quality
    H_ground_state = H @ ground_state
    eigenvalue_estimate = np.real(np.vdot(ground_state, H_ground_state))
    eigenstate_error = np.linalg.norm(H_ground_state - ground_energy * ground_state)
    print(f"Eigenstate quality check:")
    print(f"Direct eigenvalue estimate: {eigenvalue_estimate:.6f}")
    print(f"Eigenstate error norm: {eigenstate_error:.6f}")

    # Compute the time evolution unitary: exp(-iHt)
    unitary = Operator(expm(-1j * t * H))

    # Verify eigenstate property
    evolved_state = unitary.data @ ground_state
    phase = np.angle(np.vdot(ground_state, evolved_state))
    expected_phase = -ground_energy * t
    print(f"Phase verification:")
    print(f"Expected phase: {expected_phase:.6f}")
    print(f"Actual phase: {phase:.6f}")
    print(f"Phase error: {abs(phase - expected_phase):.6f}")

    # Create an eigenstate using the ground state
    eigenstate = QuantumCircuit(N)
    eigenstate.initialize(ground_state, range(N))

    print(f"Ground state energy: {ground_energy}")

    # Set up QPE parameters
    num_ancilla = 8  # Increased number of ancilla qubits for better resolution
    shots = 4096 * 4

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

    # Handle phase wrapping
    if weighted_phase > np.pi:
        weighted_phase -= 2 * np.pi
    elif weighted_phase < -np.pi:
        weighted_phase += 2 * np.pi

    # Convert phase to energy (since phase = -Et)
    estimated_energy = -weighted_phase / t

    # Save results
    filename = f"output_4site_ising_J{J}_g{g}_t{t}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(os.path.join(results_dir, filename))

    # Save circuit diagram
    qc.draw("mpl").savefig(
        os.path.join(results_dir, f"circuit_4site_ising_J{J}_g{g}_t{t}_a{num_ancilla}.png"))

    # Log data with additional diagnostics
    with open(os.path.join(results_dir, "results.log"), "w") as log_file:
        log_file.write(f"Parameters:\n")
        log_file.write(f"N (number of sites) = {N}\n")
        log_file.write(f"J = {J}\n")
        log_file.write(f"g = {g}\n")
        log_file.write(f"t = {t}\n")
        log_file.write(f"num_ancilla = {num_ancilla}\n")
        log_file.write(f"shots = {shots}\n\n")

        log_file.write("Eigenstate quality check:\n")
        log_file.write(f"Direct eigenvalue estimate: {eigenvalue_estimate:.6f}\n")
        log_file.write(f"Eigenstate error norm: {eigenstate_error:.6f}\n\n")

        log_file.write("Phase verification:\n")
        log_file.write(f"Expected phase: {expected_phase:.6f}\n")
        log_file.write(f"Actual phase: {phase:.6f}\n")
        log_file.write(f"Phase error: {abs(phase - expected_phase):.6f}\n\n")

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
    plt.title(f"Phase Distribution (N={N}, J={J}, g={g}, t={t})")
    plt.xlabel("Measurement Index (sorted by probability)")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(results_dir, f"phase_distribution_N{N}_J{J}_g{g}_t{t}_a{num_ancilla}.png"))
    plt.close()

    # Save raw data
    np.savez(os.path.join(results_dir, "raw_data.npz"),
             phases=sorted_phases,
             probabilities=sorted_probs,
             estimated_energy=estimated_energy,
             ground_energy=ground_energy,
             counts=counts,
             eigenstate_error=eigenstate_error,
             phase_error=abs(phase - expected_phase),
             N=N,
             J=J,
             g=g,
             t=t)

    print(f"\nAll results have been saved in the directory: {results_dir}")


def test_general_qpe_with_heisenberg_dm():
    """
    Test QPE on a 4-site Heisenberg model with Dzyaloshinskii-Moriya (DM) interaction.
    The Hamiltonian is:
    H = J Σᵢ (XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₁) + D Σᵢ (XᵢYᵢ₊₁ - YᵢXᵢ₊₁) + h Σᵢ Zᵢ
    where:
    - J is the Heisenberg coupling
    - D is the DM interaction strength
    - h is the external field strength
    - i+1 wraps around to 0 for the last site (periodic boundary)
    """
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"qpe_results_heisenberg_dm_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Parameters for the Heisenberg model with DM interaction
    J = 0.6    # Heisenberg coupling strength
    D = 0.2    # DM interaction strength
    h = 0.1    # External field strength
    t = 0.1    # Evolution time (reduced for better phase resolution)
    N = 4      # Number of sites

    # Create the Hamiltonian terms
    H = np.zeros((2**N, 2**N), dtype=complex)

    # Add Heisenberg terms (XX + YY + ZZ) for nearest neighbors
    for i in range(N):
        # XX term
        pauli_string = ['I'] * N
        pauli_string[i] = 'X'
        pauli_string[(i + 1) % N] = 'X'
        H += J * Pauli(''.join(pauli_string)).to_matrix()

        # YY term
        pauli_string = ['I'] * N
        pauli_string[i] = 'Y'
        pauli_string[(i + 1) % N] = 'Y'
        H += J * Pauli(''.join(pauli_string)).to_matrix()

        # ZZ term
        pauli_string = ['I'] * N
        pauli_string[i] = 'Z'
        pauli_string[(i + 1) % N] = 'Z'
        H += J * Pauli(''.join(pauli_string)).to_matrix()

    # Add DM interaction terms (XY - YX)
    for i in range(N):
        # XY term
        pauli_string = ['I'] * N
        pauli_string[i] = 'X'
        pauli_string[(i + 1) % N] = 'Y'
        H += D * Pauli(''.join(pauli_string)).to_matrix()

        # -YX term
        pauli_string = ['I'] * N
        pauli_string[i] = 'Y'
        pauli_string[(i + 1) % N] = 'X'
        H -= D * Pauli(''.join(pauli_string)).to_matrix()

    # Add external field terms (Z)
    for i in range(N):
        pauli_string = ['I'] * N
        pauli_string[i] = 'Z'
        H += h * Pauli(''.join(pauli_string)).to_matrix()

    # Get exact eigenvalues and eigenvectors
    eigvals, eigvecs = eigh(H)  # Hermitian matrix decomposition
    ground_energy = eigvals[0]
    ground_state = eigvecs[:, 0]

    # Check eigenstate quality
    H_ground_state = H @ ground_state
    eigenvalue_estimate = np.real(np.vdot(ground_state, H_ground_state))
    eigenstate_error = np.linalg.norm(H_ground_state - ground_energy * ground_state)
    print(f"Eigenstate quality check:")
    print(f"Direct eigenvalue estimate: {eigenvalue_estimate:.6f}")
    print(f"Eigenstate error norm: {eigenstate_error:.6f}")

    # Compute the time evolution unitary: exp(-iHt)
    unitary = Operator(expm(-1j * t * H))

    # Verify eigenstate property
    evolved_state = unitary.data @ ground_state
    phase = np.angle(np.vdot(ground_state, evolved_state))
    expected_phase = -ground_energy * t
    print(f"Phase verification:")
    print(f"Expected phase: {expected_phase:.6f}")
    print(f"Actual phase: {phase:.6f}")
    print(f"Phase error: {abs(phase - expected_phase):.6f}")

    # Create an eigenstate using the ground state
    eigenstate = QuantumCircuit(N)
    eigenstate.initialize(ground_state, range(N))

    print(f"Ground state energy: {ground_energy}")

    # Set up QPE parameters
    num_ancilla = 6  # Number of ancilla qubits
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

    # Handle phase wrapping
    if weighted_phase > np.pi:
        weighted_phase -= 2 * np.pi
    elif weighted_phase < -np.pi:
        weighted_phase += 2 * np.pi

    # Convert phase to energy (since phase = -Et)
    estimated_energy = -weighted_phase / t

    # Save results
    filename = f"output_heisenberg_dm_J{J}_D{D}_h{h}_t{t}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(os.path.join(results_dir, filename))

    # Save circuit diagram
    qc.draw("mpl").savefig(
        os.path.join(results_dir, f"circuit_heisenberg_dm_J{J}_D{D}_h{h}_t{t}_a{num_ancilla}.png"))

    # Log data with additional diagnostics
    with open(os.path.join(results_dir, "results.log"), "w") as log_file:
        log_file.write(f"Parameters:\n")
        log_file.write(f"N (number of sites) = {N}\n")
        log_file.write(f"J (Heisenberg coupling) = {J}\n")
        log_file.write(f"D (DM interaction) = {D}\n")
        log_file.write(f"h (external field) = {h}\n")
        log_file.write(f"t = {t}\n")
        log_file.write(f"num_ancilla = {num_ancilla}\n")
        log_file.write(f"shots = {shots}\n\n")

        log_file.write("Eigenstate quality check:\n")
        log_file.write(f"Direct eigenvalue estimate: {eigenvalue_estimate:.6f}\n")
        log_file.write(f"Eigenstate error norm: {eigenstate_error:.6f}\n\n")

        log_file.write("Phase verification:\n")
        log_file.write(f"Expected phase: {expected_phase:.6f}\n")
        log_file.write(f"Actual phase: {phase:.6f}\n")
        log_file.write(f"Phase error: {abs(phase - expected_phase):.6f}\n\n")

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
    plt.title(f"Phase Distribution (N={N}, J={J}, D={D}, h={h}, t={t})")
    plt.xlabel("Measurement Index (sorted by probability)")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(results_dir, f"phase_distribution_N{N}_J{J}_D{D}_h{h}_t{t}_a{num_ancilla}.png"))
    plt.close()

    # Save raw data
    np.savez(os.path.join(results_dir, "raw_data.npz"),
             phases=sorted_phases,
             probabilities=sorted_probs,
             estimated_energy=estimated_energy,
             ground_energy=ground_energy,
             counts=counts,
             eigenstate_error=eigenstate_error,
             phase_error=abs(phase - expected_phase),
             N=N,
             J=J,
             D=D,
             h=h,
             t=t)

    print(f"\nAll results have been saved in the directory: {results_dir}")
