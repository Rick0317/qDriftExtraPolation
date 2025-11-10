import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
import pickle
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import PhaseGate
from scripts.parameter_sweep.algos import generate_ising_hamiltonian, prepare_eigenstate_circuit, qdrift_qpe_extra_random, qdrift_qpe, qdrift_qpe_chat_gpts_take, generate_random_hamiltonian_with_pauli_tensor_structure, deterministic_qpe_qdrift_with_error_budget

# Import OpenFermion for H4 conversion
try:
    from openfermion import jordan_wigner
    OPENFERMION_AVAILABLE = True
except ImportError:
    OPENFERMION_AVAILABLE = False
    print("Warning: OpenFermion not available. H4 Hamiltonian will be skipped.")
from scripts.parameter_sweep.kitaev_qpe import get_kitaev_result, get_qDrift_kitaev_result, get_qDrift_kitaev_result_v2
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


def load_h4_hamiltonian() -> SparsePauliOp:
    """
    Load H4 STO-3G Hamiltonian from pickle file and convert to Qiskit SparsePauliOp.

    Returns:
        SparsePauliOp: The H4 Hamiltonian in Qiskit format
    """
    if not OPENFERMION_AVAILABLE:
        raise ImportError("OpenFermion is required to load H4 Hamiltonian")

    # Load the H4 STO-3G Hamiltonian
    h4_file_path = os.path.join(os.path.dirname(__file__), '..', 'ham_lib', 'h4_sto-3g.pkl')
    with open(h4_file_path, 'rb') as f:
        h4_fermion_op = pickle.load(f)

    # Convert to qubit operator using Jordan-Wigner transformation
    h4_qubit_op = jordan_wigner(h4_fermion_op)

    # Find the maximum number of qubits needed
    max_qubit = 0
    for pauli_term, coeff in h4_qubit_op.terms.items():
        if pauli_term:  # If not empty tuple (identity)
            max_qubit = max(max_qubit, max([qubit for qubit, _ in pauli_term]))

    num_qubits = max_qubit + 1

    # Convert to Qiskit SparsePauliOp
    pauli_strings = []
    coefficients = []

    for pauli_term, coeff in h4_qubit_op.terms.items():
        if abs(coeff) > 1e-12 and pauli_term != ():  # Filter out very small coefficients
            # Initialize with identity for all qubits
            pauli_list = ['I'] * num_qubits

            # Set Pauli operators
            for qubit, pauli_op in pauli_term:
                if pauli_op == 'X':
                    pauli_list[qubit] = 'X'
                elif pauli_op == 'Y':
                    pauli_list[qubit] = 'Y'
                elif pauli_op == 'Z':
                    pauli_list[qubit] = 'Z'

            pauli_str = ''.join(pauli_list)
            pauli_strings.append(pauli_str)
            coefficients.append(complex(coeff).real)  # Take real part

    # Create SparsePauliOp
    h4_sparse_pauli = SparsePauliOp(pauli_strings, coeffs=coefficients)

    return h4_sparse_pauli

# Parameters for the Ising model
NUM_QUBITS = 2
J = 0.6
G = 0.4

CSV_FILE_QDRIFT_QPE_ALL = f"qdrift_ising_model_6_nodes_{datetime.datetime.today().strftime('%Y-%m-%d')}.csv"  # New CSV file for QDRIFT tests

QDRIFT_IMPLEMENTATIONS = [(qdrift_qpe, "exponential invocations of qdrift channel")]


chebyshev_nodes12 = np.array(chebyshev_nodes(10))
TIME_VALUES = 0.01 + (0.1 - 0.01) * np.array(chebyshev_nodes12[:5])
# Create base Hamiltonians
BASE_HAMILTONIANS = [
    ("Ising_1", generate_ising_hamiltonian(NUM_QUBITS, 0.5 * J, 0.5 * G)),
    # ("Ising_2", generate_ising_hamiltonian(NUM_QUBITS, 0.8 * J, 0.3 * G)),
    # ("Ising_3", generate_ising_hamiltonian(NUM_QUBITS, 0.2 * J, 0.7 * G)),
    # ("Random_1", generate_random_hamiltonian_with_pauli_tensor_structure(NUM_QUBITS, 4)),
    # ("Random_2", generate_random_hamiltonian_with_pauli_tensor_structure(NUM_QUBITS, 6)),
    # ("Simple_Z", SparsePauliOp(["Z"*NUM_QUBITS], coeffs=[1.0]))
]

# Add H4 Hamiltonian if OpenFermion is available
HAMILTONIANS = BASE_HAMILTONIANS.copy()
# if OPENFERMION_AVAILABLE:
#     try:
#         h4_hamiltonian = load_h4_hamiltonian()
#         HAMILTONIANS.append(("H4_STO-3G", h4_hamiltonian))
#         print(f"Added H4 STO-3G Hamiltonian: {h4_hamiltonian.num_qubits} qubits, {len(h4_hamiltonian)} terms")
#     except Exception as e:
#         print(f"Warning: Could not load H4 Hamiltonian: {e}")
# else:
#     print("OpenFermion not available - skipping H4 Hamiltonian")
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

    m = 6
    estimated_phase = get_kitaev_result(H, eigenstate_circuit, m, shots_per_estimation=num_shots_per_circuit)

    print(f"Estimated phase: {estimated_phase}")
    assert abs(expected_phase - estimated_phase) < 1 / (2 ** (m + 2))



@pytest.mark.parametrize("qdrift_impl", QDRIFT_IMPLEMENTATIONS)
@pytest.mark.parametrize("calculate_ground_state", [False])
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

    m = 5
    estimated_phase = get_qDrift_kitaev_result_v2(H, eigenstate_circuit, m, 100)

    print(f"Estimated phase: {estimated_phase}")
    assert abs(expected_phase - estimated_phase) < 1 / (2 ** (m + 2))


@pytest.mark.parametrize("qdrift_impl", QDRIFT_IMPLEMENTATIONS)
@pytest.mark.parametrize("calculate_ground_state", [False])
@pytest.mark.parametrize("H", HAMILTONIANS)
@pytest.mark.parametrize("num_runs", [10])  # Number of times to run the test
def test_qdrift_kitaev_qpe_variance_analysis(
        qdrift_impl: Tuple[Callable, str],
        calculate_ground_state: bool, H, num_runs: int):
    """
    Run qDrift Kitaev QPE multiple times to analyze variance and statistics
    of phase and eigenvalue estimates.

    Args:
        qdrift_impl: The qDrift implementation to use
        calculate_ground_state: Whether to estimate ground state or largest eigenvalue
        H: Tuple of (hamiltonian_name, hamiltonian)
        num_runs: Number of independent runs to perform
    """
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
    expected_eigenvalue = exact_eigenvalue.real

    print(f"\n{'='*60}")
    print(f"Hamiltonian: {type_of_hamiltonian}")
    print(f"Expected phase: {expected_phase:.6f}")
    print(f"Exact eigenvalue: {expected_eigenvalue:.6f}")
    print(f"{'='*60}")

    m = 5
    qDrift_invocation = 100

    # Run the test multiple times to collect statistics
    estimated_phases = []
    estimated_eigenvalues = []
    phase_errors = []
    eigenvalue_errors = []

    print(f"\nRunning {num_runs} independent trials...")
    for run in range(num_runs):
        estimated_phase = get_qDrift_kitaev_result_v2(H, eigenstate_circuit, m, qDrift_invocation)
        estimated_eigenvalue = estimated_phase * 2 * np.pi

        phase_error = abs(expected_phase - estimated_phase)
        eigenvalue_error = abs(expected_eigenvalue - estimated_eigenvalue)

        estimated_phases.append(estimated_phase)
        estimated_eigenvalues.append(estimated_eigenvalue)
        phase_errors.append(phase_error)
        eigenvalue_errors.append(eigenvalue_error)

        print(f"  Run {run+1:2d}: phase={estimated_phase:.6f}, "
              f"eigenvalue={estimated_eigenvalue:.6f}, "
              f"phase_err={phase_error:.6f}, "
              f"eigenvalue_err={eigenvalue_error:.6f}")

    # Calculate statistics
    print(f"\n{'='*60}")
    print("STATISTICS SUMMARY")
    print(f"{'='*60}")

    print("\nPhase Estimates:")
    print(f"  Mean:     {np.mean(estimated_phases):.6f}")
    print(f"  Std Dev:  {np.std(estimated_phases):.6f}")
    print(f"  Variance: {np.var(estimated_phases):.6f}")
    print(f"  Min:      {np.min(estimated_phases):.6f}")
    print(f"  Max:      {np.max(estimated_phases):.6f}")
    print(f"  Median:   {np.median(estimated_phases):.6f}")

    print("\nEigenvalue Estimates:")
    print(f"  Mean:     {np.mean(estimated_eigenvalues):.6f}")
    print(f"  Std Dev:  {np.std(estimated_eigenvalues):.6f}")
    print(f"  Variance: {np.var(estimated_eigenvalues):.6f}")
    print(f"  Min:      {np.min(estimated_eigenvalues):.6f}")
    print(f"  Max:      {np.max(estimated_eigenvalues):.6f}")
    print(f"  Median:   {np.median(estimated_eigenvalues):.6f}")

    print("\nPhase Errors:")
    print(f"  Mean:     {np.mean(phase_errors):.6f}")
    print(f"  Std Dev:  {np.std(phase_errors):.6f}")
    print(f"  Min:      {np.min(phase_errors):.6f}")
    print(f"  Max:      {np.max(phase_errors):.6f}")

    print("\nEigenvalue Errors:")
    print(f"  Mean:     {np.mean(eigenvalue_errors):.6f}")
    print(f"  Std Dev:  {np.std(eigenvalue_errors):.6f}")
    print(f"  Min:      {np.min(eigenvalue_errors):.6f}")
    print(f"  Max:      {np.max(eigenvalue_errors):.6f}")

    theoretical_phase_error_bound = 1 / (2 ** (m + 2))
    theoretical_eigenvalue_error_bound = theoretical_phase_error_bound * 2 * np.pi

    print(f"\nTheoretical Bounds (m={m}):")
    print(f"  Phase error bound:      {theoretical_phase_error_bound:.6f}")
    print(f"  Eigenvalue error bound: {theoretical_eigenvalue_error_bound:.6f}")

    num_within_phase_bound = sum(1 for err in phase_errors if err <= theoretical_phase_error_bound)
    num_within_eigenvalue_bound = sum(1 for err in eigenvalue_errors if err <= theoretical_eigenvalue_error_bound)

    print(f"\nSuccess Rates:")
    print(f"  Within phase bound:      {num_within_phase_bound}/{num_runs} ({100*num_within_phase_bound/num_runs:.1f}%)")
    print(f"  Within eigenvalue bound: {num_within_eigenvalue_bound}/{num_runs} ({100*num_within_eigenvalue_bound/num_runs:.1f}%)")
    print(f"{'='*60}\n")

    # Assert that the mean error is within bounds (more lenient than requiring all runs to pass)
    mean_phase_error = np.mean(phase_errors)
    assert mean_phase_error < 1 / (2 ** (m + 2)), \
        f"Mean phase error {mean_phase_error:.6f} exceeds theoretical bound {theoretical_phase_error_bound:.6f}"


def test_qdrift_kitaev_qpe_multiple_hamiltonians_with_plot():
    """
    Test qDrift Kitaev QPE on multiple Hamiltonians with different qDrift_invocation values
    and create a plot showing the eigenvalue estimation errors.
    """
    # Test parameters
    m = 8  # Number of bits for phase estimation
    qDrift_invocations = [200]  # Different numbers of qDrift samples
    calculate_ground_state = False

    # Theoretical precision bound for Kitaev QPE
    theoretical_phase_error_bound = 1 / (2 ** (m + 2))
    theoretical_eigenvalue_error_bound = theoretical_phase_error_bound * 2 * np.pi

    # Storage for results
    results = []

    print("Testing qDrift Kitaev QPE on multiple Hamiltonians...")
    print(f"Theoretical phase error bound (m={m}): {theoretical_phase_error_bound:.6f}")
    print(f"Theoretical eigenvalue error bound: {theoretical_eigenvalue_error_bound:.6f}")

    for hamiltonian_name, H in HAMILTONIANS:
        print(f"\nTesting Hamiltonian: {hamiltonian_name}")

        # Compute exact eigenvalues and eigenvectors
        matrix = H.to_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        # Get target eigenvalue and eigenstate
        if calculate_ground_state:
            exact_eigenvalue = min(eigenvalues[eigenvalues > 0])
            eigenvector_index = np.where(eigenvalues == exact_eigenvalue)[0][0]
            eigenstate = eigenvectors[:, eigenvector_index]
        else:
            exact_eigenvalue = max(eigenvalues)
            eigenvector_index = np.where(eigenvalues == exact_eigenvalue)[0][0]
            eigenstate = eigenvectors[:, eigenvector_index]

        # Prepare eigenstate circuit
        eigenstate_circuit = prepare_eigenstate_circuit(eigenstate)

        # Expected phase
        expected_phase = (exact_eigenvalue.real / (2 * np.pi)) % 1
        expected_eigenvalue = exact_eigenvalue.real

        print(f"  Expected eigenvalue: {expected_eigenvalue:.6f}")
        print(f"  Expected phase: {expected_phase:.6f}")

        # Test different qDrift_invocation values
        for qDrift_invocation in qDrift_invocations:
            try:
                # Get qDrift Kitaev result
                estimated_phase = get_qDrift_kitaev_result_v2(
                    H, eigenstate_circuit, m, qDrift_invocation
                )

                # Convert phase back to eigenvalue
                estimated_eigenvalue = estimated_phase * 2 * np.pi

                # Calculate errors
                phase_error = abs(expected_phase - estimated_phase)
                eigenvalue_error = abs(expected_eigenvalue - estimated_eigenvalue)

                # Check if within theoretical bounds
                within_eigenvalue_bound = eigenvalue_error <= theoretical_eigenvalue_error_bound
                within_phase_bound = phase_error <= theoretical_phase_error_bound

                print(f"  qDrift_invocation={qDrift_invocation}: "
                      f"estimated_eigenvalue={estimated_eigenvalue:.6f}, "
                      f"eigenvalue_error={eigenvalue_error:.6f} "
                      f"({'✓' if within_eigenvalue_bound else '✗'} bound), "
                      f"phase_error={phase_error:.6f} "
                      f"({'✓' if within_phase_bound else '✗'} bound)")

                # Store results
                results.append({
                    'hamiltonian': hamiltonian_name,
                    'qDrift_invocation': qDrift_invocation,
                    'expected_eigenvalue': expected_eigenvalue,
                    'estimated_eigenvalue': estimated_eigenvalue,
                    'expected_phase': expected_phase,
                    'estimated_phase': estimated_phase,
                    'phase_error': phase_error,
                    'eigenvalue_error': eigenvalue_error,
                    'theoretical_phase_error_bound': theoretical_phase_error_bound,
                    'theoretical_eigenvalue_error_bound': theoretical_eigenvalue_error_bound,
                    'm': m
                })

            except Exception as e:
                print(f"  Error with qDrift_invocation={qDrift_invocation}: {e}")

    # Create plots
    create_eigenvalue_error_plots(results)

    print(f"\nCompleted testing {len(HAMILTONIANS)} Hamiltonians with {len(qDrift_invocations)} different qDrift_invocation values")


def create_eigenvalue_error_plots(results):
    """Create plots showing eigenvalue estimation errors."""

    # Convert results to structured format for plotting
    hamiltonians = list(set([r['hamiltonian'] for r in results]))
    qDrift_invocations = sorted(list(set([r['qDrift_invocation'] for r in results])))

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Eigenvalue errors vs qDrift_invocation for each Hamiltonian
    for hamiltonian in hamiltonians:
        ham_results = [r for r in results if r['hamiltonian'] == hamiltonian]
        ham_results.sort(key=lambda x: x['qDrift_invocation'])

        invocations = [r['qDrift_invocation'] for r in ham_results]
        errors = [r['eigenvalue_error'] for r in ham_results]

        ax1.plot(invocations, errors, 'o-', label=hamiltonian, linewidth=2, markersize=6)

    # Add theoretical error bound as horizontal line
    if results:
        theoretical_bound = results[0]['theoretical_eigenvalue_error_bound']
        m_value = results[0]['m']
        ax1.axhline(y=theoretical_bound, color='red', linestyle='--', linewidth=2,
                   label=f'Theoretical bound (m={m_value})')

    ax1.set_xlabel('qDrift Invocations')
    ax1.set_ylabel('Eigenvalue Error')
    ax1.set_title('Eigenvalue Estimation Error vs qDrift Invocations')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Expected vs Estimated eigenvalues (scatter plot)
    colors = plt.cm.tab10(np.linspace(0, 1, len(hamiltonians)))

    for i, hamiltonian in enumerate(hamiltonians):
        ham_results = [r for r in results if r['hamiltonian'] == hamiltonian]

        expected = [r['expected_eigenvalue'] for r in ham_results]
        estimated = [r['estimated_eigenvalue'] for r in ham_results]

        ax2.scatter(expected, estimated, c=[colors[i]], label=hamiltonian,
                   s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add perfect estimation line (y=x)
    all_expected = [r['expected_eigenvalue'] for r in results]
    all_estimated = [r['estimated_eigenvalue'] for r in results]
    min_val = min(min(all_expected), min(all_estimated))
    max_val = max(max(all_expected), max(all_estimated))

    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Estimation')
    ax2.set_xlabel('Expected Eigenvalue')
    ax2.set_ylabel('Estimated Eigenvalue')
    ax2.set_title('Expected vs Estimated Eigenvalues')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')

    # Plot 3: Phase errors vs qDrift_invocation for each Hamiltonian
    for hamiltonian in hamiltonians:
        ham_results = [r for r in results if r['hamiltonian'] == hamiltonian]
        ham_results.sort(key=lambda x: x['qDrift_invocation'])

        invocations = [r['qDrift_invocation'] for r in ham_results]
        phase_errors = [r['phase_error'] for r in ham_results]

        ax3.plot(invocations, phase_errors, 'o-', label=hamiltonian, linewidth=2, markersize=6)

    # Add theoretical phase error bound as horizontal line
    if results:
        theoretical_phase_bound = results[0]['theoretical_phase_error_bound']
        m_value = results[0]['m']
        ax3.axhline(y=theoretical_phase_bound, color='red', linestyle='--', linewidth=2,
                   label=f'Theoretical bound (m={m_value})')

    ax3.set_xlabel('qDrift Invocations')
    ax3.set_ylabel('Phase Error')
    ax3.set_title('Phase Estimation Error vs qDrift Invocations')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    # Save plot
    plot_filename = f"qdrift_kitaev_qpe_eigenvalue_errors_{datetime.datetime.today().strftime('%Y-%m-%d')}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")

    # Close the figure to free memory
    plt.close(fig)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if results:
        theoretical_bound = results[0]['theoretical_eigenvalue_error_bound']
        m_value = results[0]['m']
        print(f"Theoretical eigenvalue error bound (m={m_value}): {theoretical_bound:.6f}")
        print()

    for hamiltonian in hamiltonians:
        ham_results = [r for r in results if r['hamiltonian'] == hamiltonian]
        errors = [r['eigenvalue_error'] for r in ham_results]
        phase_errors = [r['phase_error'] for r in ham_results]

        print(f"{hamiltonian}:")
        print(f"  Eigenvalue errors:")
        print(f"    Mean: {np.mean(errors):.6f}")
        print(f"    Std:  {np.std(errors):.6f}")
        print(f"    Min:  {np.min(errors):.6f}")
        print(f"    Max:  {np.max(errors):.6f}")
        print(f"  Phase errors:")
        print(f"    Mean: {np.mean(phase_errors):.6f}")
        print(f"    Max:  {np.max(phase_errors):.6f}")

        # Check if errors are within theoretical bounds
        within_bound = all(error <= theoretical_bound for error in errors)
        print(f"  All errors within theoretical bound: {within_bound}")
        print()
