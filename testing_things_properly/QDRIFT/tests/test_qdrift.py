import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pytest
import numpy as np
from qiskit.quantum_info import Pauli, Operator
from scripts.algo.qft_qpe_qdrift import qdrift_sample
from scipy.linalg import expm

def test_qdrift_sample_basic():
    """Test basic functionality of qdrift_sample"""
    # Create a simple 2-qubit Hamiltonian with known terms
    hamiltonian_terms = [
        (1.0, Pauli('ZZ')),  # Z1Z2
        (0.5, Pauli('XX')),  # X1X2
        (0.3, Pauli('YY'))   # Y1Y2
    ]

    time = 1.0
    num_samples = 1000

    # Get samples
    sampled_unitaries, labels = qdrift_sample(hamiltonian_terms, time, num_samples)

    # Test basic properties
    assert len(sampled_unitaries) == num_samples, "Wrong number of sampled unitaries"
    assert len(labels) == num_samples, "Wrong number of labels"
    assert all(isinstance(u, np.ndarray) for u in sampled_unitaries), "Unitaries should be numpy arrays"
    assert all(isinstance(l, str) for l in labels), "Labels should be strings"

    # Test unitary properties
    for unitary in sampled_unitaries:
        assert unitary.shape == (4, 4), "Unitaries should be 4x4 matrices (2 qubits)"
        # Check if unitary (U†U = I)
        assert np.allclose(unitary.conj().T @ unitary, np.eye(4), atol=1e-10), "Matrices should be unitary"

def test_qdrift_sample_probabilities():
    """Test if sampling follows the correct probability distribution"""
    # Create Hamiltonian with distinct coefficients to test probabilities
    hamiltonian_terms = [
        (2.0, Pauli('ZZ')),  # Should be sampled ~50% of the time
        (1.0, Pauli('XX')),  # Should be sampled ~25% of the time
        (1.0, Pauli('YY'))   # Should be sampled ~25% of the time
    ]

    time = 1.0
    num_samples = 10000  # Large number for statistical significance

    # Get samples
    sampled_unitaries, labels = qdrift_sample(hamiltonian_terms, time, num_samples)

    # Count occurrences of each term
    term_counts = {0: 0, 1: 0, 2: 0}
    for label in labels:
        # Extract term index from label (e.g., "e^{i·0.1·H_0}" -> 0)
        term_idx = int(label.split('H_')[1].split('}')[0])
        term_counts[term_idx] += 1

    # Calculate empirical probabilities
    empirical_probs = {k: v/num_samples for k, v in term_counts.items()}

    # Expected probabilities based on coefficients
    total_coeff = sum(abs(coeff) for coeff, _ in hamiltonian_terms)
    expected_probs = {i: abs(coeff)/total_coeff for i, (coeff, _) in enumerate(hamiltonian_terms)}

    # Check if empirical probabilities are close to expected (within 5% error)
    for term_idx in range(len(hamiltonian_terms)):
        assert abs(empirical_probs[term_idx] - expected_probs[term_idx]) < 0.05, \
            f"Probability for term {term_idx} is off: expected {expected_probs[term_idx]:.3f}, got {empirical_probs[term_idx]:.3f}"

def test_qdrift_sample_evolution():
    """Test if the sampled unitaries correctly represent time evolution"""
    # Create a simple Hamiltonian
    hamiltonian_terms = [(1.0, Pauli('ZZ'))]
    time = 0.5
    num_samples = 1  # We only need one sample for this test

    # Get sample
    sampled_unitaries, _ = qdrift_sample(hamiltonian_terms, time, num_samples)
    sampled_unitary = sampled_unitaries[0]

    # Calculate exact evolution
    H = hamiltonian_terms[0][1].to_matrix()
    exact_unitary = expm(1j * time * H)

    # Check if the sampled unitary matches the exact evolution
    assert np.allclose(sampled_unitary, exact_unitary, atol=1e-10), \
        "Sampled unitary does not match exact time evolution"
