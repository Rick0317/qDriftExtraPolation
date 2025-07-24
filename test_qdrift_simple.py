#!/usr/bin/env python3
"""
Simple test script for qDrift_sample function using the exact Hamiltonian
from test_kitaev_qpe_qdrift.py
"""

import numpy as np
from collections import Counter
from scripts.parameter_sweep.algos import generate_ising_hamiltonian
from scripts.parameter_sweep.kitaev_qpe import qDrift_sample

def test_qdrift_with_test_hamiltonian():
    """Test qDrift_sample with the exact parameters from test_kitaev_qpe_qdrift.py"""
    
    print("qDRIFT SAMPLING TEST")
    print("=" * 50)
    
    # Use exact same parameters as in test_kitaev_qpe_qdrift.py
    NUM_QUBITS = 2
    J = 1.4
    G = 0.4
    
    # Generate the same Hamiltonian
    H = generate_ising_hamiltonian(NUM_QUBITS, 0.5 * J, 0.5 * G)
    
    print(f"Parameters: NUM_QUBITS={NUM_QUBITS}, J={J}, G={G}")
    print(f"Effective J={0.5*J}, effective G={0.5*G}")
    print(f"\nOriginal Hamiltonian: {H}")
    print(f"Number of original terms: {len(H.paulis)}")
    
    # Simplify to handle potential duplicates
    H_simplified = H.simplify()
    print(f"Simplified Hamiltonian: {H_simplified}")
    print(f"Number of simplified terms: {len(H_simplified.paulis)}")
    
    print("\nHamiltonian terms (after simplification):")
    
    original_paulis = H_simplified.paulis.to_labels()
    original_coeffs = H_simplified.coeffs
    coeffs_abs = np.abs(original_coeffs)
    lam = np.sum(coeffs_abs)
    expected_pmf = coeffs_abs / lam
    
    for i, (pauli, coeff) in enumerate(zip(original_paulis, original_coeffs)):
        print(f"  {pauli}: {coeff:.4f} (|coeff|={coeffs_abs[i]:.4f}, prob={expected_pmf[i]:.4f})")
    
    print(f"\nLambda (sum of |coeffs|): {lam:.4f}")
    
    # Test basic functionality
    print("\n" + "-" * 30)
    print("BASIC FUNCTIONALITY TEST")
    print("-" * 30)
    
    print("Sampling 10 times:")
    for i in range(10):
        sampled_H = qDrift_sample(H)
        pauli_str = sampled_H.paulis.to_labels()[0]
        coeff = sampled_H.coeffs[0]
        print(f"  Sample {i+1}: {pauli_str} with coefficient {coeff}")
        
        # Verify it's valid
        assert pauli_str in original_paulis
        assert abs(abs(coeff) - 1.0) < 1e-10
    
    # Test distribution
    print("\n" + "-" * 30)
    print("DISTRIBUTION TEST")
    print("-" * 30)
    
    num_samples = 10000
    sampled_paulis = []
    sampled_coeffs = []
    
    for _ in range(num_samples):
        sampled_H = qDrift_sample(H)
        pauli_str = sampled_H.paulis.to_labels()[0]
        coeff = sampled_H.coeffs[0]
        sampled_paulis.append(pauli_str)
        sampled_coeffs.append(coeff)
    
    # Count occurrences
    pauli_counts = Counter(sampled_paulis)
    
    print(f"Results from {num_samples} samples:")
    all_close = True
    
    for i, pauli in enumerate(original_paulis):
        count = pauli_counts.get(pauli, 0)
        observed_freq = count / num_samples
        expected_freq = expected_pmf[i]
        error = abs(observed_freq - expected_freq)
        
        print(f"  {pauli}: {count:>5} samples ({observed_freq:.4f} vs expected {expected_freq:.4f}, error={error:.4f})")
        
        if error > 0.02:  # 2% tolerance
            all_close = False
    
    if all_close:
        print("✓ Distribution test PASSED")
    else:
        print("✗ Distribution test FAILED (some frequencies off by >2%)")
    
    # Test sign preservation
    print("\n" + "-" * 30)
    print("SIGN PRESERVATION TEST")
    print("-" * 30)
    
    # Create mapping of Pauli string to expected sign (using simplified Hamiltonian)
    pauli_to_expected_sign = {}
    for pauli, coeff in zip(original_paulis, original_coeffs):
        pauli_to_expected_sign[pauli] = np.sign(coeff.real)
    
    sign_correct = True
    sign_counts = {}
    
    for pauli, coeff in zip(sampled_paulis, sampled_coeffs):
        expected_sign = pauli_to_expected_sign[pauli]
        actual_sign = np.sign(coeff.real)
        
        if pauli not in sign_counts:
            sign_counts[pauli] = {'correct': 0, 'total': 0}
        
        sign_counts[pauli]['total'] += 1
        if actual_sign == expected_sign:
            sign_counts[pauli]['correct'] += 1
        else:
            sign_correct = False
    
    print("Sign preservation results:")
    for pauli in original_paulis:
        if pauli in sign_counts:
            correct = sign_counts[pauli]['correct']
            total = sign_counts[pauli]['total']
            expected_sign = pauli_to_expected_sign[pauli]
            print(f"  {pauli} (expected sign {expected_sign:+.0f}): {correct}/{total} correct")
    
    if sign_correct:
        print("✓ Sign preservation test PASSED")
    else:
        print("✗ Sign preservation test FAILED")
    
    # Statistical properties test
    print("\n" + "-" * 30)
    print("STATISTICAL PROPERTIES TEST")
    print("-" * 30)
    
    # Compute empirical average of λ * sampled operators
    matrices = []
    for _ in range(1000):  # Smaller sample for matrix operations
        sampled_H = qDrift_sample(H)
        matrices.append(sampled_H.to_matrix() * lam)
    
    avg_matrix = np.mean(matrices, axis=0)
    original_matrix = H.to_matrix()
    
    matrix_diff = np.linalg.norm(avg_matrix - original_matrix)
    relative_error = matrix_diff / np.linalg.norm(original_matrix)
    
    print(f"Matrix difference norm: {matrix_diff:.6f}")
    print(f"Relative error: {relative_error:.6f}")
    
    if relative_error < 0.1:
        print("✓ Statistical properties test PASSED")
    else:
        print("✗ Statistical properties test FAILED")
    
    print("\n" + "=" * 50)
    print("qDRIFT SAMPLING TEST COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    test_qdrift_with_test_hamiltonian() 