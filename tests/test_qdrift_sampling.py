import numpy as np
import pytest
from collections import Counter
from qiskit.quantum_info import SparsePauliOp
from scripts.parameter_sweep.algos import generate_ising_hamiltonian
from scripts.parameter_sweep.kitaev_qpe import qDrift_sample


class TestQDriftSampling:
    """Test the qDrift sampling function with various Hamiltonians."""
    
    # Use the same parameters as in test_kitaev_qpe_qdrift.py
    NUM_QUBITS = 2
    J = 1.4
    G = 0.4
    
    @pytest.fixture
    def ising_hamiltonian(self):
        """Generate the same Ising Hamiltonian used in the main tests."""
        return generate_ising_hamiltonian(self.NUM_QUBITS, 0.5 * self.J, 0.5 * self.G)
    
    def test_qDrift_sample_basic_functionality(self, ising_hamiltonian):
        """Test that qDrift_sample returns valid SparsePauliOp objects."""
        H = ising_hamiltonian
        
        # Sample multiple times
        for _ in range(10):
            sampled_H = qDrift_sample(H)
            
            # Check that result is a SparsePauliOp
            assert isinstance(sampled_H, SparsePauliOp)
            
            # Check that it has exactly one term
            assert len(sampled_H.paulis) == 1
            
            # Check that coefficient is ±1
            assert abs(abs(sampled_H.coeffs[0]) - 1.0) < 1e-10
            assert sampled_H.coeffs[0] in [-1.0, 1.0]
            
            # Check that the Pauli string is one from the original Hamiltonian
            sampled_pauli = sampled_H.paulis.to_labels()[0]
            original_paulis = H.paulis.to_labels()
            assert sampled_pauli in original_paulis
    
    def test_qDrift_sample_distribution(self, ising_hamiltonian):
        """Test that sampling follows the correct probability distribution."""
        H = ising_hamiltonian
        
        # Get expected distribution
        coeffs_abs = np.abs(H.coeffs)
        lam = np.sum(coeffs_abs)
        expected_pmf = coeffs_abs / lam
        original_paulis = H.paulis.to_labels()
        
        print(f"\nOriginal Hamiltonian terms:")
        for i, (pauli, coeff) in enumerate(zip(original_paulis, H.coeffs)):
            print(f"  {pauli}: {coeff:.4f} (prob: {expected_pmf[i]:.4f})")
        
        # Sample many times
        num_samples = 10000
        sampled_paulis = []
        sampled_signs = []
        
        for _ in range(num_samples):
            sampled_H = qDrift_sample(H)
            pauli_str = sampled_H.paulis.to_labels()[0]
            sign = sampled_H.coeffs[0]
            
            sampled_paulis.append(pauli_str)
            sampled_signs.append(sign)
        
        # Count occurrences
        pauli_counts = Counter(sampled_paulis)
        
        print(f"\nSampling results from {num_samples} samples:")
        for pauli in original_paulis:
            count = pauli_counts.get(pauli, 0)
            observed_freq = count / num_samples
            expected_freq = expected_pmf[list(original_paulis).index(pauli)]
            
            print(f"  {pauli}: {count} samples ({observed_freq:.4f} vs expected {expected_freq:.4f})")
            
            # Statistical test (allow 3-sigma deviation)
            expected_count = expected_freq * num_samples
            std_dev = np.sqrt(expected_count * (1 - expected_freq))
            
            assert abs(count - expected_count) < 3 * std_dev, \
                f"Sampling frequency for {pauli} deviates too much from expected"
    
    def test_qDrift_sample_sign_preservation(self, ising_hamiltonian):
        """Test that signs are correctly preserved in sampling."""
        H = ising_hamiltonian
        
        original_paulis = H.paulis.to_labels()
        original_coeffs = H.coeffs
        
        # Create mapping of Pauli string to original sign
        pauli_to_sign = {}
        for pauli, coeff in zip(original_paulis, original_coeffs):
            pauli_to_sign[pauli] = np.sign(coeff.real)
        
        # Sample and check signs
        num_samples = 1000
        for _ in range(num_samples):
            sampled_H = qDrift_sample(H)
            sampled_pauli = sampled_H.paulis.to_labels()[0]
            sampled_sign = np.sign(sampled_H.coeffs[0].real)
            expected_sign = pauli_to_sign[sampled_pauli]
            
            assert sampled_sign == expected_sign, \
                f"Sign mismatch for {sampled_pauli}: got {sampled_sign}, expected {expected_sign}"
    
    def test_qDrift_sample_with_simple_hamiltonian(self):
        """Test qDrift sampling with a simple, well-understood Hamiltonian."""
        # H = 2*Z + 1*X (coefficients: [2, 1], probabilities: [2/3, 1/3])
        H = SparsePauliOp.from_list([("Z", 2.0), ("X", 1.0)])
        
        num_samples = 6000
        z_count = 0
        x_count = 0
        
        for _ in range(num_samples):
            sampled_H = qDrift_sample(H)
            pauli_str = sampled_H.paulis.to_labels()[0]
            
            if pauli_str == "Z":
                z_count += 1
                # Should always have coefficient +1 (since original is positive)
                assert sampled_H.coeffs[0] == 1.0
            elif pauli_str == "X":
                x_count += 1
                # Should always have coefficient +1 (since original is positive)
                assert sampled_H.coeffs[0] == 1.0
        
        # Check distribution (Z should appear ~2/3 of the time, X ~1/3)
        z_freq = z_count / num_samples
        x_freq = x_count / num_samples
        
        print(f"\nSimple Hamiltonian test:")
        print(f"Z frequency: {z_freq:.4f} (expected: 0.6667)")
        print(f"X frequency: {x_freq:.4f} (expected: 0.3333)")
        
        assert abs(z_freq - 2/3) < 0.05, f"Z frequency {z_freq} too far from expected 2/3"
        assert abs(x_freq - 1/3) < 0.05, f"X frequency {x_freq} too far from expected 1/3"
    
    def test_qDrift_sample_with_negative_coefficients(self):
        """Test qDrift sampling with negative coefficients."""
        # H = 1*Z - 2*X (coefficients: [1, -2], probabilities: [1/3, 2/3])
        H = SparsePauliOp.from_list([("Z", 1.0), ("X", -2.0)])
        
        num_samples = 6000
        z_positive_count = 0
        x_negative_count = 0
        
        for _ in range(num_samples):
            sampled_H = qDrift_sample(H)
            pauli_str = sampled_H.paulis.to_labels()[0]
            coeff = sampled_H.coeffs[0]
            
            if pauli_str == "Z":
                z_positive_count += 1
                # Should always have coefficient +1 (since original is positive)
                assert coeff == 1.0
            elif pauli_str == "X":
                x_negative_count += 1
                # Should always have coefficient -1 (since original is negative)
                assert coeff == -1.0
        
        # Check distribution and signs
        z_freq = z_positive_count / num_samples
        x_freq = x_negative_count / num_samples
        
        print(f"\nNegative coefficient test:")
        print(f"Z (+1) frequency: {z_freq:.4f} (expected: 0.3333)")
        print(f"X (-1) frequency: {x_freq:.4f} (expected: 0.6667)")
        
        assert abs(z_freq - 1/3) < 0.05, f"Z frequency {z_freq} too far from expected 1/3"
        assert abs(x_freq - 2/3) < 0.05, f"X frequency {x_freq} too far from expected 2/3"
    
    def test_qDrift_sample_statistical_properties(self, ising_hamiltonian):
        """Test statistical properties of the qDrift sampling."""
        H = ising_hamiltonian
        
        # Calculate expected value of sampled Hamiltonian
        # E[sampled_H] should equal the original Hamiltonian (up to normalization)
        coeffs_abs = np.abs(H.coeffs)
        lam = np.sum(coeffs_abs)
        
        num_samples = 5000
        sampled_operators = []
        
        for _ in range(num_samples):
            sampled_H = qDrift_sample(H)
            sampled_operators.append(sampled_H)
        
        # Compute empirical average
        # Convert to matrices for easier averaging
        matrices = [op.to_matrix() * lam for op in sampled_operators]  # Scale by lambda
        avg_matrix = np.mean(matrices, axis=0)
        
        # Compare with original Hamiltonian matrix
        original_matrix = H.to_matrix()
        
        # Check if they're close (within statistical error)
        matrix_diff = np.linalg.norm(avg_matrix - original_matrix)
        relative_error = matrix_diff / np.linalg.norm(original_matrix)
        
        print(f"\nStatistical properties test:")
        print(f"Matrix difference norm: {matrix_diff:.6f}")
        print(f"Relative error: {relative_error:.6f}")
        
        # Should be close due to law of large numbers
        assert relative_error < 0.1, f"Empirical average deviates too much from original: {relative_error}"
    
    def test_qDrift_sample_reproducibility(self, ising_hamiltonian):
        """Test that sampling is properly random (different results on multiple calls)."""
        H = ising_hamiltonian
        
        # Sample multiple times and ensure we get variety
        samples = []
        for _ in range(100):
            sampled_H = qDrift_sample(H)
            pauli_str = sampled_H.paulis.to_labels()[0]
            coeff = sampled_H.coeffs[0]
            samples.append((pauli_str, coeff))
        
        # Should have more than one unique sample (very high probability)
        unique_samples = set(samples)
        assert len(unique_samples) > 1, "qDrift sampling appears to be deterministic"
        
        print(f"\nReproducibility test:")
        print(f"Got {len(unique_samples)} unique samples out of 100")
        
        # Should have reasonable variety (not all the same)
        assert len(unique_samples) >= 2, "Insufficient variety in sampling"


def test_qdrift_sampling_manual():
    """Manual test runner for qDrift sampling."""
    print("=" * 60)
    print("TESTING qDRIFT SAMPLING FUNCTION")
    print("=" * 60)
    
    # Create test instance
    test_suite = TestQDriftSampling()
    
    # Generate Hamiltonian
    H = generate_ising_hamiltonian(test_suite.NUM_QUBITS, 
                                   0.5 * test_suite.J, 
                                   0.5 * test_suite.G)
    
    print(f"\nUsing Ising Hamiltonian with NUM_QUBITS={test_suite.NUM_QUBITS}, J={test_suite.J}, G={test_suite.G}")
    print("Hamiltonian terms:")
    for pauli, coeff in zip(H.paulis.to_labels(), H.coeffs):
        print(f"  {pauli}: {coeff}")
    
    tests = [
        ("Basic functionality", lambda: test_suite.test_qDrift_sample_basic_functionality(H)),
        ("Distribution correctness", lambda: test_suite.test_qDrift_sample_distribution(H)),
        ("Sign preservation", lambda: test_suite.test_qDrift_sample_sign_preservation(H)),
        ("Simple Hamiltonian", test_suite.test_qDrift_sample_with_simple_hamiltonian),
        ("Negative coefficients", test_suite.test_qDrift_sample_with_negative_coefficients),
        ("Statistical properties", lambda: test_suite.test_qDrift_sample_statistical_properties(H)),
        ("Reproducibility", lambda: test_suite.test_qDrift_sample_reproducibility(H))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"\n✓ {test_name} test passed")
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} test failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"qDRIFT SAMPLING TESTS COMPLETE: {passed}/{total} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    test_qdrift_sampling_manual() 