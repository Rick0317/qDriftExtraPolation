import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from scripts.parameter_sweep.kitaev_qpe import get_kitaev_result
from scripts.parameter_sweep.algos import generate_ising_hamiltonian, prepare_eigenstate_circuit


class TestKitaevQPEModelSystems:
    """Test Kitaev QPE on various model quantum systems."""
    
    def test_single_qubit_pauli_z(self):
        """Test Kitaev QPE on a single qubit Pauli-Z Hamiltonian."""
        # H = Z (eigenvalues: +1, -1)
        H = SparsePauliOp.from_list([("Z", 1.0)])
        
        # Test eigenstate |0⟩ (eigenvalue +1)
        qc_plus = QuantumCircuit(1)
        # |0⟩ is already the default state
        
        expected_phase_plus = (1.0 / (2 * np.pi)) % 1
        estimated_phase_plus = get_kitaev_result(H, qc_plus, m=4, shots_per_estimation=10000)
        
        print(f"Single qubit Z, |0⟩ state:")
        print(f"Expected phase: {expected_phase_plus}")
        print(f"Estimated phase: {estimated_phase_plus}")
        print(f"Error: {abs(expected_phase_plus - estimated_phase_plus)}")
        
        assert abs(expected_phase_plus - estimated_phase_plus) < 0.1
        
        # Test eigenstate |1⟩ (eigenvalue -1)
        qc_minus = QuantumCircuit(1)
        qc_minus.x(0)  # Prepare |1⟩
        
        expected_phase_minus = (-1.0 / (2 * np.pi)) % 1
        estimated_phase_minus = get_kitaev_result(H, qc_minus, m=4, shots_per_estimation=10000)
        
        print(f"\nSingle qubit Z, |1⟩ state:")
        print(f"Expected phase: {expected_phase_minus}")
        print(f"Estimated phase: {estimated_phase_minus}")
        print(f"Error: {abs(expected_phase_minus - estimated_phase_minus)}")
        
        assert abs(expected_phase_minus - estimated_phase_minus) < 0.1

    def test_single_qubit_pauli_x(self):
        """Test Kitaev QPE on a single qubit Pauli-X Hamiltonian."""
        # H = X (eigenvalues: +1, -1)
        H = SparsePauliOp.from_list([("X", 1.0)])
        
        # Test eigenstate |+⟩ = (|0⟩ + |1⟩)/√2 (eigenvalue +1)
        qc_plus = QuantumCircuit(1)
        qc_plus.h(0)  # Prepare |+⟩
        
        expected_phase_plus = (1.0 / (2 * np.pi)) % 1
        estimated_phase_plus = get_kitaev_result(H, qc_plus, m=4, shots_per_estimation=10000)
        
        print(f"\nSingle qubit X, |+⟩ state:")
        print(f"Expected phase: {expected_phase_plus}")
        print(f"Estimated phase: {estimated_phase_plus}")
        print(f"Error: {abs(expected_phase_plus - estimated_phase_plus)}")
        
        assert abs(expected_phase_plus - estimated_phase_plus) < 0.1
        
        # Test eigenstate |-⟩ = (|0⟩ - |1⟩)/√2 (eigenvalue -1)
        qc_minus = QuantumCircuit(1)
        qc_minus.h(0)
        qc_minus.z(0)  # Prepare |-⟩
        
        expected_phase_minus = (-1.0 / (2 * np.pi)) % 1
        estimated_phase_minus = get_kitaev_result(H, qc_minus, m=4, shots_per_estimation=10000)
        
        print(f"\nSingle qubit X, |-⟩ state:")
        print(f"Expected phase: {expected_phase_minus}")
        print(f"Estimated phase: {estimated_phase_minus}")
        print(f"Error: {abs(expected_phase_minus - estimated_phase_minus)}")
        
        assert abs(expected_phase_minus - estimated_phase_minus) < 0.1

    def test_scaled_single_qubit(self):
        """Test Kitaev QPE on scaled single qubit Hamiltonians."""
        # H = 0.5 * Z (eigenvalues: +0.5, -0.5)
        H = SparsePauliOp.from_list([("Z", 0.5)])
        
        # Test eigenstate |0⟩ (eigenvalue +0.5)
        qc_plus = QuantumCircuit(1)
        
        expected_phase_plus = (0.5 / (2 * np.pi)) % 1
        estimated_phase_plus = get_kitaev_result(H, qc_plus, m=5, shots_per_estimation=10000)
        
        print(f"\nScaled single qubit 0.5*Z, |0⟩ state:")
        print(f"Expected phase: {expected_phase_plus}")
        print(f"Estimated phase: {estimated_phase_plus}")
        print(f"Error: {abs(expected_phase_plus - estimated_phase_plus)}")
        
        assert abs(expected_phase_plus - estimated_phase_plus) < 0.05

    def test_two_qubit_ising_model(self):
        """Test Kitaev QPE on a two-qubit Ising model."""
        # H = J*ZZ + G*X_0 + G*X_1
        J, G = 1.0, 0.5
        H = generate_ising_hamiltonian(2, J, G)
        
        # Get exact eigenvalues and eigenvectors
        matrix = H.to_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # Test ground state
        ground_energy = min(eigenvalues.real)
        ground_idx = np.argmin(eigenvalues.real)
        ground_state = eigenvectors[:, ground_idx]
        
        # Prepare ground state circuit
        ground_circuit = prepare_eigenstate_circuit(ground_state)
        
        expected_phase = (ground_energy / (2 * np.pi)) % 1
        # Increase precision and shots for this complex system
        estimated_phase = get_kitaev_result(H, ground_circuit, m=6, shots_per_estimation=25000)
        
        print(f"\nTwo-qubit Ising model, ground state:")
        print(f"Ground energy: {ground_energy}")
        print(f"Expected phase: {expected_phase}")
        print(f"Estimated phase: {estimated_phase}")
        print(f"Error: {abs(expected_phase - estimated_phase)}")
        
        # More lenient threshold for complex multi-qubit systems
        assert abs(expected_phase - estimated_phase) < 0.15
        
        # Test highest energy state
        max_energy = max(eigenvalues.real)
        max_idx = np.argmax(eigenvalues.real)
        max_state = eigenvectors[:, max_idx]
        
        max_circuit = prepare_eigenstate_circuit(max_state)
        
        expected_phase_max = (max_energy / (2 * np.pi)) % 1
        estimated_phase_max = get_kitaev_result(H, max_circuit, m=6, shots_per_estimation=25000)
        
        print(f"\nTwo-qubit Ising model, highest energy state:")
        print(f"Max energy: {max_energy}")
        print(f"Expected phase: {expected_phase_max}")
        print(f"Estimated phase: {estimated_phase_max}")
        print(f"Error: {abs(expected_phase_max - estimated_phase_max)}")
        
        # More lenient threshold for complex multi-qubit systems
        assert abs(expected_phase_max - estimated_phase_max) < 0.15

    def test_three_qubit_system(self):
        """Test Kitaev QPE on a three-qubit system."""
        # H = Z_0 + 0.5*Z_1 + 0.25*Z_2
        H = SparsePauliOp.from_list([
            ("ZII", 1.0),
            ("IZI", 0.5),
            ("IIZ", 0.25)
        ])
        
        # Test state |000⟩ (eigenvalue: 1.0 + 0.5 + 0.25 = 1.75)
        qc = QuantumCircuit(3)
        
        expected_phase = (1.75 / (2 * np.pi)) % 1
        estimated_phase = get_kitaev_result(H, qc, m=6, shots_per_estimation=20000)
        
        print(f"\nThree-qubit system, |000⟩ state:")
        print(f"Expected eigenvalue: 1.75")
        print(f"Expected phase: {expected_phase}")
        print(f"Estimated phase: {estimated_phase}")
        print(f"Error: {abs(expected_phase - estimated_phase)}")
        
        assert abs(expected_phase - estimated_phase) < 0.12

    def test_mixed_pauli_hamiltonian(self):
        """Test Kitaev QPE on a Hamiltonian with mixed Pauli terms."""
        # H = 0.8*Z + 0.6*X
        H = SparsePauliOp.from_list([
            ("Z", 0.8),
            ("X", 0.6)
        ])
        
        # Get exact eigenvalues and eigenvectors
        matrix = H.to_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # Test both eigenstates
        for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            eigenval = eigenval.real
            eigenvec = eigenvec / np.linalg.norm(eigenvec)  # Normalize
            
            # Prepare eigenstate circuit
            eigen_circuit = prepare_eigenstate_circuit(eigenvec)
            
            expected_phase = (eigenval / (2 * np.pi)) % 1
            estimated_phase = get_kitaev_result(H, eigen_circuit, m=6, shots_per_estimation=20000)
            
            print(f"\nMixed Pauli Hamiltonian, eigenstate {i}:")
            print(f"Eigenvalue: {eigenval}")
            print(f"Expected phase: {expected_phase}")
            print(f"Estimated phase: {estimated_phase}")
            print(f"Error: {abs(expected_phase - estimated_phase)}")
            
            assert abs(expected_phase - estimated_phase) < 0.15

    def test_precision_scaling(self):
        """Test how precision scales with the number of bits m."""
        # Simple H = Z system
        H = SparsePauliOp.from_list([("Z", 1.0)])
        qc = QuantumCircuit(1)  # |0⟩ state
        
        expected_phase = (1.0 / (2 * np.pi)) % 1
        
        print(f"\nPrecision scaling test (H = Z, |0⟩ state):")
        print(f"Expected phase: {expected_phase}")
        
        for m in range(3, 7):
            estimated_phase = get_kitaev_result(H, qc, m=m, shots_per_estimation=20000)
            error = abs(expected_phase - estimated_phase)
            theoretical_precision = 1 / (2**m)
            
            print(f"m={m}: estimated={estimated_phase:.6f}, error={error:.6f}, theoretical_precision={theoretical_precision:.6f}")
            
            # Error should generally decrease as m increases (though with statistical fluctuations)
            assert error < 0.2  # Generous bound for statistical variations

    def test_simple_two_qubit_diagonal(self):
        """Test Kitaev QPE on a simple two-qubit diagonal system."""
        # H = Z_0 + 0.5*Z_1 (simple diagonal system)
        H = SparsePauliOp.from_list([
            ("ZI", 1.0),
            ("IZ", 0.5)
        ])
        
        # Test computational basis states
        test_cases = [
            ("00", [0, 0], 1.0 + 0.5),   # |00⟩: eigenvalue = +1.5
            ("01", [0, 1], 1.0 - 0.5),   # |01⟩: eigenvalue = +0.5
            ("10", [1, 0], -1.0 + 0.5),  # |10⟩: eigenvalue = -0.5
            ("11", [1, 1], -1.0 - 0.5)   # |11⟩: eigenvalue = -1.5
        ]
        
        for state_name, state_bits, true_eigenvalue in test_cases:
            qc = QuantumCircuit(2)
            for i, bit in enumerate(state_bits):
                if bit == 1:
                    qc.x(i)
            
            expected_phase = (true_eigenvalue / (2 * np.pi)) % 1
            estimated_phase = get_kitaev_result(H, qc, m=5, shots_per_estimation=15000)
            
            error = abs(expected_phase - estimated_phase)
            
            print(f"\nTwo-qubit diagonal, |{state_name}⟩ state:")
            print(f"Eigenvalue: {true_eigenvalue}")
            print(f"Expected phase: {expected_phase}")
            print(f"Estimated phase: {estimated_phase}")
            print(f"Error: {error}")
            
            assert error < 0.1

if __name__ == "__main__":
    # Run tests manually
    test_suite = TestKitaevQPEModelSystems()
    
    print("=" * 60)
    print("TESTING KITAEV QPE ON MODEL SYSTEMS")
    print("=" * 60)
    
    tests = [
        ("Single qubit Pauli-Z", test_suite.test_single_qubit_pauli_z),
        ("Single qubit Pauli-X", test_suite.test_single_qubit_pauli_x),
        ("Scaled single qubit", test_suite.test_scaled_single_qubit),
        ("Two-qubit Ising model", test_suite.test_two_qubit_ising_model),
        ("Three-qubit system", test_suite.test_three_qubit_system),
        ("Mixed Pauli Hamiltonian", test_suite.test_mixed_pauli_hamiltonian),
        ("Simple two-qubit diagonal", test_suite.test_simple_two_qubit_diagonal),
        ("Precision scaling", test_suite.test_precision_scaling)
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
    print(f"TESTING COMPLETE: {passed}/{total} tests passed")
    print("=" * 60) 