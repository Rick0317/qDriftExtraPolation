from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, circuit_drawer
from typing import Optional, Union
import random
import numpy as np
from functools import reduce
import scipy.linalg
import pandas as pd

# Function to generate a random Hamiltonian
def generate_random_hamiltonian(num_qubits, num_terms):
    pauli_matrices = [Pauli('X'), Pauli('Z'), Pauli('I')]
    hamiltonian_terms = []
    for _ in range(num_terms):
        pauli_string = reduce(lambda x, y: x.tensor(y),
                              random.choices(pauli_matrices, k=num_qubits))
        hamiltonian_terms.append((random.uniform(0, 1), pauli_string))
    return hamiltonian_terms

# Function to sample unitaries from the qDRIFT distribution
def qdrift_sample(hamiltonian_terms, time, num_samples):
    lam = sum(abs(term[0]) for term in hamiltonian_terms)
    tau = time * lam / num_samples
    sampled_unitaries = []
    hamiltonian_specific_pmf = [abs(coeff) for coeff, _ in hamiltonian_terms]
    num_terms = len(hamiltonian_terms)
    for _ in range(num_samples):
        j = random.choices(range(num_terms), weights=hamiltonian_specific_pmf, k=1)[0]
        h_j = hamiltonian_terms[j][1]
        v = scipy.linalg.expm(1j * tau * h_j.to_matrix())
        sampled_unitaries.append(v)
    return sampled_unitaries

# Function to construct controlled unitaries
def construct_controlled_unitary(sampled_unitaries):
    controlled_unitaries = []
    for unitary in sampled_unitaries:
        controlled_unitary = UnitaryGate(unitary).control(1)
        controlled_unitaries.append(controlled_unitary)
    return controlled_unitaries

# Function to calculate the eigenstate associated with the smallest eigenvalue
def calculate_smallest_eigenstate(hamiltonian_terms):
    # Construct the full Hamiltonian matrix
    H = sum(coeff * op.to_matrix() for coeff, op in hamiltonian_terms)
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    # Find the smallest eigenvalue and corresponding eigenstate
    smallest_eigenvalue = eigenvalues[0]
    smallest_eigenstate = eigenvectors[:, 0]
    return smallest_eigenvalue, smallest_eigenstate

# Function to perform qDRIFT-based QPE
def qdrift_qpe(hamiltonian_terms, time, num_samples, eigenstate, num_qubits, num_ancilla):
    # Initialize the quantum circuit
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)
    
    # Prepare the eigenstate
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circuit = QuantumCircuit(num_qubits, name='eigenstate')
        eigenstate_circuit.initialize(eigenstate)
    else:
        eigenstate_circuit = eigenstate
    
    qc.append(eigenstate_circuit, range(num_ancilla, num_ancilla + num_qubits))
    
    # Apply QFT to the ancilla qubits
    qc.append(QFT(num_ancilla), range(num_ancilla))
    
    # Perform controlled qDRIFT unitaries
    for k in range(num_ancilla):
        sampled_unitaries = qdrift_sample(hamiltonian_terms, time, 2**k)
        controlled_unitaries = construct_controlled_unitary(sampled_unitaries)
        for unitary in controlled_unitaries:
            qc.append(unitary, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))
    
    # Apply inverse QFT
    qc.append(QFT(num_ancilla, inverse=True), range(num_ancilla))
    
    # Measure the ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))
    
    return qc

# Function to calculate the error between estimated and actual eigenvalues
def calculate_error(estimated_phase, actual_eigenvalue, num_ancilla):
    # Convert the estimated phase to an eigenvalue
    estimated_eigenvalue = 2 * np.pi * estimated_phase / (2**num_ancilla)
    # Calculate the absolute error
    error = abs(estimated_eigenvalue - actual_eigenvalue)
    return error

# Function to store results in a Pandas DataFrame
def store_results(estimated_phases, actual_eigenvalues, errors):
    data = {
        'Estimated Phase': estimated_phases,
        'Actual Eigenvalue': actual_eigenvalues,
        'Error': errors
    }
    df = pd.DataFrame(data)
    return df

# Example usage
num_qubits = 3
num_terms = 5
num_ancilla = 3
time = 1.0
num_samples = 100

# Generate a random Hamiltonian
hamiltonian_terms = generate_random_hamiltonian(num_qubits, num_terms)

# Calculate the smallest eigenvalue and corresponding eigenstate
smallest_eigenvalue, smallest_eigenstate = calculate_smallest_eigenstate(hamiltonian_terms)

# Run qDRIFT-based QPE
qc = qdrift_qpe(hamiltonian_terms, time, num_samples, smallest_eigenstate, num_qubits, num_ancilla)

# Visualize the circuit
print("Quantum Circuit:")
print(qc.draw())

# Simulate the circuit
backend = Aer.get_backend('qasm_simulator')
job = backend.run(transpile(qc, backend), shots=1024)
result = job.result()
counts = result.get_counts()

# Calculate the estimated phase
estimated_phase = max(counts, key=counts.get)  # Most frequent measurement
estimated_phase = int(estimated_phase, 2)  # Convert binary string to integer

# Calculate the error
error = calculate_error(estimated_phase, smallest_eigenvalue, num_ancilla)

# Store results in a Pandas DataFrame
results_df = store_results([estimated_phase], [smallest_eigenvalue], [error])
print("\nResults:")
print(results_df)

# Visualize the measurement results
plot_histogram(counts)