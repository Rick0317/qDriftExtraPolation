import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Union
import scipy
import random
from functools import reduce

from qiskit import transpile
from qiskit_aer import AerSimulator  # as of 25Mar2025
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, UnitaryGate, PhaseGate, RZGate
from qiskit.quantum_info import Operator
from sympy import Matrix, latex
from IPython.display import display, Math
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.circuit.library import Initialize


# import basic plot tools
from qiskit.visualization import plot_histogram

# Type aliases
coefficient = float
Hamiltonian = list[tuple[coefficient, Pauli]]




# ---------------------------------------------------- utils ----------------------------------------------------

def generate_ising_hamiltonian(num_qubits: int, J, g) -> Union[SparsePauliOp, Pauli]:
    z_terms = []
    z_coeffs = []
    
    # ZZ interaction terms
    for j in range(num_qubits):
        pauli_string = ['I'] * num_qubits
        pauli_string[j] = 'Z'
        pauli_string[(j + 1) % num_qubits] = 'Z'  # Periodic boundary conditions
        z_terms.append("".join(pauli_string))
        z_coeffs.append(-J)  # Coefficient for ZZ interaction

    x_terms = []
    x_coeffs = []
    
    # X field terms
    for j in range(num_qubits):
        pauli_string = ['I'] * num_qubits
        pauli_string[j] = 'X'
        x_terms.append("".join(pauli_string))
        x_coeffs.append(-g)  # Coefficient for X term

    # Combine the Z and X terms into a single Hamiltonian
    all_terms = z_terms + x_terms
    all_coeffs = z_coeffs + x_coeffs

    return SparsePauliOp(all_terms, coeffs=all_coeffs)


def exponentiate_hamiltonian(hamiltonian: SparsePauliOp, time: float) -> Operator:
    """Exponentiates the Hamiltonian to obtain U = e^(-i H t)."""
    matrix = hamiltonian.to_matrix()
    unitary_matrix = scipy.linalg.expm(1j * time * matrix)
    return Operator(unitary_matrix)


def calculate_ground_state_and_energy(H: SparsePauliOp) -> List[complex]:
    """Calculates the eigenvalues of the Hamiltonian."""
    matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # Sort eigenvalues and eigenvectors
    ground_state = eigenvectors[:, np.argmin(eigenvalues)]
    ground_energy = np.min(eigenvalues)
    return ground_state, ground_energy


def prepare_eigenstate_circuit(ground_state: np.ndarray) -> QuantumCircuit:
    """
    Prepare a quantum circuit that initializes the ground state.
    
    Args:
        ground_state (np.ndarray): State vector representing the ground state.

    Returns:
        QuantumCircuit: Circuit that prepares the ground state.
    """
    num_qubits = int(np.log2(len(ground_state)))
    if 2 ** num_qubits != len(ground_state):
        raise ValueError("The length of the state vector must be a power of 2.")

    # Normalize the state vector
    ground_state = ground_state / np.linalg.norm(ground_state)

    # Initialize circuit
    qc = QuantumCircuit(num_qubits)
    init_gate = Initialize(ground_state)
    qc.append(init_gate, range(num_qubits))

    # Optional: Simplify the circuit using transpile
    # from qiskit import transpile
    # qc = transpile(qc, optimization_level=3)

    qc.barrier()
    return qc

# -------------------------------------------------------------------------- actial algos -----------------------------

def generate_qpe_circuit_simple(total_qubits, phase):
    """
    Assumptions: 
    - The target unitary acts on *one* qubit (the last one)
    - The target unitary is a phase gate:
        P(\theta) =
                \begin{pmatrix}
                    1 & 0 \\
                    0 & e^{i\theta}
                \end{pmatrix}
    """
    num_ancilla = total_qubits-1
    qpe = QuantumCircuit(total_qubits, num_ancilla) # num qubits, num classical bits (to store meaurements)
    qpe.x(num_ancilla) # because ket(1) is an eigenvector of the phase gate

    for qubit in range(num_ancilla):
        qpe.h(qubit)
        
    repetitions = 1
    for counting_qubit in range(num_ancilla):
        for i in range(repetitions):
            qpe.cp(phase, counting_qubit, num_ancilla); # Apply C-PhaseGate to last qubit (target qubit) controlled by counting qubit
        repetitions *= 2
        
    # Apply the inverse QFT
    list_of_ancilla_qubits = [i for i in range(num_ancilla)]
    qpe.append(QFT(3, inverse=True), list_of_ancilla_qubits) 

    qpe.measure(list_of_ancilla_qubits, list_of_ancilla_qubits) # Measure the ancilla qubits
    return qpe


def standard_qpe(unitary: Operator, eigenstate: QuantumCircuit, num_ancilla: int) -> QuantumCircuit:
    """Constructs a standard Quantum Phase Estimation (QPE) circuit using repeated controlled-U applications."""
    num_qubits = unitary.num_qubits
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    # Prepare eigenstate on system qubits
    qc.append(eigenstate, range(num_ancilla, num_ancilla + num_qubits))

    # Apply Hadamard gates to ancilla qubits
    qc.h(range(num_ancilla))

    # Apply controlled-U^(2^k) using repeated controlled applications of U
    for k in range(num_ancilla):
        controlled_U = UnitaryGate(unitary.data).control(1, label=f"U")
        
        # Apply controlled-U 2^k times
        for _ in range(2**k):  
            qc.append(controlled_U, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))

    # Apply inverse QFT on ancilla qubits
    qc.append(QFT(num_ancilla, inverse=True, do_swaps=True), range(num_ancilla))

    # Measure ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))

    return qc


# Function to sample unitaries from the qDRIFT distribution
def qdrift_sample(hamiltonian: SparsePauliOp, time: float, num_samples: int) -> Tuple[List[SparsePauliOp], List[str]]:
    # Extract coefficients and Pauli strings
    coeffs_absolute_values = np.abs(hamiltonian.coeffs)
    paulis = hamiltonian.paulis.to_labels()
    
    # Calculate lambda and tau
    lam = np.sum(coeffs_absolute_values)
    tau = time * lam / num_samples
    
    # Define sampling distribution
    pmf = coeffs_absolute_values / lam
    
    sampled_unitaries = []
    labels = []
    
    # Sample based on the distribution
    for _ in range(num_samples):
        idx = random.choices(population=range(len(paulis)), weights=pmf, k=1)[0]
        pauli_string = paulis[idx]
        original_coeff = hamiltonian.coeffs[idx]

        if original_coeff < 0:
            h_j = SparsePauliOp([pauli_string], [-1.0])
        else:
            h_j = SparsePauliOp([pauli_string], [1.0])

        unitary = exponentiate_hamiltonian(h_j, tau)

        sampled_unitaries.append(unitary)
        
        # Label for visualization
        labels.append(f"$e^{{i \\tau {pauli_string}}}$")
    
    return sampled_unitaries, labels


# Function to construct controlled unitaries
def construct_controlled_unitary(sampled_unitaries, labels):
    controlled_unitaries = []
    for unitary, label in zip(sampled_unitaries, labels):
        controlled_unitary = UnitaryGate(unitary, label=label).control(1)
        controlled_unitaries.append(controlled_unitary)
    return controlled_unitaries

# Function to perform qDRIFT-based QPE
def qdrift_qpe(hamiltonian: SparsePauliOp, time: float, eigenstate, num_qubits: int, num_ancilla: int):
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    # Initialize the eigenstate
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circuit = QuantumCircuit(num_qubits, name='Eigenstate')
        eigenstate_circuit.initialize(eigenstate)
    else:
        eigenstate_circuit = eigenstate

    qc.append(eigenstate_circuit, range(num_ancilla, num_ancilla + num_qubits))
    
    # Apply QFT to ancilla qubits
    qc.append(QFT(num_ancilla), range(num_ancilla))
    
    # Controlled qDRIFT unitaries
    for k in range(num_ancilla):
        for _ in range(2 ** k):
            # Sample unitaries using the new qdrift_sample function
            sampled_unitaries, labels = qdrift_sample(hamiltonian, time, num_samples=1)
            for unitary, label in zip(sampled_unitaries, labels):
                controlled_unitary = UnitaryGate(unitary, label=label).control(1)
                qc.append(controlled_unitary, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))
    
    # Apply inverse QFT
    qc.append(QFT(num_ancilla, inverse=True), range(num_ancilla))
    
    # Measure the ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))
    
    return qc