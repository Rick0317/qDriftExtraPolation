import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Union, Optional
import scipy
import random
from functools import reduce
from tqdm import tqdm

from qiskit import transpile
from qiskit_aer import AerSimulator  # as of 25Mar2025
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, UnitaryGate, PhaseGate, RZGate
from qiskit.quantum_info import Operator, Statevector
from sympy import Matrix, latex
from IPython.display import display, Math
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.circuit.library import Initialize
from line_profiler import profile

# import basic plot tools
from qiskit.visualization import plot_histogram

# Type aliases
coefficient = float
Hamiltonian = list[tuple[coefficient, Pauli]]




# ---------------------------------------------------- utils ----------------------------------------------------

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

@profile
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

def generate_random_hamiltonian_with_pauli_tensor_structure(num_qubits: int, num_terms: int) -> SparsePauliOp:
    """
    Generates a random Hamiltonian with a specified number of Pauli terms.
    
    Args:
        num_qubits (int): Number of qubits in the Hamiltonian.
        num_terms (int): Number of Pauli terms in the Hamiltonian.

    Returns:
        SparsePauliOp: Random Hamiltonian with Pauli tensor structure.
    """
    paulis = []
    coeffs = []

    while len(paulis) < num_terms:
        # Generate a random Pauli string (I, X, Y, Z) of length num_qubits
        term = ''.join(np.random.choice(['I', 'X', 'Z'], size=num_qubits))
        if term not in paulis:
            paulis.append(term)
            coeffs.append(np.random.uniform(-1, 1))  # real coefficient

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))

# -------------------------------------------------------------------------- Non-stochastic QPE -----------------------------
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

@profile
def standard_qpe(unitary: Operator, eigenstate: QuantumCircuit, num_ancilla: int) -> QuantumCircuit:
    """Constructs a standard Quantum Phase Estimation (QPE) circuit using repeated controlled-U applications."""
    num_qubits = unitary.num_qubits
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    # Prepare eigenstate on system qubits
    qc.append(eigenstate, range(num_ancilla, num_ancilla + num_qubits))

    # Apply Hadamard gates to ancilla qubits
    qc.h(range(num_ancilla))

    # Apply controlled-U^(2^k) using repeated controlled applications of U
    for k in tqdm(range(num_ancilla), desc="Applying controlled-U powers"):
        controlled_U = UnitaryGate(unitary.data).control(1, label=f"U")
        
        # Apply controlled-U 2^k times
        for _ in range(2**k):  
            qc.append(controlled_U, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))

    # Apply inverse QFT on ancilla qubits
    qc.append(QFT(num_ancilla, inverse=True, do_swaps=True), range(num_ancilla))

    # Measure ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))

    return qc

# ---------------------------------------------------- qDRIFT-QPE ----------------------------------------------------
def qdrift_sample_naive(hamiltonian: SparsePauliOp, time: float, num_samples: int) -> Tuple[List[SparsePauliOp], List[str]]:
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

# Function to perform qDRIFT-based QPE
def qdrift_qpe(hamiltonian: SparsePauliOp, time: float, eigenstate: Union[np.array, Statevector, QuantumCircuit], num_qubits: int, num_ancilla: int, num_samples_per_channel_invocation=1, include_measurements=True):
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    # Initialize the eigenstate
    if isinstance(eigenstate, np.ndarray) or isinstance(eigenstate, Statevector):
        eigenstate_circuit = QuantumCircuit(num_qubits, name='Eigenstate')
        eigenstate_circuit.initialize(eigenstate)
    else:
        eigenstate_circuit = eigenstate

    qc.append(eigenstate_circuit, range(num_ancilla, num_ancilla + num_qubits))
    
    # Apply QFT to ancilla qubits
    qc.append(QFT(num_ancilla), range(num_ancilla))
    k = 0
    # Controlled qDRIFT unitaries
    for k in range(num_ancilla):
        for _ in range(2 ** k):
            # Sample unitaries using the new qdrift_sample function
            sampled_unitaries, labels = qdrift_sample_naive(hamiltonian, time, num_samples=num_samples_per_channel_invocation)
            for unitary, label in zip(sampled_unitaries, labels):
                controlled_unitary = UnitaryGate(unitary, label=label).control(1)
                qc.append(controlled_unitary, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))
    
    # Apply inverse QFT
    qc.append(QFT(num_ancilla, inverse=True), range(num_ancilla))
    
    # Measure the ancilla qubits
    if include_measurements:
        qc.measure(range(num_ancilla), range(num_ancilla))
    
    return qc

def qdrift_qpe_extra_random(hamiltonian: SparsePauliOp, 
                            eigenstate:Union[QuantumCircuit, np.array], 
                            num_qubits: int, num_ancilla: int, 
                            total_simulation_time, 
                            num_samples: int = 1000000) -> QuantumCircuit:
    
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

    # random sampling of unitaries
    lam = np.sum(np.abs(hamiltonian.coeffs))
    tau = total_simulation_time * lam / num_samples
    direct_access_table_of_unitaries = [] # this is so we don't have to recompute the same matrix exponential multiple times

    # Note: the direct_access_table optimization was not incorporated before bcause a) I didn't think of it 
    # and b) if we only sample a few unitaries (or just one), the overhead of computing the matrix exponentials is not that high.

    for coeff, pauli in zip(hamiltonian.coeffs, hamiltonian.paulis.to_labels()):
        if coeff < 0:
            h_j = SparsePauliOp([pauli], [-1.0])
        else:
            h_j = SparsePauliOp([pauli], [1.0])
        unitary = exponentiate_hamiltonian(h_j, tau)
        direct_access_table_of_unitaries.append(unitary)
    
    qdrift_pmf = np.abs(hamiltonian.coeffs) / lam
    for k in tqdm(range(num_ancilla), desc="Applying qDRIFT unitaries"):
        for _ in range(2 ** k):
            idx = random.choices(population=range(len(direct_access_table_of_unitaries)), weights=qdrift_pmf, k=1)[0]
            unitary = direct_access_table_of_unitaries[idx]
            controlled_unitary = UnitaryGate(unitary, label=f"QDrift_{k}").control(1)
            qc.append(controlled_unitary, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))
    
    # Apply inverse QFT
    qc.append(QFT(num_ancilla, inverse=True), range(num_ancilla))

    # Measure the ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))

    return qc





