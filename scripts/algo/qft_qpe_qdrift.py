from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, circuit_drawer
from typing import Optional, Union
import random
import numpy as np
from functools import reduce
import scipy.linalg
import pandas as pd
from qiskit.quantum_info import diamond_norm, SuperOp

# Type aliases
coefficient = float
Hamiltonian = list[tuple[coefficient, Pauli]]

# --------------------------------------------- Part 1: just qDRIFT ------------------------------------------------

# Function to sample unitaries from the qDRIFT distribution
def qdrift_sample(hamiltonian_terms: list[tuple[coefficient, Pauli]], time, num_samples, tau, total_N) -> tuple[list[Operator], list[str]]:
    lam = sum(abs(term[0]) for term in hamiltonian_terms)
    tau = lam * time / total_N
    sampled_unitaries = []
    hamiltonian_specific_pmf = [abs(coeff) for coeff, _ in hamiltonian_terms]
    num_terms = len(hamiltonian_terms)
    labels = []
    for _ in range(num_samples):
        j = random.choices(range(num_terms), weights=hamiltonian_specific_pmf, k=1)[0]
        h_j = hamiltonian_terms[j][1]
        v = scipy.linalg.expm(1j * tau * h_j.to_matrix())
        sampled_unitaries.append(v)
        # Format the label as LaTeX
        labels.append(f"$e^{{i\\cdot {float('%.1g' % tau)}\\cdot H_{j}}}$")
    return sampled_unitaries, labels

def estimate_theoretical_qdrift_errror(num_samples, lam, time):
    """
    Comparing E and UN , we see that the zeroth and first
    order terms match whenever τ = tλ/N. The higher order
    terms will not typically match and more careful analysis
    (see App. B) shows that the channels E and UN differ by
    an amount bounded by
    δ ≤(2λ^2t^2)/(N^2) * e^{2λt/N }

    This is the error for a single random operation. The error for the whole circuit is the sum of the errors for each
    random operation:
    error = Nδ \leq 2λ^2 t^2 / N .
. """
    return (2 * lam**2 * time**2 / num_samples) * np.exp(2 * lam * time / num_samples)


def exact_unitary_evolution(hamiltonian_terms: Hamiltonian, time: float) -> Operator:
    """
    Compute the exact unitary evolution U = exp(-iHt) for a given Hamiltonian.
    """
    H = sum(coeff * op.to_matrix() for coeff, op in hamiltonian_terms)
    U_exact = scipy.linalg.expm(1j * time * H)
    return Operator(U_exact)


def unitary_error_2_norm(U_exact: Operator, U_qdrift: Operator) -> float:
    """
    Compute the operator 2-norm difference between the exact unitary and qDRIFT evolution.
    """
    return np.linalg.norm(U_exact.data - U_qdrift.data, ord=2)  # Spectral norm

def unitary_error_inf_norm(U_exact: Operator, U_qdrift: Operator) -> float:
    """
    Compute the operator 2-norm difference between the exact unitary and qDRIFT evolution.
    """
    return np.linalg.norm(U_exact.data - U_qdrift.data, ord=np.inf)  # Spectral norm


def compute_diamond_distance(U_exact: Operator, U_qdrift: Operator) -> float:
    """
    Compute the diamond distance between the exact and qDRIFT quantum channels.
    """
    exact_channel = SuperOp(U_exact)
    qdrift_channel = SuperOp(U_qdrift)
    return diamond_norm(exact_channel - qdrift_channel)

# ----------------------------------------------- Part 2: QPE ------------------------------------------------------

def standard_qpe(unitary: Operator, eigenstate: QuantumCircuit, num_ancilla: int) -> QuantumCircuit:
    """Constructs a standard Quantum Phase Estimation (QPE) circuit using repeated controlled-U applications."""
    num_qubits = unitary.num_qubits
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    # Prepare eigenstate on system qubits
    qc.append(eigenstate, range(num_ancilla, num_ancilla + num_qubits))

    # Apply Hadamard gates to ancilla qubits
    # qc.append(QFT(num_ancilla), range(num_ancilla))
    qc.h(range(num_ancilla))

    # Apply controlled-U^(2^k) using repeated controlled applications of U
    for k in range(num_ancilla):
        control_qubit = num_ancilla - k - 1  # Reverse order
        controlled_U = UnitaryGate(unitary.data).control(1, label=f"U^{2 ** k}")
        for _ in range(2 ** k):
            qc.append(controlled_U, [control_qubit] + list(
                range(num_ancilla, num_ancilla + num_qubits)))

    # Apply inverse QFT on ancilla qubits
    qc.append(QFT(num_ancilla, inverse=True, do_swaps=False).decompose(),
              range(num_ancilla))

    # Measure ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))

    return qc



# ----------------------------------------------- Part 3: qDRIFT-based QPE ----------------------------------------

# Function to construct controlled unitaries
def construct_controlled_unitary(sampled_unitaries, labels):
    controlled_unitaries = []
    for unitary, label in zip(sampled_unitaries, labels):
        controlled_unitary = UnitaryGate(unitary, label=label).control(1)
        controlled_unitaries.append(controlled_unitary)
    return controlled_unitaries

# Function to perform qDRIFT-based QPE
def qdrift_qpe(hamiltonian_terms, time, eigenstate, num_qubits, num_ancilla, num_samples, tau):
    # Initialize the quantum circuit
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    # Prepare the eigenstate on the system qubits
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circuit = QuantumCircuit(num_qubits, name=f'eigenstate')
        eigenstate_circuit.initialize(eigenstate)
    else:
        eigenstate_circuit = eigenstate

    qc.append(eigenstate_circuit, range(num_ancilla, num_ancilla + num_qubits))

    # Apply QFT to the ancilla qubits
    # qc.append(QFT(num_ancilla), range(num_ancilla ))


    qc.h(range(num_ancilla))



    # Perform controlled qDRIFT unitaries
    for k in range(num_ancilla):
        control_qubit = num_ancilla - k - 1
        for _ in range(2**k):
            # # Apply controlled unitaries
            # Sample unitaries from qDRIFT distribution
            sampled_unitaries, labels = qdrift_sample(
                hamiltonian_terms=hamiltonian_terms, time=time, num_samples=1,
                tau=tau, total_N=num_samples)
            # Construct controlled unitaries
            controlled_unitaries = construct_controlled_unitary(
                sampled_unitaries,
                labels)
            for controlled_unitary in controlled_unitaries:
                # apply controlled unitary such that the control qubit is ancilla qubit k
                qc.append(controlled_unitary, [control_qubit] + list(range(num_ancilla, num_ancilla + num_qubits)))
    # Apply inverse QFT
    qc.append(QFT(num_ancilla, inverse=True, do_swaps=False).decompose(),
              range(num_ancilla))
    # Measure the ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))

    return qc

