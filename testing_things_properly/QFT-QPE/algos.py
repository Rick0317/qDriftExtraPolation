import matplotlib.pyplot as plt
import numpy as np
import math

from qiskit import transpile
from qiskit_aer import AerSimulator  # as of 25Mar2025
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, UnitaryGate, PhaseGate, RZGate
from qiskit.quantum_info import Operator

# import basic plot tools
from qiskit.visualization import plot_histogram

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