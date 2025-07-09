from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit_aer import Aer
from typing import Optional, Union
import random
from functools import reduce
import numpy as np
from qiskit import QuantumCircuit
import scipy
import scipy.linalg
import os
import csv


def babys_first_qpe(U: Union[UnitaryGate, np.array], eigenstate: Optional[Union[QuantumCircuit, np.array]],
                    num_qubits: int) -> QuantumCircuit:
    """
    Returns the QuantumCircuit of the baby's first QPE

    Args:
        U: abstraction of U gate
        eigenstate: eigenstate of U
        num_qubits: number of qubits in the circuit

    Returns:
        the QuantumCircuit of the baby's first QPE
    """

    # Convert U to a gate if it's a numpy array
    if isinstance(U, np.ndarray):
        U = UnitaryGate(U)

    elif isinstance(U, SparsePauliOp):
        op = Operator(U)
        U = UnitaryGate(op)

    elif isinstance(U, Operator):
        U = UnitaryGate(U)

    # Prepare the initial eigenstate
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circuit = QuantumCircuit(num_qubits, name='eigenstate')
        eigenstate_circuit.initialize(eigenstate)
    else:
        eigenstate_circuit = eigenstate

    # Create the circuit
    qc = QuantumCircuit(num_qubits + 1, 1)

    # Apply the eigenstates
    qc.append(eigenstate_circuit, range(1, num_qubits + 1))

    # Hadamard gate to the controlled qubit
    qc.h(0)
    qc.s(0)

    # Create and apply the controlled U gate
    u = U.control(1)
    qc.append(u, range(num_qubits + 1))

    # Hadamard gate to the controlled qubit
    qc.h(0)

    # Measure the controlled qubit
    qc.measure(0, cbit=0)
    qc.barrier()

    return qc


def random_pauli_string(num_qubits, num_terms):
    """
    Generate randomly a list of pauli operators
    Args:
        num_qubits: the number of qubits
        num_terms: the desired number of pauli operators

    Returns:
        a list of random pauli operators
    """
    # pauli_matrices = [Pauli('X'), Pauli('Y'), Pauli('Z'), Pauli('I')]
    pauli_matrices = [Pauli('X'), Pauli('Z'), Pauli('I')]
    hamiltonian_terms = []
    for _ in range(num_terms):
        pauli_string = reduce(lambda x, y: x.tensor(y),
                              random.choices(pauli_matrices, k=num_qubits))

        # Weights have to be positive
        hamiltonian_terms.append((random.uniform(0, 1), pauli_string))
    return hamiltonian_terms


def qdrift_qpe(pauli_list, time, num_samples, num_qubits, eigenstate):
    """
    Return the result of the QPE algorithm on the qdrift channel sampled from hamiltonian_terms
    with the given parameters for the QPE circuit

    Args:
        pauli_list: a list of pauli operators
        time: time of the QPE algorithm
        num_samples: number of samples in the qdrift channel
        num_qubits: number of qubits in the circuit
        eigenstate: an eigenstate for the QPE algorithm

    Returns:
        the result of quantum phase estimation
    """

    '''
    For j = 1 to N_samples:
        draw H_j randomly
        ...
    '''
    for term in pauli_list:
        assert isinstance(term[1], Pauli)

    lam = sum(abs(term[0]) for term in pauli_list)
    tau = time * lam / num_samples
    v_lst = []
    results = []
    hamiltonian_specific_pmf = [abs(coeff) for coeff, _ in pauli_list]
    # print("Lambda:", lam)
    # print("Prob weights:", hamiltonian_specific_pmf)
    backend = Aer.get_backend('qasm_simulator')
    num_terms = len(pauli_list)
    for i in range(num_samples):
        j = random.choices(range(num_terms), weights=hamiltonian_specific_pmf, k=1)[0]
        h_j = pauli_list[j][1]
        v = scipy.linalg.expm(1j * tau * h_j.to_matrix())
        qc = babys_first_qpe(v, eigenstate=eigenstate,
                             num_qubits=num_qubits)

        shots = 4096 * 128

        # simulate the circuit
        job = backend.run(transpile(qc, backend), shots=shots)
        result = job.result()
        counts = result.get_counts()
        results.append(np.arcsin(1 - 2 * counts["0"] / shots))

    return results, v_lst


if __name__ == '__main__':

    qubits_trials = [4, 5]
    for num_qubits in qubits_trials:
        num_experiements = 5
        filename = "qDrift_n_qubit_test.csv"
        for _ in range(num_experiements):
            num_terms = 2 * num_qubits
            # Generate a random Hamiltonian
            hamiltonian = random_pauli_string(num_qubits, num_terms)

            H = sum(coeff * term.to_matrix() for coeff, term in hamiltonian)

            eigenvalues, eigenvectors = scipy.linalg.eigh(H)

            eigenstate = eigenvectors[:, np.argmin(eigenvalues)]
            print(f"Num Qubits: {num_qubits}")
            print(f"Num Terms: {num_terms}")
            print(f"Terms: {[term for _, term in hamiltonian]}")
            print(f"Coefficients: {[coeff for coeff, _ in hamiltonian]}")
            print(f"Correct Eig:{eigenvalues[0]}")

            # Define the time and number of samples
            time = 1
            epsilon = 0.01
            num_samples = 100
            suggest_samples = (2 * sum((abs(term[0]) for term in hamiltonian)) ** 2 * time ** 2) / epsilon
            print(
                f"number of samples following the formula from the paper: {suggest_samples}")

            # Estimate the Baby's first phase
            qpe_results, v_lst = qdrift_qpe(hamiltonian, time, num_samples,
                                                num_qubits, eigenstate)
            print(f"Estimated Eigenvalue:{sum(qpe_results)}")

            file_exists = os.path.isfile(filename)
            # Open the file in append mode or write mode
            with open(filename, mode='a' if file_exists else 'w', newline='',
                      encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write the header only if the file doesn't exist
                if not file_exists:
                    writer.writerow(
                        ['N-Qubits', 'N-Terms', 'Coefficients', 'Hamiltonian', 'Correct Eigenvalue',
                         "Estimated Eigenvalue", "Suggested Samples",
                         "Used Samples"])

                # Write the data
                writer.writerow([num_qubits, num_terms, [coeff for coeff, term in hamiltonian],
                                 [term for coeff, term in hamiltonian],
                                 eigenvalues[0], sum(qpe_results),
                                 suggest_samples, num_samples])

