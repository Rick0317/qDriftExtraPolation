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


def babys_first_qpe(U: Union[UnitaryGate, np.array],
                    num_evals: int,
                    eigenstate: Optional[Union[QuantumCircuit, np.array]],
                    backend,
                    num_qubits=2,
                    print_results=False) -> QuantumCircuit:
    # Data type management:
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
        # the eigenstate is a vector of complex numbers that determine the state of the qubits other than the control qubit
        eigenstate_circuit.initialize(eigenstate)
    else:
        eigenstate_circuit = eigenstate

    # Actually create the circuit
    qc = QuantumCircuit(num_qubits + 1, 1)
    qc.append(eigenstate_circuit,
              range(1, num_qubits + 1))  # Apply the eigenstate to the qubits
    qc.h(0)  # Apply Hadamard to the control qubit
    qc.s(0)
    u = U.control(1)  # Create the controlled-U gate
    qc.append(u, range(num_qubits + 1))  # Apply the controlled-U gate
    qc.h(0)  # Apply Hadamard to the control qubit
    qc.measure(0, cbit=0)  # Measure the control qubit
    qc.barrier()

    return qc


def generate_random_hamiltonian(num_qubits, num_terms):
    # Generate a random Pauli string
    # pauli_matrices = [Pauli('X'), Pauli('Y'), Pauli('Z'), Pauli('I')]
    pauli_matrices = [Pauli('X'), Pauli('Z'), Pauli('I')]
    hamiltonian_terms = []
    for _ in range(num_terms):
        pauli_string = reduce(lambda x, y: x.tensor(y),
                              random.choices(pauli_matrices, k=num_qubits))

        # Weights have to be positive
        hamiltonian_terms.append((random.uniform(0, 1), pauli_string))
    return hamiltonian_terms


def test_qpe():
    print("Test QPE")
    coeff = random.uniform(0, 1)
    H = coeff * (Pauli(('Z')).tensor(Pauli('Z'))).to_matrix()

    eigenvalues, eigenvectors = scipy.linalg.eigh(H)

    eigenstate = eigenvectors[:, np.argmin(eigenvalues)]  # for small eigenvalue

    print(f"Correct phase: {eigenvalues[0]}")


    tau = coeff

    v = scipy.linalg.expm(1j * tau * (Pauli(('Z')).tensor(Pauli('Z'))).to_matrix())

    backend = Aer.get_backend('qasm_simulator')

    qc = babys_first_qpe(v, num_evals=1, eigenstate=eigenstate,
                         backend=backend)

    shots = 4096 * 128

    # simulate the circuit
    job = backend.run(transpile(qc, backend), shots=shots)
    result = job.result()
    counts = result.get_counts()
    print(counts["0"])

    # phase = 2 * np.arccos(np.sqrt(counts["0"] / shots))
    phase = np.arcsin(1 - 2 * counts['0'] / shots)
    print(f"Estimated phase: {phase}")


def test_babys_first_qpe():

    print("Test iterative qpe:")
    coeff = random.uniform(0, 1)
    H = coeff * (Pauli(('Z')).tensor(Pauli('Z'))).to_matrix()

    eigenvalues, eigenvectors = scipy.linalg.eigh(H)

    eigenstate = eigenvectors[:, np.argmin(eigenvalues)]  # for small eigenvalue
    print(f"Correct eigenvalue: {eigenvalues[0]}")


    num_samples = 100
    time = 1

    tau = time * coeff / num_samples

    v = scipy.linalg.expm(1j * tau * (Pauli(('Z')).tensor(Pauli('Z'))).to_matrix())

    backend = Aer.get_backend('qasm_simulator')
    results = []
    for i in range(num_samples):
        qc = babys_first_qpe(v, num_evals=1, eigenstate=eigenstate,
                             backend=backend)

        shots = 4096 * 128

        # simulate the circuit
        job = backend.run(transpile(qc, backend), shots=shots)
        result = job.result()
        counts = result.get_counts()
        results.append(np.arcsin(1 - 2 * counts['0'] / shots))

    estimated_eig = sum(results)

    print(f"Estimated eigenvalue: {estimated_eig}")

def test_qDrift_channel():

    print("Test qDrift Channel:")
    coeff1 = random.uniform(0, 1)
    coeff2 = random.uniform(0, 1)
    H = coeff1 * (Pauli(('Z')).tensor(Pauli(('X')))).to_matrix() + coeff2 * (Pauli(('X')).tensor(Pauli(('I')))).to_matrix()

    eigenvalues, eigenvectors = scipy.linalg.eigh(H)

    eigenstate = eigenvectors[:, np.argmin(eigenvalues)]  # for small eigenvalue
    print(f"Correct eigenvalue: {eigenvalues[0]}")
    print(f"Coeffs for (ZX, XI) : {coeff1}, {coeff2}")

    hamil_list = [(Pauli(('Z')).tensor(Pauli(('X')))), (Pauli(('X')).tensor(Pauli(('I'))))]


    num_samples = 100
    time = 1
    lam = coeff1 + coeff2


    backend = Aer.get_backend('qasm_simulator')
    results = []
    for i in range(num_samples):

        j = random.choices(range(2), weights=[abs(coeff1), abs(coeff2)],
                           k=1)[0]
        tau = time * lam / num_samples
        h_j = hamil_list[j]
        v = scipy.linalg.expm(1j * tau * h_j.to_matrix())
        qc = babys_first_qpe(v, num_evals=1, eigenstate=eigenstate,
                             backend=backend)

        shots = 4096 * 128

        # simulate the circuit
        job = backend.run(transpile(qc, backend), shots=shots)
        result = job.result()
        counts = result.get_counts()
        results.append(np.arcsin(1 - 2 * counts['0'] / shots))

    estimated_eig = sum(results)

    print(f"Estimated eigenvalue: {estimated_eig}")


def qdrift_channel(hamiltonian_terms, time, num_samples, eigenstate):
    '''
    For j = 1 to N_samples:
        draw H_j randomly
        ...
    '''
    for term in hamiltonian_terms:
        assert isinstance(term[1], Pauli)

    lam = sum(abs(term[0]) for term in hamiltonian_terms)
    tau = time * lam / num_samples
    print("Lambda:", lam)
    v_lst = []
    results = []
    hamiltonian_specific_pmf = [abs(coeff) for coeff, _ in
                                hamiltonian_terms]
    print("Prob weights:", hamiltonian_specific_pmf)
    backend = Aer.get_backend('qasm_simulator')
    num_terms = len(hamiltonian_terms)
    for i in range(num_samples):
        j = \
        random.choices(range(num_terms), weights=hamiltonian_specific_pmf, k=1)[
            0]
        h_j = hamiltonian_terms[j][1]
        v = scipy.linalg.expm(1j * tau * h_j.to_matrix())
        qc = babys_first_qpe(v, num_evals=1, eigenstate=eigenstate,
                             backend=backend)

        shots = 4096 * 128

        # simulate the circuit
        job = backend.run(transpile(qc, backend), shots=shots)
        result = job.result()
        counts = result.get_counts()
        results.append(np.arcsin(1 - 2 * counts["0"] / shots))

    return results, v_lst


if __name__ == "__main__":
    # test_qDrift_channel()
    # test_qDrift_channel()
    # Define the parameters
    num_experiements = 5
    filename = "qDrift_two_qubit_test.csv"
    for _ in range(num_experiements):
        num_qubits = 2
        num_terms = 6
        # Generate a random Hamiltonian
        hamiltonian = generate_random_hamiltonian(num_qubits, num_terms)

        H = sum(coeff * term.to_matrix() for coeff, term in hamiltonian)

        eigenvalues, eigenvectors = scipy.linalg.eigh(H)

        # while any(eigenval.real < 0 for eigenval in eigenvalues):
        #     new_hamiltonian = generate_random_hamiltonian(num_qubits, num_terms)
        #     H = sum(coeff * term.to_matrix() for coeff, term in new_hamiltonian)
        #     eigenvalues, eigenvectors = scipy.linalg.eigh(H)

        eigenstate = eigenvectors[:, np.argmin(eigenvalues)]  # for small eigenvalue

        print("Random Hamiltonian:")
        print(H)
        print(f"Terms: {[term for _, term in hamiltonian]}")
        print(f"Coefficients: {[coeff for coeff, _ in hamiltonian]}")
        print(f"Eigenstate:{eigenstate}")


        print("Eigenvalues:")
        print(sorted(eigenvalues))

        # Define the time and number of samples
        time = 1
        epsilon = 0.01
        num_samples = 100
        suggest_samples = (2 * sum((abs(term[0]) for term in hamiltonian)) ** 2 * time ** 2) / epsilon
        print(
            f"number of samples following the formula from the paper: {suggest_samples}")

        # Estimate the Baby's first phase
        qpe_results, v_lst = qdrift_channel(hamiltonian, time, num_samples,
                                            eigenstate)
        print(f"Estimated Eigenvalue:{sum(qpe_results)}")

        file_exists = os.path.isfile(filename)
        # Open the file in append mode or write mode
        with open(filename, mode='a' if file_exists else 'w', newline='',
                  encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header only if the file doesn't exist
            if not file_exists:
                writer.writerow(
                    ['Coefficients', 'Hamiltonian', 'Correct Eigenvalue',
                     "Estimated Eigenvalue", "Suggested Samples",
                     "Used Samples"])

            # Write the data
            writer.writerow([[coeff for coeff, term in hamiltonian],
                             [term for coeff, term in hamiltonian],
                             eigenvalues[0], sum(qpe_results),
                             suggest_samples, num_samples])
