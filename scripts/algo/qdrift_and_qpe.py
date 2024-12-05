# This is totally reinventing the wheel btw; I did not use the API we had previously defined, but it's a good start

# Baby's first phase estimation + qDRIFT
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


def babys_first_qpe(U: Union[UnitaryGate, np.array], 
                               num_evals: int, 
                               eigenstate: Optional[Union[QuantumCircuit, np.array]],
                               backend,
                               num_qubits=1,
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
    qc = QuantumCircuit(num_qubits + 1,1)
    qc.append(eigenstate_circuit, range(1, num_qubits + 1)) # Apply the eigenstate to the qubits
    qc.h(0) # Apply Hadamard to the control qubit
    u = U.control(1) # Create the controlled-U gate
    qc.append(u, range(num_qubits + 1)) # Apply the controlled-U gate
    qc.h(0) # Apply Hadamard to the control qubit
    qc.measure(0, cbit=0) # Measure the control qubit
    qc.barrier()
    
    return qc



def generate_random_hamiltonian(num_qubits, num_terms):
    # Generate a random Pauli string
    pauli_matrices = [Pauli('X'), Pauli('Y'), Pauli('Z'), Pauli('I')]
    hamiltonian_terms = []
    for _ in range(num_terms):
        pauli_string = reduce(lambda x, y: x.tensor(y), random.choices(pauli_matrices, k=num_qubits))
        hamiltonian_terms.append((random.uniform(-1, 1), pauli_string))
    return hamiltonian_terms

def qdrift_channel(hamiltonian_terms, time, num_samples, eigenstate):
    '''
    For j = 1 to N_samples:
        draw H_j randomly
        ...
    '''
    for term in hamiltonian_terms:
        assert isinstance(term[1], Pauli)

    l = sum(abs(term[0]) for term in hamiltonian_terms)
    tau = time * l / num_samples
    v_lst = []
    results = []
    hamiltonian_specific_pmf = [abs(coeff) / l for coeff, _ in hamiltonian_terms]
    backend = Aer.get_backend('qasm_simulator')
    for i in range(num_samples):
        j = random.choices(range(num_terms), weights=hamiltonian_specific_pmf, k=1)[0]
        h_j = hamiltonian_terms[j][1]

        # perform phase estimation
        qpe_circuit = QuantumCircuit(h_j.num_qubits + 1)
        qpe_circuit.h(0)
        v = scipy.linalg.expm(-1j * tau * h_j.to_matrix())

        # exact_eigenvalue, eigenvectors= scipy.linalg.eig(v)
        # eigenstate = eigenvectors[:, np.argmin(exact_eigenvalue)] # this is kinda arbitray

        v_lst.append(v)
        qc = babys_first_qpe(v, num_evals=1, eigenstate=eigenstate, backend=backend)

        # simulate the circuit
        job = backend.run(transpile(qc, backend), shots=1)
        result = job.result()
        counts = result.get_counts()
        if "0" in counts and counts["0"] > 0:
            results.append(0)
        elif "1" in counts and counts["1"] > 0:
            results.append(1)

    return results, v_lst



if __name__ == "__main__":
    # Define the parameters
    num_qubits = 1
    num_terms = 20000
    # Generate a random Hamiltonian
    hamiltonian = generate_random_hamiltonian(num_qubits, num_terms)

    H = sum(coeff * term.to_matrix() for coeff, term in hamiltonian)

    eigenvalues, eigenvectors = scipy.linalg.eig(H)

    while any(eigenval.real < 0 for eigenval in eigenvalues):
        new_hamiltonian = generate_random_hamiltonian(num_qubits, num_terms)
        H = sum(coeff * term.to_matrix() for coeff, term in new_hamiltonian)
        eigenvalues, eigenvectors = scipy.linalg.eig(H)

    eigenstate = eigenvectors[:, np.argmin(eigenvalues)] # for small eigenvalue


    print("Random Hamiltonian:")
    print(H)

    print("Eigenvalues:")
    print(sorted(eigenvalues))

    # Define the time and number of samples
    time = 1000
    num_samples = 10000
    epsilon = 0.01
    print(f"number of samples following the formula from the paper: {(2 * sum((abs(term[0]) for term in hamiltonian)) ** 2 * time **2)/epsilon}")
    
    # Estimate the Baby's first phase
    qpe_results, v_lst = qdrift_channel(hamiltonian, time, num_samples, eigenstate)
    print(f"Estimated Eigenvalue:{2 * np.arccos(np.sqrt(1 - (sum(qpe_results)/num_samples) ))}")
