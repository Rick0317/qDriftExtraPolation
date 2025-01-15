from scripts.algo.chebyshev import chebyshev_nodes, chebyshev_barycentric_interp_point
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
from matplotlib import pyplot as plt



def babys_first_qpe(U: Union[UnitaryGate, np.array],
                    num_evals: int,
                    eigenstate: Optional[Union[QuantumCircuit, np.array]],
                    backend,
                    num_qubits,
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


def qdrift_channel(hamiltonian_terms, time, num_samples, eigenstate, shots, qubits):
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
                             backend=backend, num_qubits=qubits,)

        # simulate the circuit
        job = backend.run(transpile(qc, backend), shots=shots)
        result = job.result()
        counts = result.get_counts()
        if '0' in counts:
            results.append(np.arcsin(1 - 2 * counts["0"] / shots))

    return results, v_lst

def draw_plot(axs, idx, title:str, x:list[float], x_label:str, y:list[float], y_label:str):
    "Draw the subplot of title with the x-axis and the y-axis labeled as x-label and y-axis relatively."
    axs[idx[0], idx[1]].plot(x, y)
    axs[idx[0], idx[1]].set_xlabel(x_label)
    axs[idx[0], idx[1]].set_ylabel(y_label)
    axs[idx[0], idx[1]].set_title(title)


def extrapolation(time, hamiltonian, eigenstate, expected, shots, num_qubits, num_terms):

    fig, axs = plt.subplots(2, 2)  # Figure
    fig.set_figheight(10)
    fig.set_figwidth(10)
    domain = np.linspace(-1, 1, 200)  # Domain for extrapolation
    axs_x = 0
    axs_y = 0

    for n in range(2, 10, 2):  # number of the Chebyshev nodes
        data_point = []  # Storage for data points to be interpolated
        nodes = chebyshev_nodes(n)  # The n Chebyshev nodes

        for i in range(n // 2):
            node = nodes[i]
            num_samples = int(np.ceil(np.abs(time / node)))
            qpe_results, v_lst = qdrift_channel(hamiltonian, time, num_samples,
                                                eigenstate, shots, num_qubits)

            data_point.append(sum(qpe_results) / time)

        data_point.extend(data_point[::-1])

        # plot
        draw_plot(axs, (axs_x, axs_y),
                  f'Extrapolation with {len(nodes)} Chebyshev nodes',
                  domain, 'x',
                  [chebyshev_barycentric_interp_point(x, n, data_point) for
                   x in domain], 'estimation')
        axs[axs_x, axs_y].plot(nodes, data_point, 'bo')
        axs[axs_x, axs_y].axhline(expected, color='r', linestyle='--')
        axs_x = (axs_x + axs_y) % 2
        axs_y = (axs_y + 1) % 2

        print(f"Estimated eigenvalues: {data_point}")

        estimate_at_zero = chebyshev_barycentric_interp_point(0, n, data_point)

        data = [
            [num_qubits, num_terms, "Estimate", estimate_at_zero, n, time, shots]
        ]

        with open("extrapolation_result_symmetric_8nodes_10q.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

        # Save each plot as a file
        file_name = f'plot_node_{n}_{time}_{shots}.png'
        plt.savefig(file_name)
        print(f"Saved plot to {file_name}")


if __name__ == '__main__':

    num_qubits = 6

    num_terms = 2 * num_qubits
    # Generate a random Hamiltonian
    hamiltonian = generate_random_hamiltonian(num_qubits, num_terms)

    H = sum(coeff * term.to_matrix() for coeff, term in hamiltonian)

    eigenvalues, eigenvectors = scipy.linalg.eigh(H)

    eigenstate = eigenvectors[:, np.argmin(eigenvalues)]

    times = [100, 500, 1000, 2000, 5000]
    shots_list = [2048, 4096, 8192]
    for time in times:
        for shots in shots_list:

            print(f"Correct eigenvalue: {eigenvalues[0]}")
            data = [
                ["N Qubits", "N Terms", "Correct or Estimate", "Value",
                 "N Points", "time", "shots"],
                [num_qubits, num_terms, "Correct", eigenvalues[0], 0, time, 0]
            ]

            with open("extrapolation_result_symmetric_8nodes_10q.csv", mode="a",
                      newline="") as file:
                writer = csv.writer(file)
                writer.writerows(data)

            extrapolation(time, hamiltonian, eigenstate, eigenvalues[0], shots, num_qubits, num_terms)
