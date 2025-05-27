from scripts.algo.qft_qpe_qdrift import qdrift_qpe
import numpy as np
from qiskit.quantum_info import Operator, Pauli
from qiskit.circuit.library import PhaseGate
from scripts.algo import standard_qpe
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
import math
from scipy.linalg import expm, eigh
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scripts.algo.chebyshev import chebyshev_nodes, chebyshev_barycentric_interp_point
import csv


def qDrift_QPE_w_samples(H_terms, time, ground_state, N, num_ancilla, num_samples, node):
    results = []
    for i in range(int(np.ceil(num_samples))):
        qc = qdrift_qpe(H_terms, time, ground_state, N, num_ancilla, num_samples, node)

        # Execute the circuit
        simulator = AerSimulator()
        job = simulator.run(transpile(qc, simulator), shots=shots)
        result = job.result()
        counts = result.get_counts(qc)

        # Analyze the full distribution
        total_shots = sum(counts.values())
        phase_estimates = []
        probabilities = []

        for bitstring, count in counts.items():
            # Convert bitstring to decimal and then to phase
            decimal = int(bitstring, 2) / (2 ** num_ancilla)
            phase = decimal
            probability = count / total_shots

            phase_estimates.append(phase)
            probabilities.append(probability)

        # Sort by probability for better visualization
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_phases = np.array(phase_estimates)[sorted_indices]
        sorted_probs = np.array(probabilities)[sorted_indices]

        # Calculate weighted average of phases
        weighted_phase = np.sum(sorted_phases * sorted_probs)

        # Convert phase to energy (since phase = -Et)
        estimated_energy = (-2 * np.pi * weighted_phase / time) % (
                    2 * np.pi / time)
        # Optional: shift to center around 0
        if estimated_energy > np.pi / time:
            estimated_energy -= 2 * np.pi / time
        results.append(-estimated_energy)

    return results


def draw_plot(axs, idx, title: str, x: list[float], x_label: str,
              y: list[float], y_label: str):
    "Draw the subplot of title with the x-axis and the y-axis labeled as x-label and y-axis relatively."
    axs[idx[0], idx[1]].plot(x, y)
    axs[idx[0], idx[1]].set_xlabel(x_label)
    axs[idx[0], idx[1]].set_ylabel(y_label)
    axs[idx[0], idx[1]].set_title(title)


def extrapolation_combined_algo(H_terms, eigenstate, num_qubits, num_terms,
                                expected, time):
    lam = sum(abs(term[0]) for term in H_terms)

    fig, axs = plt.subplots(2, 2)  # Figure
    fig.set_figheight(10)
    fig.set_figwidth(10)
    domain = np.linspace(-1, 1, 200)  # Domain for extrapolation
    axs_x = 0
    axs_y = 0

    for n in range(2, 10, 2):  # number of the Chebyshev nodes
        data_point = []  # Storage for data points to be interpolated
        nodes = chebyshev_nodes(n)

        for i in range(n // 2):
            node = nodes[i]
            num_samples = int(np.ceil(np.abs(lam * time / node)))
            qpe_results = qDrift_QPE_w_samples(H_terms, time,
                                     eigenstate, num_qubits, num_ancilla, num_samples, node)

            data_point.append(np.mean(qpe_results))

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
            [num_qubits, num_terms, "Estimate", estimate_at_zero, n, time,
             shots]
        ]

        with open("extrapolation_result_symmetric_fid095.csv", mode="a",
                  newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

        # Save each plot as a file
        file_name = f'plot_node_{n}_{time}_{shots}.png'
        plt.savefig(file_name)
        print(f"Saved plot to {file_name}")


if __name__ == '__main__':

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"qpe_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Parameters for the Transverse Ising model
    J = 0.6  # Coupling strength
    g = 0.1  # Transverse field strength
    t = 1.0  # Evolution time

    # Create the Hamiltonian terms
    z1z2 = Pauli('ZZ').to_matrix()
    x1 = Pauli('XI').to_matrix()
    x2 = Pauli('IX').to_matrix()

    N = 2
    num_ancilla = 6

    # Construct the full Hamiltonian: -J Z1Z2 + gX1 + gX2
    # H = -J * z1z2 + g * x1 + g * x2
    H = J * z1z2
    #H_terms = [(-J, Pauli('ZZ')), (g,  Pauli('XI')), (g, Pauli('IX')) ]
    H_terms = [(J, Pauli('ZZ'))]

    time = 1
    shots = 4096

    eigvals, eigvecs = eigh(H)  # Hermitian matrix decomposition
    ground_energy = eigvals[0]
    ground_state = eigvecs[:, 0]
    lam = sum(abs(term[0]) for term in H_terms)
    print(f"Lam: {lam}")

    print(f"ED Groundstate energy: {ground_energy}")

    node = 1
    num_samples = 1 # int(np.ceil(np.abs(lam * time / node)))
    results = qDrift_QPE_w_samples(H_terms, time, ground_state, N, num_ancilla, num_samples, node)
    print(np.mean(results))
    # extrapolation_combined_algo(H_terms, ground_state, N, len(H_terms), ground_energy, time)


    # print(N)
    # results = []
    #
    # for i in range(int(np.ceil(Num))):
    #
    #     qc = qdrift_qpe(H_terms, time, ground_state, N, num_ancilla)
    #
    #     # Execute the circuit
    #     simulator = AerSimulator()
    #     job = simulator.run(transpile(qc, simulator), shots=shots)
    #     result = job.result()
    #     counts = result.get_counts(qc)
    #
    #     # Analyze the full distribution
    #     total_shots = sum(counts.values())
    #     phase_estimates = []
    #     probabilities = []
    #
    #     for bitstring, count in counts.items():
    #         # Convert bitstring to decimal and then to phase
    #         decimal = int(bitstring, 2) / (2 ** num_ancilla)
    #         phase = 2 * math.pi * decimal
    #         probability = count / total_shots
    #
    #         phase_estimates.append(phase)
    #         probabilities.append(probability)
    #
    #     # Sort by probability for better visualization
    #     sorted_indices = np.argsort(probabilities)[::-1]
    #     sorted_phases = np.array(phase_estimates)[sorted_indices]
    #     sorted_probs = np.array(probabilities)[sorted_indices]
    #
    #     # Calculate weighted average of phases
    #     weighted_phase = np.sum(sorted_phases * sorted_probs)
    #
    #     # Convert phase to energy (since phase = -Et)
    #     estimated_energy = -weighted_phase / t
    #
    #     results.append(estimated_energy)
    #
    # # Save results
    # filename = f"output_n_paulis_N{N}_t{t}_a{num_ancilla}.png"
    # plot_histogram(counts).savefig(os.path.join(results_dir, filename))
    # print(f"Estimated energy: {np.mean(results):.4f}")
    # print(f"Exact ground state energy: {ground_energy:.4f}")


