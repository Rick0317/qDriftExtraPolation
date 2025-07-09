from testing_things_properly.qft_qpe.algos import generate_ising_hamiltonian
import numpy as np

if __name__ == '__main__':
    J = 1
    G = 0.8
    NUM_QUBITS = 2
    Jt = np.sqrt((np.pi / (4 * 0.01)) ** 2 - 4 * G ** 2) / 2
    print(Jt)
    H = generate_ising_hamiltonian(NUM_QUBITS, Jt, G)
    matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print(eigenvalues)

    # Let's do the ground state
    first_positive_eigenvalue = min(eigenvalues[eigenvalues > 0])
    eigenvector_index = np.where(eigenvalues == first_positive_eigenvalue)[0][0]
    eigenstate = eigenvectors[:, eigenvector_index]


    # Expected phase calculation
    expected_phase = (first_positive_eigenvalue.real * 0.01) / (2 * np.pi) % 1

    print(expected_phase)


