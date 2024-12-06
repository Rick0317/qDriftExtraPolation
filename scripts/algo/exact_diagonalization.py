import numpy as np

a = np.array([[0, 1], [0, 0]])
a_dag = np.array([[0, 0], [1, 0]])
pauli_z = np.array([[1, 0], [0, -1]])


def a_j(j: int, n: int):
    tensor = 1
    if j == 0:
        tensor = a
        for i in range(n - 1):
            tensor = np.kron(np.eye(2), tensor)
        return tensor
    for _ in range(j):
        tensor = np.kron(pauli_z, tensor)

    tensor = np.kron(a, tensor)

    for _ in range(n - j - 1):
        tensor = np.kron(pauli_z, tensor)

    return tensor


def a_dag_j(j: int, n: int):
    tensor = 1
    if j == 0:
        tensor = a_dag
        for i in range(n - 1):
            tensor = np.kron(np.eye(2), tensor)
        return tensor
    for _ in range(j):
        tensor = np.kron(pauli_z, tensor)

    tensor = np.kron(a_dag, tensor)

    for _ in range(n - j - 1):
        tensor = np.kron(pauli_z, tensor)

    return tensor


def num_j(j: int, n: int):
    return a_dag_j(j, n) @ a_j(j, n)


def hubbard(n, t, U):
    total = 0
    for i in range(n - 1):
        total += -t * a_dag_j(i + 1, n) @ a_j(i, n)
        total += -t * a_dag_j(i, n) @ a_j(i + 1, n)
    for i in range(n):
        total += U * num_j(i, n)

    return total


if __name__ == '__main__':
    eig, eigv = np.linalg.eigh(hubbard(3, 2, 1))
