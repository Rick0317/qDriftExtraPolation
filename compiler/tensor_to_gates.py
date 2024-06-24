import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators import Operator, SparsePauliOp
import scipy

def unitary_to_unitary_gate(unitary: np.ndarray):
    """
    Converts a tensor to unitary gates
    :param tensor:
    :return:
    """
    unitary_gate = UnitaryGate(unitary)
    print(unitary_gate)
    return unitary_gate

def unitary_to_sparse_pauli(unitary: np.ndarray):
    """
    Converts a tensor to sparse pauli operator
    :param tensor:
    :return:
    """
    operator = Operator(unitary)
    return SparsePauliOp.from_operator(operator)


def matrix_to_sparse_pauli(tensor: np.ndarray):
    """
    Converts a matrix to sparse pauli operator
    :param tensor:
    :return:
    """
    unitary = scipy.linalg.expm(tensor)
    return unitary_to_sparse_pauli(unitary)
