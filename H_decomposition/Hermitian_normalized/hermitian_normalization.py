from H_data_storage.data_interface import TensorData, DecomposedHamiltonian
import typing


def hermitian_normalized_decomposition(Hamiltonian):
    """
    Hermitian normalized decomposition. We decompose the Hamiltonian so that each term is Hermitian and its largest
    singular value is 1. They also need to be so that e^{itH_j} can be implemented on quantum hardware for any t.
    :param Hamiltonian: The Hamiltonian to decompose
    :return:
    """
    if Hamiltonian is TensorData:
        return Hamiltonian.decompose()

    # TODO: Implement other cases
