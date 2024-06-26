import Utils.fermion_util.openfermion_util as of_util
import openfermion as of
import numpy as np
import scipy


def ed_ground_state(obt, tbt, spin_orb=True):
    """
    Exact diagonalization to find the ground state and the ground state energy.
    :param obt: one-body tensor of the Hamiltonian
    :param tbt: two-body tensor of the Hamiltonian
    :param spin_orb: Whether it's in spin orbital or not
    :return: Ground state energy, ground state
    """
    of_hamil = (of_util.get_ferm_op_one(obt, spin_orb=spin_orb)
                + of_util.get_ferm_op_two(tbt, spin_orb=spin_orb))
    sparse_op = of.get_sparse_operator(of_hamil)
    gse, gs = of.get_ground_state(sparse_op)
    return gse, gs


if __name__ == "__main__":
    sample_obt = np.array([[1.0, 0.0], [0.0, 1.0]])
    tbt = np.zeros((2, 2, 2, 2))
    obt = of_util.get_ferm_op_one(sample_obt, spin_orb=True)
    print(obt)
    sparse_op = scipy.sparse.csr_matrix(sample_obt)
    print(sparse_op)
    gse, gs = of.linalg.get_ground_state(sparse_op)
    print(gse)
