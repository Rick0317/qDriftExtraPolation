from openfermion import FermionOperator, hermitian_conjugated, normal_ordered


def get_ferm_op_one(obt, spin_orb: bool):
    """
    Return the corresponding FermionOperator for obt in chemist notation
    :param obt: Iterable tensor
    :param spin_orb: whether it's in spin orbital or not.
    :return:
    """
    n = obt.shape[0]
    op = FermionOperator.zero()
    for p in range(n):
        for q in range(n):
            if not spin_orb:
                for a in range(2):
                    op += FermionOperator(
                        term=(
                            (2*p+a, 1), (2*q+a, 0)
                        ), coefficient=obt[p, q]
                    )
            else:
                op += FermionOperator(
                    term=(
                        (q, 1), (q, 0)
                    ), coefficient=obt[p, q]
                )
    return op


def get_ferm_op_two(tbt, spin_orb: bool):
    """
    Return the corresponding FermionOperator for tbt in chemist notation
    :param tbt: Iterable tensor
    :param spin_orb: whether it's in spin orbital or not.
    :return:
    """
    n = tbt.shape[0]
    op = FermionOperator.zero()
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    if not spin_orb:
                        for a in range(2):
                            for b in range(2):
                                op += FermionOperator(
                                    term=(
                                        (2*p+a, 1), (2*q+a, 0),
                                        (2*r+b, 1), (2*s+b, 0)
                                    ), coefficient=tbt[p, q, r, s]
                                )
                    else:
                        op += FermionOperator(
                            term=(
                                (p, 1), (q, 0),
                                (r, 1), (s, 0)
                            ), coefficient=tbt[p, q, r, s]
                        )
    return op
