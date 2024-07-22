import numpy as np


class Tensor:
    """
    Tensor object for generalization of every Hamiltonian term.
    """

    coefficient = 0
    matrix = np.array(list())
    matrix_1d = list()

    def __init__(self, coefficient, matrix, matrix_1d):
        self.coefficient = coefficient
        self.matrix = matrix
        self.matrix_1d = matrix_1d


class DecomposedHamiltonian:
    """
    Hamiltonian object for decomposed Hamiltonian

    Attributes:
        sum_coeff (float): the sum of the coefficients
        lst_Hamil: the parallel list of Hermitians
        mapping (Dict[str, tuple[float, TensorData]]): mapping from the name of a term to the tuple of a coefficient and a Hamiltonian
    """

    def __init__(self, names: list[str], term_list: list[Tensor]):
        self.sum_coeff = 0
        self.sum_abs_coeff = 0
        self.lst_Hamil = term_list
        self.mapping = {}
        for i, name in enumerate(names):
            assert name not in self.mapping, "duplicate name error"
            self.mapping[name] = (term_list[i])
            self.sum_coeff += term_list[i].coefficient
            self.sum_abs_coeff += np.abs(term_list[i].coefficient)

    def get_term(self, name: str) -> Tensor:
        """Return the Hamiltonian term corresponding to the name

        Args:
            name: the name of the Hamiltonian of request

        Returns: the Tensor object of the Hamiltonian if the name is valid, None otherwise
        """
        if name in self.mapping:
            return self.mapping[name]


class Hamiltonian(Tensor):
    """
    The unitary hamiltonian class to store its relevant properties.

    decomp Dict[str, tuple(int, Hamiltonian)]: a mapping from the name of each term to the strength and the Hamiltonian that decomposes the Hamiltonian
    """

    decomp = None

    def __init__(self, coefficient, matrix, matrix_1d):
        """Initialize the Hamiltonian instance with its coefficient, and corresponding representations in
        1D tensor and multidimensional tensor.
        """
        super().__init__(coefficient, matrix, matrix_1d)

    def set_decomp(self, names: list[str], terms: list[Tensor]) -> None:
        '''
        Decompose this Hamiltonian object to store its relevant properties.
        By its format, the coefficient and the tensor data already satisfy
        the definition of the decomposed Hamiltonian. It initializes the decomp attribute.

        :names: the list of names of the terms
        :terms: the list of sub-Hamiltonian terms
        '''

        decomposed_hamiltonian = DecomposedHamiltonian(names, terms)
        self.decomp = decomposed_hamiltonian

    def get_decomp(self) -> DecomposedHamiltonian:
        assert self.decomp is not None, "no decomposition found (call .decompose first)"
        return self.decomp


class Hubbard(Hamiltonian):
    """The Hubbard model

    Attributes:
        spatial_orb (int): spatial orbital
    """

    def __init__(self, spatial_orb: int):
        self.spatial_orb = spatial_orb
        super().__init__(1, None, list())

    def make_t_term(self, t):
        """
        Prepares the t term of the Hubbard Hamiltonian.
        :param t: The strength of the t term
        :return: the t term tensor of the Hamiltonian and the 1d version of it.
        """
        n = self.spatial_orb
        tensor = np.zeros((n, n))

        one_body_1d = []
        for p in range(n - 1):
            tensor[p+1, p] = -t
            tensor[p, p+1] = -t
            one_body_1d.append((p+1, p, -t))
            one_body_1d.append((p, p+1, -t))

        return tensor, one_body_1d, -t

    def make_spin_t_term(self, t):
        """
        Prepares the t term of the Hubbard Hamiltonian.
        :param t: The strength of the t term
        :return: the t term tensor of the Hamiltonian and the 1d version of it.
        """
        n = self.spatial_orb
        tensor = np.zeros((2 * n, 2 * n))

        one_body_1d = []
        for i in range(n - 1):
            tensor[2 * i + 2, 2 * i] = -t
            tensor[2 * i + 3, 2 * i + 1] = -t
            tensor[2 * i, 2 * i + 2] = -t
            tensor[2 * i + 1, 2 * i + 3] = -t
            one_body_1d.append((2 * i + 2, 2 * i, -t))
            one_body_1d.append((2 * i + 3, 2 * i + 1, -t))
            one_body_1d.append((2 * i, 2 * i + 2, -t))
            one_body_1d.append((2 * i + 1, 2 * i + 3, -t))

        return tensor, one_body_1d, -t

    def make_u_term(self, u):
        """
        Prepares the U term of the Hubbard Hamiltonian.
        :param u: The strength of the u term\
        :return: The u term tensor of the Hubbard Hamilotnian and the 1D array version of the tensor.
        """
        n = self.spatial_orb
        tensor = np.zeros((n, n))

        two_body_1d = []
        for p in range(n):
            tensor[p, p] = u
            two_body_1d.append((p, p, u))

        return tensor, two_body_1d, u

    def make_spin_u_term(self, u):
        """
        Prepares the U term of the Hubbard Hamiltonian.
        :param u: The strength of the u term\
        :return: The u term tensor of the Hubbard Hamilotnian and the 1D array version of the tensor.
        """
        n = self.spatial_orb
        tensor = np.zeros((2 * n, 2 * n, 2 * n, 2 * n))

        two_body_1d = []
        for p in range(n):
            tensor[2 * p, 2 * p, 2 * p + 1, 2 * p + 1] = u
            two_body_1d.append((2 * p, 2 * p, 2 * p + 1, 2 * p + 1, u))

        return tensor, two_body_1d, u

    def decompose(self, t: float, u: float):
        """
        Decomposes the Hubbard Hamiltonian into one-body-term and two-body-term with given coefficients
        t and u respectively.
        Args:
            t: the coefficient of the one-body-term (hopping integral)
            u: the coefficient of the two-body-term (interaction)

        Returns:

        """
        obt_m, obt_1d, obt_c = self.make_t_term(t)
        tbt_m, tbt_1d, tbt_c = self.make_u_term(u)

        obt = Tensor(obt_c, obt_m, obt_1d)
        tbt = Tensor(tbt_c, tbt_m, tbt_1d)

        self.set_decomp(['obt', 'tbt'], [obt, tbt])
