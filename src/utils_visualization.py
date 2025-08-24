import numpy as np
from sympy import symbols, I, exp, latex, simplify, Matrix
from IPython.display import display, Math
from qiskit.quantum_info import SparsePauliOp

def format_hamiltonian_matrix_as_latex(matrix: np.ndarray):
    """Formats a matrix as LaTeX."""
    sympy_matrix = Matrix(matrix)
    return latex(sympy_matrix)

def pauli_term_to_tensor_product(pauli_str: str) -> str:
    """
    Converts a Pauli string (e.g., 'XI') to an explicit tensor product representation.
    """
    pauli_symbols = {
        'I': symbols('I'),
        'X': symbols('X'),
        'Y': symbols('Y'),
        'Z': symbols('Z')
    }
    
    # Construct the tensor product explicitly
    term_expr = []
    for char in pauli_str:
        term_expr.append(pauli_symbols[char])
    
    # Format as tensor product
    tensor_expr = ' \\otimes '.join([latex(term) for term in term_expr])
    
    return tensor_expr


def pauli_term_to_simplified(pauli_str: str) -> str:
    """
    Converts a Pauli string (e.g., 'XI') to a simplified symbolic form.
    """
    pauli_symbols = {
        'I': symbols('I'),
        'X': symbols('X'),
        'Y': symbols('Y'),
        'Z': symbols('Z')
    }
    
    # Construct the product expression (compact form)
    term_expr = 1
    for char in pauli_str:
        term_expr *= pauli_symbols[char]
    
    return term_expr


def hamiltonian_explicit(H: SparsePauliOp) -> str:
    """
    Construct the Hamiltonian with full tensor product notation.
    """
    pauli_terms = H.to_list()
    explicit_terms = []

    for pauli_str, coeff in pauli_terms:
        tensor_product = pauli_term_to_tensor_product(pauli_str)
        # Construct the term explicitly
        term_expr = f"{latex(coeff)} \\cdot {tensor_product}"
        explicit_terms.append(term_expr)
    
    # Join all terms into a single LaTeX string
    explicit_hamiltonian = ' + '.join(explicit_terms)
    return explicit_hamiltonian


def hamiltonian_simplified(H: SparsePauliOp) -> str:
    """
    Construct the Hamiltonian in a simplified symbolic form.
    """
    pauli_terms = H.to_list()
    H_sym = 0

    for pauli_str, coeff in pauli_terms:
        # Construct each term in simplified form
        term_expr = coeff * pauli_term_to_simplified(pauli_str)
        H_sym += term_expr
    
    # Simplify the expression to make it more readable
    # H_sym = simplify(H_sym)
    
    # Return the LaTeX string representation
    return latex(H_sym)

def format_hamiltonian_matrix_as_latex(matrix: np.ndarray):
    """Formats a matrix as LaTeX."""
    sympy_matrix = Matrix(matrix)
    return latex(sympy_matrix)

def display_hamiltonian(H: SparsePauliOp):
    """Display both the explicit tensor product form and the simplified form of the Hamiltonian."""
    
    # Explicit tensor product form
    explicit_latex = hamiltonian_explicit(H)
    display(Math(f"H = {explicit_latex}"))
    
    # Simplified form
    simplified_latex = hamiltonian_simplified(H)
    display(Math(f"H = {simplified_latex}"))

    # matrx representation
    matrix = H.to_matrix()
    matrix_latex = format_hamiltonian_matrix_as_latex(matrix)
    display(Math(f"H = {matrix_latex}"))
