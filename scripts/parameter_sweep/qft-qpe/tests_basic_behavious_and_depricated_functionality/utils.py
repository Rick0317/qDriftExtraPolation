import numpy as np
import scipy.linalg
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli
from qiskit.circuit import QuantumCircuit, Gate
from sympy import symbols, I, exp, latex, simplify, Matrix
from IPython.display import display, Math
from typing import List, Tuple
import matplotlib.pyplot as plt


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


def format_coefficient(coeff) -> str:
    """
    Formats the coefficient to omit the imaginary part if it is zero.
    """
    # Extract real and imaginary parts
    real_part = coeff.real
    imag_part = coeff.imag

    # Check for zero imaginary part
    if imag_part == 0:
        return f"{real_part}"
    else:
        # Use latex to handle complex numbers
        return latex(coeff)


def hamiltonian_explicit(H: SparsePauliOp) -> str:
    """
    Construct the Hamiltonian with full tensor product notation, omitting zero imaginary components.
    """
    pauli_terms = H.to_list()
    explicit_terms = []

    for pauli_str, coeff in pauli_terms:
        # Format the coefficient, omitting + 0j
        formatted_coeff = format_coefficient(coeff)
        
        # Construct the tensor product string
        tensor_product = pauli_term_to_tensor_product(pauli_str)
        
        # Construct the term explicitly
        term_expr = f"{formatted_coeff} \\cdot {tensor_product}"
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
    H_sym = simplify(H_sym)
    
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


def plot_complex_unit_circle(estimated_phase, expected_phase, num_ancilla, time, filename):
    """
    Plots the complex unit circle with markers for the estimated and expected phases.

    Parameters:
    - estimated_phase: The phase derived from the measured bitstring.
    - expected_phase: The expected phase calculated from the known eigenvalue.
    - num_ancilla: Number of ancilla qubits used in QPE.
    - time: Evolution time.
    - filename: Path to save the plot.
    """

    # Prepare the complex unit circle
    angles = np.linspace(0, 2 * np.pi, 1000)
    unit_circle = np.exp(1j * angles)

    # Calculate the angle positions for expected and estimated phases
    est_angle = 2 * np.pi * estimated_phase
    exp_angle = 2 * np.pi * expected_phase

    # Complex points on the unit circle
    est_point = np.exp(1j * est_angle)
    exp_point = np.exp(1j * exp_angle)

    # Plot unit circle with shading
    plt.figure(figsize=(8, 8))
    plt.fill_between(np.real(unit_circle), np.imag(unit_circle), where=(np.imag(unit_circle) >= 0), color='red', alpha=0.1)
    plt.fill_between(np.real(unit_circle), np.imag(unit_circle), where=(np.imag(unit_circle) < 0), color='blue', alpha=0.1)

    # Plot unit circle boundary
    plt.plot(np.real(unit_circle), np.imag(unit_circle), label="Unit Circle", color='gray', linestyle='--')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # Plot expected phase
    plt.plot([0, np.real(exp_point)], [0, np.imag(exp_point)], 'darkblue', label=f"Expected Phase ({expected_phase:.3f})", linewidth=1.5)
    plt.plot(np.real(exp_point), np.imag(exp_point), 'bo', label="Expected Point")

    # Plot estimated phase
    plt.plot([0, np.real(est_point)], [0, np.imag(est_point)], 'darkred', label=f"Estimated Phase ({estimated_phase:.3f})", linewidth=1.5)
    plt.plot(np.real(est_point), np.imag(est_point), 'ro', label="Estimated Point")

    # Annotate angles in radians
    plt.text(1.1, 0, "0 / 2π", fontsize=10, ha='center')
    plt.text(-1.1, 0, "π", fontsize=10, ha='center')
    plt.text(0, 1.1, "π/2", fontsize=10, va='center')
    plt.text(0, -1.1, "3π/2", fontsize=10, va='center')

    # Add legend for shading
    plt.plot([], [], color='red', alpha=0.1, label="Positive Eigenvalue Zone")
    plt.plot([], [], color='blue', alpha=0.1, label="Negative Eigenvalue Zone")

    # Set plot limits and title
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Complex Unit Circle - Time: {time}, Ancilla: {num_ancilla}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(filename)
    plt.close()
