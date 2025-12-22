"""
DEPRECATED: This module has been merged into visualization_utils.py

All functions from this module are now available in visualization_utils.py
Please update your imports:
    
    OLD: from src.utils.utils_visualization import hamiltonian_explicit
    NEW: from src.utils.visualization_utils import hamiltonian_explicit

This file is kept for backward compatibility and will be removed in a future version.
"""

import warnings
from src.utils.visualization_utils import (
    format_hamiltonian_matrix_as_latex,
    pauli_term_to_tensor_product,
    pauli_term_to_simplified,
    hamiltonian_explicit,
    hamiltonian_simplified,
    display_hamiltonian
)

warnings.warn(
    "utils_visualization is deprecated. Use visualization_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility
__all__ = [
    'format_hamiltonian_matrix_as_latex',
    'pauli_term_to_tensor_product',
    'pauli_term_to_simplified',
    'hamiltonian_explicit',
    'hamiltonian_simplified',
    'display_hamiltonian'
]
