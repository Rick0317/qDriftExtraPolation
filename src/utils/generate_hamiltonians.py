"""
Hamiltonian construction and utilities for quantum simulation experiments.

This module contains functions for creating various Hamiltonians used in 
quantum phase estimation and qDRIFT simulations.
"""
from __future__ import annotations
import warnings
from typing import Dict

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def create_h2_minimal_basis_hamiltonian() -> SparsePauliOp:
    """
    Create the H2 molecule Hamiltonian in minimal basis.
    
    Returns:
        SparsePauliOp: H2 Hamiltonian with precomputed coefficients
    """
    # Define the integrals
    h00 = h11 = -1.252477
    h22 = h33 = -0.475934
    h0110 = 0.674493
    h2332 = 0.697397
    h0220 = h0330 = h1221 = h1331 = 0.663472
    h0202 = h1313 = h0312 = h0132 = 0.181287

    # Calculate coefficients for each Pauli string
    coeffs = [
        # IIII
        0.5*(h00 + h11 + h22 + h33) + 0.25*(h0110 + h2332 + h0330 + h1221 + h0220 - h0202 + h1331 - h1313),
        # ZIII
        -0.5*h00 - 0.25*(h0110 + h0330 + h0220 - h0202),
        # IZII
        0.25*h0110,
        # IIZI
        -0.5*h22 - 0.25*(h2332 + h1221 + h0220 - h0202),
        # ZZII
        -0.5*h11 - 0.25*(h0110 + h1221 + h1331 - h1313),
        # ZIZI
        0.25*(h0220 - h0202),
        # IZIZ
        0.25*h2332,
        # XZXI
        0.125*(h0132 + h0312),
        # YZYI
        0.125*(h0132 + h0312),
        # ZZZI
        0.25*h1221,
        # ZIZZ
        0.25*(h1331 - h1313),
        # IZZZ
        -0.5*h33 - 0.25*(h2332 + h0330 + h1331 - h1313),
        # XZXZ
        0.125*(h0132 + h0312),
        # YZYZ
        0.125*(h0132 + h0312),
        # ZZZZ
        0.25*h0330
    ]

    pauli_strings = [
        "IIII", "ZIII", "IZII", "IIZI", "ZZII", "ZIZI", "IZIZ", 
        "XZXI", "YZYI", "ZZZI", "ZIZZ", "IZZZ", "XZXZ", "YZYZ", "ZZZZ"
    ]

    return SparsePauliOp(data=pauli_strings, coeffs=coeffs)


def generate_ising_hamiltonian(num_qubits: int, J, g) -> SparsePauliOp:
    z_terms = []
    z_coeffs = []
    
    # ZZ interaction terms
    for j in range(num_qubits):
        pauli_string = ['I'] * num_qubits
        pauli_string[j] = 'Z'
        pauli_string[(j + 1) % num_qubits] = 'Z'  # Periodic boundary conditions
        z_terms.append("".join(pauli_string))
        z_coeffs.append(-J)  # Coefficient for ZZ interaction

    x_terms = []
    x_coeffs = []
    
    # X field terms
    for j in range(num_qubits):
        pauli_string = ['I'] * num_qubits
        pauli_string[j] = 'X'
        x_terms.append("".join(pauli_string))
        x_coeffs.append(-g)  # Coefficient for X term

    # Combine the Z and X terms into a single Hamiltonian
    all_terms = z_terms + x_terms
    all_coeffs = z_coeffs + x_coeffs

    return SparsePauliOp(all_terms, coeffs=all_coeffs)


def calculate_minimum_evolution_time(hamiltonians: Dict[str, SparsePauliOp], 
                                   m: int) -> Dict[str, float]:
    """
    Calculate the minimum evolution time t for each Hamiltonian given m bits of precision.
    
    For QPE, we need to satisfy the constraint: λt > 1/(2^m * t)
    
    Rearranging: λt² > 1/2^m
    Therefore: t > sqrt(1/(2^m * λ))
    
    The minimum evolution time is: t_min = sqrt(1/(2^m * λ))
    
    Args:
        hamiltonians: Dict mapping names to SparsePauliOp objects
        m: Number of precision bits for QPE
        
    Returns:
        Dict mapping Hamiltonian names to minimum evolution times
    """
    if m <= 0:
        raise ValueError("Number of precision bits m must be positive")
    
    results = {}
    
    for name, hamiltonian in hamiltonians.items():
        if not isinstance(hamiltonian, SparsePauliOp):
            raise TypeError(f"Hamiltonian '{name}' must be a SparsePauliOp")
        
        # Calculate the largest eigenvalue magnitude (spectral norm)
        # For Pauli operators, this is the sum of absolute values of coefficients
        lambda_max = np.sum(np.abs(hamiltonian.coeffs))
        
        if lambda_max == 0:
            warnings.warn(f"Hamiltonian '{name}' has zero norm, setting t_min to infinity")
            results[name] = float('inf')
            continue
        
        # Calculate minimum evolution time: t_min = sqrt(1/(2^m * λ))
        t_min = np.sqrt(1.0 / (2**m * lambda_max))
        results[name] = float(t_min)
    
    return results

def get_hamiltonian_info(hamiltonian: SparsePauliOp) -> Dict[str, any]:
    """
    Extract useful information about a Hamiltonian.
    
    Args:
        hamiltonian: The Hamiltonian to analyze
        
    Returns:
        Dict containing eigenvalues, spectral norm, etc.
    """
    # Calculate eigenvalues
    eigvals = np.linalg.eigvals(hamiltonian.to_matrix()).real
    
    # Calculate spectral properties
    spectral_norm = np.sum(np.abs(hamiltonian.coeffs))
    
    info = {
        "num_qubits": hamiltonian.num_qubits,
        "num_terms": len(hamiltonian.coeffs),
        "eigenvalues": eigvals.tolist(),
        "ground_state_energy": float(np.min(eigvals)),
        "excited_state_energy": float(np.max(eigvals)),
        "spectral_norm": float(spectral_norm),
        "coefficients": hamiltonian.coeffs.tolist(),
        "pauli_strings": hamiltonian.paulis.to_labels()
    }
    
    return info


def calculate_minimum_evolution_time(hamiltonians: Dict[str, SparsePauliOp], 
                                   m: int) -> Dict[str, float]:
    """
    Calculate the minimum evolution time t for each Hamiltonian given m bits of precision.
    
    For QPE, we need to satisfy the constraint: λt > 1/(2^m * t)
    
    Rearranging: λt² > 1/2^m
    Therefore: t > sqrt(1/(2^m * λ))
    
    The minimum evolution time is: t_min = sqrt(1/(2^m * λ))
    
    Args:
        hamiltonians: Dict mapping names to SparsePauliOp objects
        m: Number of precision bits for QPE
        
    Returns:
        Dict mapping Hamiltonian names to minimum evolution times
    """
    if m <= 0:
        raise ValueError("Number of precision bits m must be positive")
    
    results = {}
    
    for name, hamiltonian in hamiltonians.items():
        if not isinstance(hamiltonian, SparsePauliOp):
            raise TypeError(f"Hamiltonian '{name}' must be a SparsePauliOp")
        
        # Calculate the largest eigenvalue magnitude (spectral norm)
        # For Pauli operators, this is the sum of absolute values of coefficients
        lambda_max = np.sum(np.abs(hamiltonian.coeffs))
        
        if lambda_max == 0:
            warnings.warn(f"Hamiltonian '{name}' has zero norm, setting t_min to infinity")
            results[name] = float('inf')
            continue
        
        # Calculate minimum evolution time: t_min = sqrt(1/(2^m * λ))
        t_min = np.sqrt(1.0 / (2**m * lambda_max))
        results[name] = float(t_min)
    
    return results
