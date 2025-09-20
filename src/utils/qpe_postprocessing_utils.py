"""
Optimized utilities for QPE count processing with wrap-around correction.

The wrap-around correction is crucial for handling negative eigenvalues in QPE:
- Phases in [0, 0.5] map to positive eigenvalues
- Phases in (0.5, 1) are wrapped to (-0.5, 0] for negative eigenvalues
- This ensures eigenvalues are correctly reconstructed in the range [-π/t, π/t]
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numba
from numba import jit, prange


def int_to_bitstring(value: int, m: int) -> str:
    if value < 0:
        raise ValueError("value must be non-negative")
    if value >= 2**m:
        raise ValueError(f"value {value} cannot be represented with {m} bits")

    return format(value, f"0{m}b")


@jit(nopython=True)
def bitstring_to_int_vectorized(bitstrings: np.ndarray) -> np.ndarray:
    """
    Convert array of bitstrings to integers using vectorized operations.
    Optimized with Numba for speed.
    """
    n_bits = len(bitstrings[0])
    result = np.zeros(len(bitstrings), dtype=np.int32)
    
    for i in prange(len(bitstrings)):
        val = 0
        for j in range(n_bits):
            if bitstrings[i][j] == ord('1'):
                val |= (1 << (n_bits - 1 - j))
        result[i] = val
    
    return result


def merge_counts_optimized(counts_list: List[Dict[str, int]]) -> Dict[str, int]:
    if not counts_list:
        return {}
    
    if len(counts_list) == 1:
        return counts_list[0]
    
    # Use Counter for efficient merging
    total = Counter()
    for counts in counts_list:
        total.update(counts)
    
    return dict(total)


def process_counts_with_median(counts: Dict[str, int], n_anc: int) -> str:
    """
    Process counts dictionary to find the median bitstring.
    
    Args:
        counts: Dictionary mapping bitstrings to their observed counts
        n_anc: Number of ancilla qubits (length of bitstrings)
    
    Returns:
        Median bitstring as a binary string of length `n_anc`
    """
    
    keys = np.array(list(counts.keys()), dtype='S')  # Use byte strings for Numba compatibility
    freqs = np.array(list(counts.values()), dtype=np.int32)
    
    expanded_arr = np.repeat(keys, freqs)
    expanded_arr_int = bitstring_to_int_vectorized(expanded_arr)
    
    median_int = int(round(np.median(expanded_arr_int)))
    
    return int_to_bitstring(value=median_int, m=n_anc)


def batch_process_counts(
    counts_list: List[Dict[str, int]],
    shots_per_circuit: int,
    n_anc: int,
    group_by: str = "median"
) -> Dict[str, int]:
    """
    Process a batch of count dictionaries with grouping.
    
    Args:
        counts_list: List of count dictionaries from multiple circuits
        shots_per_circuit: Number of shots per circuit
        n_anc: Number of ancilla qubits
        group_by: Grouping method ("median", "mean", or "none")
    
    Returns:
        Merged count dictionary
    """
    if group_by == "none" or shots_per_circuit == 1:
        return merge_counts_optimized(counts_list)
    
    processed_counts = []
    
    if group_by == "median":
        for counts in counts_list:
            median_bs = process_counts_with_median(counts, n_anc)
            processed_counts.append({median_bs: 1})
    
    elif group_by == "mean":
        for counts in counts_list:
            if not counts:
                processed_counts.append({"0" * n_anc: 1})
                continue
            keys = list(counts.keys())
            freqs = np.array(list(counts.values()))
            int_values = np.array([int(bs, 2) for bs in keys])
            
            mean_val = np.average(int_values, weights=freqs)
            mean_bs = format(int(round(mean_val)), f'0{n_anc}b')
            processed_counts.append({mean_bs: 1})
    
    return merge_counts_optimized(processed_counts)

@jit(nopython=True)
def calculate_energies_vectorized(
    bitstrings: np.ndarray,
    frequencies: np.ndarray,
    total_time: float,
    n_anc: int,
    wrap_around_correction: bool = True
) -> Tuple[float, float, float, float, float]:
    """
    Calculate energy statistics from measurement outcomes using vectorized operations.
    
    Args:
        bitstrings: Array of integer representations of measurement outcomes
        frequencies: Array of counts for each bitstring
        total_time: Evolution time used in QPE
        n_anc: Number of ancilla qubits
        wrap_around_correction: Apply correction for phases > 0.5 (for negative eigenvalues)
    
    Returns:
        Tuple of (median, mean, std, min, max) energies
    """
    # Convert bitstrings to phase values with wrap-around correction
    n_measurements = np.sum(frequencies)
    energies = np.zeros(n_measurements)
    idx = 0
    
    for i in range(len(bitstrings)):
        # Convert integer to phase in [0, 1)
        phase = bitstrings[i] / (2**n_anc)
        
        # Apply wrap-around correction: map phases > 0.5 to (-0.5, 0.5]
        if wrap_around_correction and phase > 0.5:
            phase -= 1.0
        
        # Convert phase to energy
        energy = 2 * np.pi * phase / total_time
        
        # Expand by frequency
        for _ in range(frequencies[i]):
            energies[idx] = energy
            idx += 1
    
    # Calculate statistics
    median = np.median(energies)
    mean = np.mean(energies)
    std = np.std(energies) if n_measurements > 1 else 0.0
    min_e = np.min(energies) if n_measurements > 0 else 0.0
    max_e = np.max(energies) if n_measurements > 0 else 0.0
    
    return median, mean, std, min_e, max_e


def analyse_counts_optimized(
    counts: Dict[str, int],
    total_time: float,
    n_anc: int,
    wrap_around_correction: bool = True
) -> Tuple[str, float, float, float, float, float]:
    """
    Optimized version of analyse_counts using vectorized operations.
    
    Args:
        counts: Dictionary mapping bitstrings to their observed counts
        total_time: Evolution time used in the QPE experiment
        n_anc: Number of ancilla qubits used in the QPE experiment
        wrap_around_correction: Whether to apply correction for phases > 0.5
                               (maps to (-0.5, 0.5] for negative eigenvalues)
    
    Returns:
        Tuple of (most_likely_bitstring, median_energy, mean_energy, 
                 std_energy, min_energy, max_energy)
    """
    if not counts:
        zero_bs = "0" * n_anc
        return zero_bs, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Find most likely bitstring
    most_likely_bs = max(counts, key=counts.get)
    
    # Convert to arrays for vectorized processing
    keys = list(counts.keys())
    freqs = np.array(list(counts.values()))
    
    # Convert bitstrings to integers
    int_values = np.array([int(bs, 2) for bs in keys])
    
    # Calculate energy statistics with wrap-around correction
    median, mean, std, min_e, max_e = calculate_energies_vectorized(
        int_values, freqs, total_time, n_anc, wrap_around_correction
    )
    
    return most_likely_bs, median, mean, std, min_e, max_e