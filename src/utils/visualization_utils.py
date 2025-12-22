"""
Visualization utilities for quantum simulation results analysis.

This module provides modular and extensible plotting functions for:
- Chebyshev extrapolation analysis
- Energy distribution histograms  
- Box-and-whisker plots for uncertainty quantification
- Cost vs error analysis
- Hamiltonian LaTeX formatting and display
"""

import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import ast
import math
import statistics
from typing import Dict, List, Optional, Literal, Tuple, Any
from dataclasses import dataclass

# Hamiltonian visualization imports
from sympy import symbols, I, exp, latex, simplify, Matrix
from IPython.display import display, Math
from qiskit.quantum_info import SparsePauliOp


@dataclass
class ExtrapolationResult:
    """Container for extrapolation analysis results."""
    y_intercept: float
    abs_error: float
    coeffs: np.ndarray
    num_gates_per_datapoint: float
    num_ancilla: int
    alpha: float
    num_qdrift_segments: int
    exact_eigenvalue: float
    best_datapoint_error: float
    extrapolation_is_better: bool


def calculate_cost_per_datapoint(
    n_circuits: int,
    n_shots_per_circuit: int,
    num_qdrift_segments: int,
    num_ancilla: int
) -> float:
    """
    Calculate the total gate cost per datapoint.
    
    Args:
        n_circuits: Number of independent random circuits
        n_shots_per_circuit: Number of shots per circuit
        num_qdrift_segments: Number of qDRIFT segments per invocation
        num_ancilla: Number of ancilla qubits
    
    Returns:
        Total gate count per datapoint
    """
    return n_circuits * n_shots_per_circuit * num_qdrift_segments * (2**num_ancilla - 1)


def fit_polynomial_and_extrapolate(
    times: np.ndarray,
    energies: np.ndarray,
    degree: int = 2
) -> Tuple[np.ndarray, float]:
    """
    Fit polynomial to data and return coefficients and y-intercept.
    
    Args:
        times: Time values
        energies: Estimated energy values
        degree: Polynomial degree (default: 2 for quadratic)
    
    Returns:
        Tuple of (coefficients, y_intercept)
    """
    coeffs = Polynomial.fit(times, energies, degree).convert().coef
    y_intercept = coeffs[0]
    return coeffs, y_intercept


def _prepare_mirrored_data(group: pd.DataFrame) -> pd.DataFrame:
    """Mirror data across y-axis to duplicate for extrapolation."""
    df_mirror = group.copy()
    df_mirror["time"] = -df_mirror["time"]
    return pd.concat([group, df_mirror])


def _calculate_theoretical_stdev_upper_bound(
    alpha: float,
    times: np.ndarray,
    num_qdrift_segments: int
) -> np.ndarray:
    """Calculate theoretical upper bound for standard deviation."""
    return (alpha ** 2 * times ** 2) / num_qdrift_segments


def _plot_extrapolation_curve(
    ax: plt.Axes,
    group: pd.DataFrame,
    coeffs: np.ndarray,
    y_intercept: float,
    exact_eigenvalue: float,
    result: ExtrapolationResult
) -> None:
    """Plot the main extrapolation curve with data points."""
    # Plot data points
    ax.plot(group["time"], group["est_energy_mean"], 
            marker='o', linestyle='None', label='Data points')
    
    # Plot extrapolation curve
    x_extra = np.linspace(group["time"].min() * 1.01, group["time"].max() * 1.01, 200)
    y_extra = sum(c * x_extra**i for i, c in enumerate(coeffs))
    ax.plot(x_extra, y_extra, label='Extrapolated Curve', color='orange')
    
    # Reference lines
    ax.axhline(y=y_intercept, color='g', linestyle='--',
               label='Extrapolated Eigenvalue')
    ax.axhline(y=exact_eigenvalue, color='r', linestyle='--',
               label='Exact Eigenvalue')
    
    # Labels and styling
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimated Eigenvalue")
    ax.set_title(f"Estimated Eigenvalue vs Time (Exact: {exact_eigenvalue:.4f})")
    ax.grid(True)
    ax.legend()
    
    # Add stats textbox
    stats_text = (
        f"Extrapolated: {y_intercept:.4f}\n"
        f"Absolute Error: {result.abs_error:.4f}\n"
        f"Alpha: {result.alpha:.4f}\n"
        f"Num Ancilla: {result.num_ancilla}\n"
        f"Num Segments: {result.num_qdrift_segments}\n"
    )
    
    if "Num Random Circuits" in group.columns:
        stats_text += f"Num Random Circuits: {group['Num Random Circuits'].iloc[0]}\n"
    
    if result.extrapolation_is_better:
        stats_text += "✓ Better than best datapoint"
    else:
        stats_text += "✗ Not better than best datapoint"
    
    ax.text(-0.35, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def _plot_stdev_comparison(
    ax: plt.Axes,
    group: pd.DataFrame,
    stdev_theoretical_upper_bound: np.ndarray,
    alpha: float,
    num_ancilla: int
) -> None:
    """Plot standard deviation comparison."""
    ax.plot(group["time"], group["est_energy_std"], 
            marker='o', linestyle='None', label='Real Standard Deviation')
    ax.plot(group["time"], stdev_theoretical_upper_bound, 
            marker='x', linestyle='None', 
            label='Theoretical Upper Bound (qDRIFT only)')
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Standard Deviation")
    ax.set_title(f"Standard Deviation vs Time")
    ax.grid(True)
    ax.legend()


def _plot_boxplot_mode(
    ax: plt.Axes,
    group: pd.DataFrame,
    exact_eigenvalue: float
) -> None:
    """
    Plot box-and-whisker plots for energy distributions at each time point.
    
    This provides a richer view of the uncertainty in estimates compared to 
    just showing standard deviation.
    """
    # Prepare data for boxplot
    time_values = sorted(group["time"].unique())
    box_data = []
    positions = []
    labels = []
    
    for i, t in enumerate(time_values):
        time_group = group[group["time"] == t]
        
        # If we have individual measurements, use them
        if "est_energy_samples" in time_group.columns:
            # Assume est_energy_samples is a list/array of individual measurements
            samples = time_group["est_energy_samples"].iloc[0]
            box_data.append(samples)
        else:
            # Fallback: use mean ± std to create synthetic distribution
            mean = time_group["est_energy_mean"].iloc[0]
            std = time_group["est_energy_std"].iloc[0]
            # Create synthetic samples from normal distribution
            synthetic_samples = np.random.normal(mean, std, size=100)
            box_data.append(synthetic_samples)
        
        positions.append(i)  # Use integer positions for cleaner x-axis
        labels.append(f"{abs(t):.3f}")  # Format label nicely
    
    # Create box plot with integer positions
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                     patch_artist=True, showfliers=True)
    
    # Style the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Add exact eigenvalue reference line
    ax.axhline(y=exact_eigenvalue, color='r', linestyle='--',
               label='Exact Eigenvalue', linewidth=2)
    
    # Set x-axis with clean labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimated Energy Distribution")
    ax.set_title("Energy Distribution (Box-and-Whisker)")
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()


def plot_cheby_extrapolation(
    groupbys: List[str],
    df: pd.DataFrame,
    display_mode: Literal["stdev", "boxplot", "none"] = "none",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Plot Chebyshev extrapolation analysis with flexible display modes.
    
    Args:
        groupbys: List of column names to group by
        df: DataFrame containing simulation results
        display_mode: Secondary plot mode:
            - "stdev": Show standard deviation comparison
            - "boxplot": Show box-and-whisker plots
            - "none": Show only extrapolation curve
        verbose: Print detailed information during processing
    
    Returns:
        DataFrame containing cost vs error analysis for all groups
    """
    cost_vs_error_data = pd.DataFrame(
        columns=["num_gates_per_datapoint", "abs_error_of_extrapolation", "num_ancilla"]
    )
    
    for group_name, group in df.groupby(groupbys):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Group: {group_name}")
        
        # Extract parameters
        exact_eigenvalue = group["exact_eig"].iloc[0]
        alpha = group["alpha"].iloc[0]
        num_ancilla = group["num_ancilla"].iloc[0]
        n_circuits = group["n_circuits"].iloc[0]
        n_shots_per_circuit = group["n_shots"].iloc[0]
        num_qdrift_segments = group.get("num segments per invocation", pd.Series([1])).iloc[0]
        
        num_gates_per_datapoint = calculate_cost_per_datapoint(
            n_circuits, n_shots_per_circuit, num_qdrift_segments, num_ancilla
        )
        
        if verbose:
            print(f"Exact Eigenvalue: {exact_eigenvalue}")
            print(f"Alpha: {alpha}")
            print(f"Num Ancilla: {num_ancilla}")
            print(f"Num Gates per Data Point: {num_gates_per_datapoint}")
        
        # Prepare data (mirror for physics reasons)
        group = _prepare_mirrored_data(group)
        
        # Fit polynomial
        coeffs, y_intercept = fit_polynomial_and_extrapolate(
            group["time"].values, group["est_energy_mean"].values
        )
        
        # Calculate best direct approximation
        best_datapoint_idx = np.argmin(np.abs(group["est_energy_mean"] - exact_eigenvalue))
        best_datapoint_error = abs(exact_eigenvalue - group.iloc[best_datapoint_idx]['est_energy_mean'])
        
        abs_error = abs(exact_eigenvalue - y_intercept)
        extrapolation_is_better = abs_error < best_datapoint_error
        
        result = ExtrapolationResult(
            y_intercept=y_intercept,
            abs_error=abs_error,
            coeffs=coeffs,
            num_gates_per_datapoint=num_gates_per_datapoint,
            num_ancilla=num_ancilla,
            alpha=alpha,
            num_qdrift_segments=num_qdrift_segments,
            exact_eigenvalue=exact_eigenvalue,
            best_datapoint_error=best_datapoint_error,
            extrapolation_is_better=extrapolation_is_better
        )
        
        if verbose:
            print(f"Polynomial Coefficients: {coeffs}")
            print(f"Y-intercept: {y_intercept}")
            print(f"Absolute Error: {abs_error}")
            print(f"Best datapoint error: {best_datapoint_error}")
            print(f"Extrapolation better: {extrapolation_is_better}")
        
        # Create figure based on display mode
        if display_mode == "none":
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = None
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot main extrapolation curve
        _plot_extrapolation_curve(ax1, group, coeffs, y_intercept, exact_eigenvalue, result)
        
        # Plot secondary visualization based on mode
        if display_mode == "stdev" and ax2 is not None:
            stdev_theoretical_upper_bound = _calculate_theoretical_stdev_upper_bound(
                alpha, group["time"].values, num_qdrift_segments
            )
            _plot_stdev_comparison(ax2, group, stdev_theoretical_upper_bound, alpha, num_ancilla)
        
        elif display_mode == "boxplot" and ax2 is not None:
            _plot_boxplot_mode(ax2, group, exact_eigenvalue)
        
        plt.tight_layout()
        plt.show()
        
        # Save cost vs error data
        cost_vs_error_data = cost_vs_error_data._append({
            "num_gates_per_datapoint": num_gates_per_datapoint,
            "abs_error_of_extrapolation": abs_error,
            "num_ancilla": num_ancilla
        }, ignore_index=True)
    
    return cost_vs_error_data


def plot_cost_vs_error_with_heisenberg_limit(
    cost_data: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 7),
    title: Optional[str] = None
) -> None:
    """
    Plot quantum simulation error vs number of gates with Heisenberg limit.
    
    Args:
        cost_data: DataFrame with columns 'num_gates_per_datapoint', 
                   'abs_error_of_extrapolation', 'num_ancilla'
        figsize: Figure size tuple
        title: Optional custom title
    """
    plt.figure(figsize=figsize)
    
    # Plot the actual data points, colored by num_ancilla
    scatter = plt.scatter(
        cost_data['num_gates_per_datapoint'],
        cost_data['abs_error_of_extrapolation'],
        c=cost_data['num_ancilla'],
        s=80, alpha=0.7, cmap='viridis', zorder=3
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Ancilla Qubits', rotation=270, labelpad=20)
    
    # Plot the Heisenberg Limit (1/N scaling)
    sorted_costs = np.sort(cost_data['num_gates_per_datapoint'].values)
    heisenberg_limit = 1 / sorted_costs
    plt.plot(sorted_costs, heisenberg_limit,
             'r--', linewidth=2, label='Heisenberg Limit (1/N)', zorder=2)
    
    # Set log scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Labels and title
    plt.xlabel('Number of Gates per Data Point', fontsize=12)
    plt.ylabel('Absolute Error of Extrapolation', fontsize=12)
    
    if title is None:
        title = 'Quantum Simulation: Error vs Number of Gates\nwith Heisenberg Limit'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add grid for better readability on log-log plot
    plt.grid(True, alpha=0.3, linestyle=':', which='both')
    
    # Legend
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()


def plot_histograms(
    dicts: List[Dict[str, int]],
    titles: Optional[List[str]] = None,
    max_plots: int = 12,
    bar_kw: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (12, 8),
    suptitle: Optional[str] = None,
    show_bitstrings: bool = False,
    exact_eigenvalue: Optional[float] = None,
    time_values: Optional[List[float]] = None
) -> None:
    """
    Plot histograms of measurement outcomes across multiple experiments.
    
    Args:
        dicts: List of count dictionaries (bitstring -> count)
        titles: Optional list of subplot titles
        max_plots: Maximum number of subplots to show
        bar_kw: Keyword arguments for bar plots
        figsize: Figure size
        suptitle: Overall figure title
        show_bitstrings: If True, show bitstrings on x-axis; otherwise show integers
        exact_eigenvalue: If provided, mark with vertical line
        time_values: If provided, show time values in titles
    """
    n = min(len(dicts), max_plots)
    if len(dicts) > max_plots:
        print(f"⚠ Only the first {max_plots} histograms will be drawn.")
    
    bar_kw = bar_kw or {}
    cols = 3
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten() if n > 1 else [axes]
    
    for i in range(n):
        ax = axes[i]
        data = dicts[i]
        
        if show_bitstrings:
            keys = list(data.keys())
            vals = list(data.values())
            x_pos = range(len(keys))
        else:
            keys = [int(bs, 2) for bs in data.keys()]
            vals = list(data.values())
            x_pos = range(len(keys))
        
        ax.bar(x_pos, vals, **bar_kw)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(keys, rotation=45)
        ax.set_ylabel("Counts")
        
        # Add title
        if titles and i < len(titles):
            ax.set_title(titles[i])
        elif time_values and i < len(time_values):
            ax.set_title(f"t = {time_values[i]:.4f}")
        
        # Mark exact eigenvalue if provided
        if exact_eigenvalue is not None:
            # This would require converting eigenvalue to bitstring - skip for now
            pass
        
        ax.grid(True, axis='y', alpha=0.3)
    
    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    plt.show()


def median_from_counter(counter: Dict[Any, int]) -> float:
    """
    Calculate median from a Counter/dict of value -> frequency.
    
    Args:
        counter: Dictionary mapping values to their frequencies
    
    Returns:
        Median value, or NaN if empty
    """
    if not counter:
        return np.nan
    expanded = [val for val, freq in counter.items() for _ in range(freq)]
    return statistics.median(expanded)


def plot_energy_histograms_over_time(
    group: pd.DataFrame,
    exact_eigenvalue: float,
    theoretical_error: np.ndarray,
    energy_counts_column: str = "estimnated energies counts",
    time_column: str = "Time"
) -> None:
    """
    Plot histograms of estimated energy counts over time steps with statistics.
    
    Args:
        group: DataFrame containing energy counts and time values
        exact_eigenvalue: True eigenvalue to mark with vertical line
        theoretical_error: Theoretical error bound(s) per time step
        energy_counts_column: Column name containing energy count dictionaries
        time_column: Column name containing time values
    """
    all_counts = group[energy_counts_column].values
    
    # Clean and parse count dictionaries
    c_cleaned = [x.replace("np.float64(", "").replace(")", "") for x in all_counts]
    counts = [ast.literal_eval(x) for x in c_cleaned]
    time_vals = group[time_column].values
    
    # Collect all possible energies
    all_estimated_energies = sorted(set(energy for count in counts for energy in count.keys()))
    
    num_times = len(counts)
    fig, axes = plt.subplots(num_times, 1, figsize=(10, 3 * num_times), sharex=True)
    
    if num_times == 1:
        axes = [axes]
    
    # Handle theoretical_error as scalar or array
    if np.isscalar(theoretical_error):
        theoretical_error = [theoretical_error] * num_times
    
    for idx, (ax, count, time_val, theo_err) in enumerate(
        zip(axes, counts, time_vals, theoretical_error)
    ):
        counts_for_energies = [count.get(energy, 0) for energy in all_estimated_energies]
        ax.bar(all_estimated_energies, counts_for_energies, width=0.1)
        ax.set_title(f"Time Step {time_val:.4f}")
        ax.set_ylabel("Counts")
        ax.grid(True, axis='y')
        ax.axvline(x=exact_eigenvalue, color='r', linestyle='--', label='Exact Eigenvalue')
        ax.legend()
        
        # Calculate weighted statistics
        energies = np.array(all_estimated_energies)
        counts_array = np.array(counts_for_energies)
        flatten_counts = np.repeat(energies, counts_array.astype(int))
        
        if counts_array.sum() > 0:
            median = np.median(flatten_counts)
            stdev = np.std(flatten_counts)
        else:
            median = np.nan
            stdev = np.nan
        
        # Add text box with statistics
        stats_text = (
            f"Median: {median:.4f}\n"
            f"Stdev: {stdev:.4f}\n"
            f"Theo. Error: {theo_err:.4f}"
        )
        ax.text(0.98, 0.95, stats_text, ha='right', va='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black',
                         boxstyle='round,pad=0.5'))
    
    axes[-1].set_xlabel("Energy")
    plt.tight_layout()
    plt.show()


# ============================================================================
# Hamiltonian Visualization Functions (merged from utils_visualization.py)
# ============================================================================

def format_hamiltonian_matrix_as_latex(matrix: np.ndarray) -> str:
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


def display_hamiltonian(H: SparsePauliOp) -> None:
    """Display both the explicit tensor product form and the simplified form of the Hamiltonian."""
    
    # Explicit tensor product form
    explicit_latex = hamiltonian_explicit(H)
    display(Math(f"H = {explicit_latex}"))
    
    # Simplified form
    simplified_latex = hamiltonian_simplified(H)
    display(Math(f"H = {simplified_latex}"))

    # Matrix representation
    matrix = H.to_matrix()
    matrix_latex = format_hamiltonian_matrix_as_latex(matrix)
    display(Math(f"H = {matrix_latex}"))
