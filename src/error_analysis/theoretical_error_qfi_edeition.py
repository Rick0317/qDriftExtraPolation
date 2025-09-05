import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

# If you already have these objects defined in your session, you can skip the import of qiskit.
try:
    from qiskit.quantum_info import SparsePauliOp
except ImportError:
    SparsePauliOp = None  # Only needed for type hints; not required at runtime if your dict is already built.


# --------------------------
# QFI-driven selector (from earlier, adapted)
# --------------------------

@dataclass
class EspressoMartiniQFIParams:
    t_lower_inclusive_minimum_sip_time: float = 1e-3
    t_upper_inclusive_maximum_sip_time: float = 2.0
    n_t_points_to_pick: int = 20
    n_dense_candidate_grid_points: int = 4096

    # QFI model F_Q(t) = 4 Var(H) t^2 (pure-state, unitary encoding)
    qfi_prefactor_for_unitary_pure_state: float = 4.0

    # Diversity controls in log-space
    minimum_required_logtime_separation_fraction_of_range: float = 0.05
    fisher_vs_spacing_tradeoff_beta: float = 1.0

    # Plotting controls
    make_plot: bool = True
    figure_size_inches: Tuple[float, float] = (7.2, 4.6)
    show_grid: bool = True


def quantum_fisher_information_for_lambda_vs_time(
    t_array: np.ndarray,
    qfi_prefactor_for_unitary_pure_state: float,
    generator_variance_value: float
) -> np.ndarray:
    # F_Q(t) = 4 Var(H) t^2
    return (qfi_prefactor_for_unitary_pure_state * generator_variance_value) * (t_array ** 2)


def pick_t_values_by_qfi_with_logspace_diversity(
    t_lower: float,
    t_upper: float,
    n_points: int,
    generator_variance_value: float,
    qfi_prefactor: float = 4.0,
    n_dense: int = 4096,
    min_log_sep_frac: float = 0.05,
    beta_spacing_weight: float = 1.0
) -> Dict[str, np.ndarray]:
    if not (t_lower > 0 and t_upper > t_lower):
        raise ValueError("Require 0 < t_lower < t_upper.")

    t_candidates_dense = np.geomspace(t_lower, t_upper, n_dense)
    qfi_candidates = quantum_fisher_information_for_lambda_vs_time(
        t_candidates_dense, qfi_prefactor, generator_variance_value
    )

    n = int(n_points)
    if n < 1:
        raise ValueError("n_points must be >= 1.")

    log_t_dense = np.log(t_candidates_dense)
    total_log_span = log_t_dense[-1] - log_t_dense[0]
    min_sep = min_log_sep_frac * total_log_span

    # Start with the single most informative point
    selected_indices = [int(np.argmax(qfi_candidates))]
    selected_t_values = [t_candidates_dense[selected_indices[0]]]

    def min_log_distance_to_current_selection(idx: int) -> float:
        return float(np.min(np.abs(log_t_dense[idx] - np.log(np.array(selected_t_values)))))

    for _ in range(1, n):
        weighted_scores = np.zeros_like(qfi_candidates)
        for idx in range(t_candidates_dense.size):
            d = min_log_distance_to_current_selection(idx)
            if d < min_sep:
                weighted_scores[idx] = 0.0
            else:
                weighted_scores[idx] = qfi_candidates[idx] * (d ** beta_spacing_weight)

        if weighted_scores.max() <= 0.0:
            break

        next_idx = int(np.argmax(weighted_scores))
        selected_indices.append(next_idx)
        selected_t_values.append(t_candidates_dense[next_idx])

    selected_t_values = np.array(sorted(selected_t_values))
    selected_qfi_values = quantum_fisher_information_for_lambda_vs_time(
        selected_t_values, qfi_prefactor, generator_variance_value
    )

    return dict(
        selected_t_values=selected_t_values,
        selected_qfi_values=selected_qfi_values,
        dense_t_candidates=t_candidates_dense,
        dense_qfi_candidates=qfi_candidates
    )


def plot_qfi_with_selected_points_for_hamiltonian(
    selection: Dict[str, np.ndarray],
    hamiltonian_name: str,
    generator_variance_value: float,
    params: EspressoMartiniQFIParams
) -> None:
    t_grid = selection["dense_t_candidates"]
    qfi_grid = selection["dense_qfi_candidates"]
    t_sel = selection["selected_t_values"]
    qfi_sel = selection["selected_qfi_values"]

    plt.figure(figsize=params.figure_size_inches)
    plt.loglog(t_grid, qfi_grid, color="#2ca02c", lw=2.0, label=f"QFI(t) = 4 Var(H) t^2, Var(H)={generator_variance_value:.6f}")
    plt.scatter(t_sel, qfi_sel, color="k", s=24, zorder=5, label="Selected t")
    plt.xlabel("t (evolution time)")
    plt.ylabel("Quantum Fisher Information about λ")
    plt.title(f"QFI-driven t selection for {hamiltonian_name}")
    if params.show_grid:
        plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


# --------------------------
# Hamiltonian-specific: Var(H) on |+ +>
# --------------------------

def variance_on_plus_plus_for_diagonal_Z_pauli_sum(pauli_labels: np.ndarray, coeffs: np.ndarray) -> float:
    """
    Compute Var(H) on |+ +> for H = sum_j c_j P_j, where P_j are 2-qubit diagonal Z-strings (I, Z) x (I, Z).
    For |+> (eigenstate of X), <Z>=0 and <I>=1. Then:
      <H> = sum_j c_j <P_j> = c_{II} only,
      <H^2> = sum_j c_j^2 (cross terms have zero expectation on |+ +>),
      Var(H) = <H^2> - <H>^2 = (sum_j c_j^2) - c_{II}^2 = sum of squares of non-identity coefficients.
    """
    # Identify the identity term index (if any)
    cII = 0.0
    sum_sq = float(np.sum(np.real(coeffs)**2 + np.imag(coeffs)**2))  # robust if coeffs complex
    for p, c in zip(pauli_labels, coeffs):
        if p == "II":
            cII = float(np.real(c)) if np.isrealobj(c) else float(np.real(c))
            # If coeffs may be complex, identity expectation is Re(c), but for variance we use magnitudes below carefully
            # Here, since all provided coeffs are real, this is fine.
            break
    var = sum_sq - (cII ** 2)
    return float(var)


# --------------------------
# Bring it together for your dict
# --------------------------

def select_t_for_each_hamiltonian(
    hamiltonians: Dict[str, Any],
    params: Optional[EspressoMartiniQFIParams] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For each Hamiltonian in the dict (SparsePauliOp-like with .coeffs and .paulis/.pauli_strings),
    compute Var(H) on |+ +>, perform QFI-based t selection, and optionally plot.
    Returns a dict: name -> {"t_values": ..., "qfi_values": ...}.
    """
    if params is None:
        params = EspressoMartiniQFIParams()

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for name, H in hamiltonians.items():
        # Extract pauli labels and coeffs
        # Qiskit's SparsePauliOp exposes .paulis.to_labels() (array of strings) and .coeffs
        if hasattr(H, "paulis"):
            pauli_labels = np.array(H.paulis.to_labels())
            coeffs = np.array(H.coeffs)
        elif hasattr(H, "data") and hasattr(H, "coeffs"):
            # If the object mimics your construction
            pauli_labels = np.array(H.data)  # e.g., ["ZI", "ZZ", "IZ", "II"]
            coeffs = np.array(H.coeffs)
        else:
            raise TypeError(f"Hamiltonian {name} is not a recognized SparsePauliOp-like object.")

        # Compute Var(H) on |+ +>
        var_H = variance_on_plus_plus_for_diagonal_Z_pauli_sum(pauli_labels, coeffs)
        if var_H <= 0:
            # If variance is zero (e.g., H is proportional to identity), information is zero for this probe
            print(f"[WARN] Var(H)=0 for {name} on |+ +>. Skipping selection.")
            continue

        # QFI-based selection
        selection = pick_t_values_by_qfi_with_logspace_diversity(
            t_lower=params.t_lower_inclusive_minimum_sip_time,
            t_upper=params.t_upper_inclusive_maximum_sip_time,
            n_points=params.n_t_points_to_pick,
            generator_variance_value=var_H,
            qfi_prefactor=params.qfi_prefactor_for_unitary_pure_state,
            n_dense=params.n_dense_candidate_grid_points,
            min_log_sep_frac=params.minimum_required_logtime_separation_fraction_of_range,
            beta_spacing_weight=params.fisher_vs_spacing_tradeoff_beta
        )

        results[name] = dict(
            t_values=selection["selected_t_values"],
            qfi_values=selection["selected_qfi_values"],
            var_H=np.array([var_H])
        )

        # Plot if requested
        if params.make_plot:
            plot_qfi_with_selected_points_for_hamiltonian(selection, name, var_H, params)

    return results

