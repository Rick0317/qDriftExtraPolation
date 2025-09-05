import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy.optimize import brentq


@dataclass
class ErrorModelParams:
    alpha_l1_norm_of_hamiltonian_terms: float  # alpha = sum_j |h_j|
    qdrift_total_samples_for_target_evolution: int  # N (kept fixed across t sweep)
    qpe_ancilla_qubits_count: int  # m
    target_total_error_for_plot_window: float  # eps_max, just to size the window and bounds
    number_of_t_values_to_return: int = 20
    crossover_visual_band_multipliers: Tuple[float, float] = (0.9, 1.2)  # for shading


def compute_explicit_constants(alpha_l1_norm_of_hamiltonian_terms: float,
                               qdrift_total_samples_for_target_evolution: int,
                               qpe_ancilla_qubits_count: int) -> Dict[str, float]:
    # These are kept explicit in comments, but code returns descriptive names.
    # qDRIFT prefactor: 2 * alpha^2 / N^2
    qdrift_prefactor_two_alpha_sq_over_N_sq = (
        2.0
        * (alpha_l1_norm_of_hamiltonian_terms ** 2)
        / (qdrift_total_samples_for_target_evolution ** 2)
    )
    # qDRIFT exponential rate: 2 * alpha / N
    qdrift_exponent_rate_two_alpha_over_N = (
        2.0
        * alpha_l1_norm_of_hamiltonian_terms
        / qdrift_total_samples_for_target_evolution
    )
    # QPE phase resolution constant: 2*pi / 2^m
    qpe_phase_step_size_two_pi_over_two_pow_m = (
        2.0 * np.pi / (2 ** qpe_ancilla_qubits_count)
    )
    return dict(
        qdrift_prefactor_two_alpha_sq_over_N_sq=qdrift_prefactor_two_alpha_sq_over_N_sq,
        qdrift_exponent_rate_two_alpha_over_N=qdrift_exponent_rate_two_alpha_over_N,
        qpe_phase_step_size_two_pi_over_two_pow_m=qpe_phase_step_size_two_pi_over_two_pow_m
    )


def qdrift_error_upper_bound(t, qdrift_prefactor_two_alpha_sq_over_N_sq, qdrift_exponent_rate_two_alpha_over_N):
    # E_qDRIFT(t) = (2 alpha^2 / N^2) * t * exp((2 alpha / N) * t)
    return qdrift_prefactor_two_alpha_sq_over_N_sq * t * np.exp(qdrift_exponent_rate_two_alpha_over_N * t)


def qpe_error_upper_bound(t, qpe_phase_step_size_two_pi_over_two_pow_m):
    # E_QPE(t) = (2pi / 2^m) * (1/t)
    t_safe = np.maximum(t, 1e-300)
    return qpe_phase_step_size_two_pi_over_two_pow_m / t_safe


def total_error_sum_upper_bound(t, qdrift_prefactor_two_alpha_sq_over_N_sq, qdrift_exponent_rate_two_alpha_over_N,
                                qpe_phase_step_size_two_pi_over_two_pow_m):
    return (
        qdrift_error_upper_bound(t, qdrift_prefactor_two_alpha_sq_over_N_sq, qdrift_exponent_rate_two_alpha_over_N)
        + qpe_error_upper_bound(t, qpe_phase_step_size_two_pi_over_two_pow_m)
    )


def total_error_rms_surrogate(t, qdrift_prefactor_two_alpha_sq_over_N_sq, qdrift_exponent_rate_two_alpha_over_N,
                              qpe_phase_step_size_two_pi_over_two_pow_m):
    eq = qdrift_error_upper_bound(t, qdrift_prefactor_two_alpha_sq_over_N_sq, qdrift_exponent_rate_two_alpha_over_N)
    ep = qpe_error_upper_bound(t, qpe_phase_step_size_two_pi_over_two_pow_m)
    return np.sqrt(eq**2 + ep**2)


def solve_for_crossover_t_star(qdrift_prefactor_two_alpha_sq_over_N_sq, qdrift_exponent_rate_two_alpha_over_N,
                               qpe_phase_step_size_two_pi_over_two_pow_m):
    # Solve: (2 alpha^2 / N^2) * t^2 * exp((2 alpha / N) * t) = (2 pi / 2^m)
    def f(t):
        return (
            qdrift_prefactor_two_alpha_sq_over_N_sq * (t**2) * np.exp(qdrift_exponent_rate_two_alpha_over_N * t)
            - qpe_phase_step_size_two_pi_over_two_pow_m
        )

    t_lo = 0.0
    t_hi = 1.0
    while f(t_hi) <= 0.0:
        t_hi *= 2.0
        if t_hi > 1e6:
            return t_hi
    return brentq(f, t_lo, t_hi, maxiter=200, xtol=1e-14)


def solve_for_t_max_given_qdrift_budget(qdrift_prefactor_two_alpha_sq_over_N_sq, qdrift_exponent_rate_two_alpha_over_N,
                                        qdrift_error_budget):
    # Solve: (2 alpha^2 / N^2) * t * exp((2 alpha / N) * t) = qdrift_error_budget
    def g(t):
        return (
            qdrift_prefactor_two_alpha_sq_over_N_sq * t * np.exp(qdrift_exponent_rate_two_alpha_over_N * t)
            - qdrift_error_budget
        )

    t_lo = 0.0
    t_hi = 1.0
    while g(t_hi) <= 0.0:
        t_hi *= 2.0
        if t_hi > 1e6:
            return t_hi
    return brentq(g, t_lo, t_hi, maxiter=200, xtol=1e-14)


def design_t_values(params: ErrorModelParams) -> Dict[str, np.ndarray]:
    consts = compute_explicit_constants(
        params.alpha_l1_norm_of_hamiltonian_terms,
        params.qdrift_total_samples_for_target_evolution,
        params.qpe_ancilla_qubits_count
    )
    qdrift_prefactor = consts["qdrift_prefactor_two_alpha_sq_over_N_sq"]
    qdrift_rate = consts["qdrift_exponent_rate_two_alpha_over_N"]
    qpe_phase_step = consts["qpe_phase_step_size_two_pi_over_two_pow_m"]

    # Compute bounds from explicit formulas
    # Lower bound t_min from QPE: (2pi / 2^m) / t <= eps_max/3  => t >= 3*(2pi / 2^m)/eps_max
    t_min_based_on_qpe = 3.0 * qpe_phase_step / params.target_total_error_for_plot_window

    # Crossover t_star solves (2 alpha^2 / N^2) * t^2 * exp((2 alpha / N) t) = (2pi / 2^m)
    t_star = solve_for_crossover_t_star(qdrift_prefactor, qdrift_rate, qpe_phase_step)

    # Upper bound t_max from qDRIFT: (2 alpha^2 / N^2) * t * exp((2 alpha / N) t) <= eps_max/3
    t_max_based_on_qdrift = solve_for_t_max_given_qdrift_budget(
        qdrift_prefactor, qdrift_rate, params.target_total_error_for_plot_window / 3.0
    )

    # Clamp crossover
    t_star = np.clip(t_star, t_min_based_on_qpe, t_max_based_on_qdrift)

    # Build a 7-6-7 allocation (log-spacing + crossover cluster)
    left_upper = max(t_min_based_on_qpe, min(0.5 * t_star, t_max_based_on_qdrift))
    if left_upper <= t_min_based_on_qpe * (1 + 1e-12):
        small_regime_values = np.array([])
    else:
        small_regime_values = np.geomspace(t_min_based_on_qpe, left_upper, num=7)

    crossover_cluster = np.array([0.70, 0.85, 1.00, 1.18, 1.40, 1.70]) * t_star
    crossover_cluster = np.clip(crossover_cluster, t_min_based_on_qpe, t_max_based_on_qdrift)

    right_lower = max(1.5 * t_star, t_min_based_on_qpe)
    if right_lower >= t_max_based_on_qdrift * (1 - 1e-12):
        large_regime_values = np.array([])
    else:
        large_regime_values = np.geomspace(right_lower, t_max_based_on_qdrift, num=7)

    t_values = np.unique(np.concatenate([small_regime_values, crossover_cluster, large_regime_values]))

    # Adjust to desired length
    n = params.number_of_t_values_to_return
    if t_values.size > n:
        idx = np.linspace(0, t_values.size - 1, n).round().astype(int)
        t_values = t_values[idx]
    elif t_values.size < n:
        pad = np.geomspace(t_min_based_on_qpe, t_max_based_on_qdrift, num=n)
        t_values = np.unique(np.concatenate([t_values, pad]))
        if t_values.size > n:
            idx = np.linspace(0, t_values.size - 1, n).round().astype(int)
            t_values = t_values[idx]

    # Jitter to avoid QPE aliasing mid-bin pathologies
    rng = np.random.default_rng(12345)
    t_values = t_values * (1.0 + rng.uniform(-1e-3, 1e-3, size=t_values.shape))

    return dict(
        t_values=np.sort(t_values),
        t_star=t_star,
        t_min=t_min_based_on_qpe,
        t_max=t_max_based_on_qdrift,
        qdrift_prefactor_two_alpha_sq_over_N_sq=qdrift_prefactor,
        qdrift_exponent_rate_two_alpha_over_N=qdrift_rate,
        qpe_phase_step_size_two_pi_over_two_pow_m=qpe_phase_step
    )

def plot_error_behaviour(design: Dict[str, np.ndarray], params: ErrorModelParams, num_plot_pts: int = 1200):
    # Unpack
    qdrift_prefactor = design["qdrift_prefactor_two_alpha_sq_over_N_sq"]
    qdrift_rate = design["qdrift_exponent_rate_two_alpha_over_N"]
    qpe_phase_step = design["qpe_phase_step_size_two_pi_over_two_pow_m"]
    t_min = design["t_min"]
    t_max = design["t_max"]
    t_star = design["t_star"]

    # If caller included sampled t's, ensure the plotting domain covers them
    t_vals = design.get("t_values", None)
    # Base domain intended by design
    base_lo = max(t_min / 3.0, 1e-12)
    base_hi = t_max * 1.5

    # Extend domain to include all sampled points, with 10% padding
    if t_vals is not None and t_vals.size > 0:
        pad_lo = np.min(t_vals) * 0.9
        pad_hi = np.max(t_vals) * 1.1
        t_lo = min(base_lo, pad_lo)
        t_hi = max(base_hi, pad_hi)
    else:
        t_lo, t_hi = base_lo, base_hi

    # Safety on log grid
    t_lo = max(t_lo, 1e-16)
    t_hi = max(t_hi, t_lo * (1.0 + 1e-6))

    # Grid for smooth curves
    t_grid = np.geomspace(t_lo, t_hi, num=num_plot_pts)

    # Component and composite errors
    def qdrift_error_upper_bound(t):
        return qdrift_prefactor * t * np.exp(qdrift_rate * t)

    def qpe_error_upper_bound(t):
        return qpe_phase_step / t

    eq = qdrift_error_upper_bound(t_grid)
    ep = qpe_error_upper_bound(t_grid)
    esum = eq + ep
    erms = np.sqrt(eq**2 + ep**2)

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.8))

    # Plot curves
    ax.loglog(t_grid, ep, color="#1f77b4", lw=2.0, label="QPE: (2π/2^m)/t")
    ax.loglog(t_grid, eq, color="#d62728", lw=2.0, label="qDRIFT: (2α²/N²)·t·e^{(2α/N)t}")
    ax.loglog(t_grid, esum, color="#2ca02c", lw=2.0, label="Sum bound")
    ax.loglog(t_grid, erms, color="#9467bd", lw=1.6, ls="--", label="RMS surrogate")

    # Shaded regions via axvspan (works well on log x)
    band_lo = params.crossover_visual_band_multipliers[0] * t_star
    band_hi = params.crossover_visual_band_multipliers[1] * t_star
    # Clip bands to current domain
    a_lo = max(t_lo, min(band_lo, t_hi))
    a_hi = max(t_lo, min(band_hi, t_hi))
    # Shade QPE-dominated (left of band_lo)
    if t_lo < a_lo:
        ax.axvspan(t_lo, a_lo, color="#1f77b4", alpha=0.08, label="QPE-dominated")
    # Shade crossover
    if a_lo < a_hi:
        ax.axvspan(a_lo, a_hi, color="#ff7f0e", alpha=0.10, label="Crossover")
    # Shade qDRIFT-dominated (right of band_hi)
    if a_hi < t_hi:
        ax.axvspan(a_hi, t_hi, color="#d62728", alpha=0.06, label="qDRIFT-dominated")

    # Mark crossover
    if t_lo <= t_star <= t_hi:
        ax.axvline(t_star, color="#ff7f0e", ls=":", lw=2.2, label="t* (crossover)")

    # Scatter the selected t's and include them in axis limits and y-limits
    y_values_at_samples = None
    if t_vals is not None and t_vals.size > 0:
        y_values_at_samples = qdrift_error_upper_bound(t_vals) + qpe_error_upper_bound(t_vals)
        ax.scatter(
            t_vals,
            y_values_at_samples,
            color="k", s=18, zorder=5, label="Selected t"
        )

    # Final axis limits (explicit), using both grid and sample points for y-range
    # X-limits already fixed by t_lo, t_hi
    ax.set_xlim(left=t_lo, right=t_hi)

    # Compute y-limits from curves and sampled points
    y_all = [ep, eq, esum, erms]
    y_min = min(np.min(arr[np.isfinite(arr)]) for arr in y_all)
    y_max = max(np.max(arr[np.isfinite(arr)]) for arr in y_all)
    if y_values_at_samples is not None and y_values_at_samples.size > 0:
        y_min = min(y_min, float(np.min(y_values_at_samples[y_values_at_samples > 0])))
        y_max = max(y_max, float(np.max(y_values_at_samples)))

    # Pad the y-range a bit (log scale-safe)
    ax.set_ylim(bottom=y_min * 0.7, top=y_max * 1.3)

    ax.set_xlabel("t")
    ax.set_ylabel("Error scale (upper-bound models)")
    ax.set_title("Expected error behavior vs t (qDRIFT + QPE)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    params = ErrorModelParams(
        alpha_l1_norm_of_hamiltonian_terms=4.0,
        qdrift_total_samples_for_target_evolution=256,
        qpe_ancilla_qubits_count=10,
        target_total_error_for_plot_window=1e-2,
        number_of_t_values_to_return=20
    )
    design = design_t_values(params)
    print("Suggested t values:\n", design["t_values"])
    print("crossover t* =", design["t_star"])
    print("t_min =", design["t_min"], "t_max =", design["t_max"])
    plot_error_behaviour(design, params)