import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from scipy.optimize import brentq

# Error models
def qdrift_error(t: np.ndarray, alpha: float, N: int) -> np.ndarray:
    return (2.0 * alpha**2 / N**2) * t * np.exp((2.0 * alpha / N) * t)

def qpe_error(t: np.ndarray, m: int) -> np.ndarray:
    return (2.0 * np.pi / (2**m)) / t

def sum_error(t: np.ndarray, alpha: float, N: int, m: int) -> np.ndarray:
    return qdrift_error(t, alpha, N) + qpe_error(t, m)

def rms_surrogate(t: np.ndarray, alpha: float, N: int, m: int) -> np.ndarray:
    eq = qdrift_error(t, alpha, N)
    ep = qpe_error(t, m)
    return np.sqrt(eq**2 + ep**2)

@dataclass
class RegimeConfig:
    dominance_ratio: float = 3.0 # ratio between qpe and qdrift error at crossover point
    t_min_search: float = 1e-9 # minimum t to consider when searching for regime boundaries
    t_max_search: float = 1e9 # maximum t to consider when searching for regime boundaries
    max_bracket_iter: int = 2000 # max iterations when trying to bracket root
    grow_factor: float = 2.0 # factor by which to grow the search interval
    grid_points_for_plot: int = 2000 # number of grid points for plotting
    xpad_fraction: float = 0.15 # padding fraction for x-axis
    ypad_fraction: float = 0.25 # padding fraction for y-axis
    show_annotations: bool = True # whether to show annotations on the plot
    random_seed: int = 12345

def _scan_bracket(f, t_lo: float, t_hi_cap: float, grow: float, max_iter: int):
    t_hi = max(2.0 * t_lo, 1.0)
    f_lo = f(t_lo)
    f_hi = f(t_hi)
    it = 0
    while np.sign(f_hi) == np.sign(f_lo) and t_hi < t_hi_cap and it < max_iter:
        t_hi *= grow
        f_hi = f(t_hi)
        it += 1
    if np.sign(f_hi) == np.sign(f_lo):
        return None
    return (t_lo, t_hi)

def _solve_for_t_where_At2expct_equals_K(alpha: float, N: int, K: float, cfg: RegimeConfig) -> Optional[float]:
    A = 2.0 * alpha**2 / N**2
    c = 2.0 * alpha / N
    def f(t): return A * (t**2) * np.exp(c * t) - K
    br = _scan_bracket(f, cfg.t_min_search, cfg.t_max_search, cfg.grow_factor, cfg.max_bracket_iter)
    if br is None:
        return None
    try:
        return float(brentq(f, br[0], br[1], maxiter=1000, xtol=1e-14))
    except Exception:
        return None

def find_regime_boundaries(alpha: float, m: int, N: int, cfg: RegimeConfig) -> Dict[str, Optional[float]]:
    B = 2.0 * np.pi / (2**m)  # balance constant
    rho = float(cfg.dominance_ratio)
    t_qpe_edge = _solve_for_t_where_At2expct_equals_K(alpha, N, B / rho, cfg)
    t_star     = _solve_for_t_where_At2expct_equals_K(alpha, N, B,       cfg)
    t_qd_edge  = _solve_for_t_where_At2expct_equals_K(alpha, N, B * rho, cfg)
    return dict(t_qpe_edge=t_qpe_edge, t_star=t_star, t_qd_edge=t_qd_edge)

def _fmt_t(val: Optional[float]) -> str:
    if val is None or not np.isfinite(val):
        return "n/a"
    s = f"{val:.3e}"
    m, e = s.split("e")
    return f"{m}e{int(e):+d}"

def _compute_plot_domain(edges: Dict[str, Optional[float]],
                         cfg: RegimeConfig,
                         selected_t: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Robust domain builder:
    - gather anchors from finite edges, selected_t (if any), and cfg caps,
    - apply multiplicative padding,
    - enforce t_max > t_min, with a minimal factor gap if needed.
    """
    finite_edges = [x for x in edges.values() if x is not None and np.isfinite(x)]
    anchors_low = [cfg.t_min_search]
    anchors_high = [cfg.t_max_search]

    if finite_edges:
        anchors_low.append(min(finite_edges))
        anchors_high.append(max(finite_edges))
    if selected_t is not None and selected_t.size > 0:
        anchors_low.append(float(np.min(selected_t)))
        anchors_high.append(float(np.max(selected_t)))

    t_min_anchor = min(anchors_low)
    t_max_anchor = max(anchors_high)

    # multiplicative padding (asymmetrically: shrink left, expand right)
    t_min_plot = max(cfg.t_min_search, t_min_anchor / (1.0 + cfg.xpad_fraction))
    t_max_plot = max(cfg.t_max_search, t_max_anchor * (1.0 + cfg.xpad_fraction))

    # Ensure a sensible gap
    if not np.isfinite(t_min_plot) or not np.isfinite(t_max_plot) or t_max_plot <= t_min_plot:
        # Fallback: at least one decade
        t_min_plot = max(cfg.t_min_search, t_min_anchor)
        t_max_plot = max(t_min_plot * 10.0, min(cfg.t_max_search, t_max_anchor * 2.0))

    # Final sanity (strictly positive, reasonable ratio)
    t_min_plot = max(t_min_plot, 1e-16)
    if t_max_plot < 1.2 * t_min_plot:
        t_max_plot = 1.2 * t_min_plot

    return (t_min_plot, t_max_plot)

def plot_errors_with_shading_and_edge_labels(alpha: float, m: int, N: int,
                                             cfg: RegimeConfig = RegimeConfig(),
                                             selected_t: Optional[np.ndarray] = None) -> Dict[str, Optional[float]]:
    # 1) Find regime edges
    edges = find_regime_boundaries(alpha, m, N, cfg)
    t_qpe_edge = edges["t_qpe_edge"]
    t_star     = edges["t_star"]
    t_qd_edge  = edges["t_qd_edge"]

    # 2) Compute robust domain from edges + any selected_t you provide
    t_min, t_max = _compute_plot_domain(edges, cfg, selected_t)

    # 3) Curves
    t_grid = np.geomspace(t_min, t_max, cfg.grid_points_for_plot)
    ep = qpe_error(t_grid, m)
    eq = qdrift_error(t_grid, alpha, N)
    esum = ep + eq
    erms = np.sqrt(ep**2 + eq**2)

    # 4) Plot
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.2))
    ax.loglog(t_grid, ep,  color="#1f77b4", lw=2.0, label=r"$E_{\mathrm{QPE}}(t)$")
    ax.loglog(t_grid, eq,  color="#d62728", lw=2.0, label=r"$E_{\mathrm{qDRIFT}}(t)$")
    ax.loglog(t_grid, esum, color="#2ca02c", lw=2.0, label=r"$E_{\mathrm{sum}}(t)$")
    ax.loglog(t_grid, erms, color="#9467bd", lw=1.8, ls="--", label=r"$E_{\mathrm{rms}}(t)$")

    # 5) Shading (QPE-dominated, crossover, qDRIFT-dominated)
    def shade(x0, x1, color, alpha_s, label=None):
        if x0 is not None and x1 is not None and x1 > x0:
            lo = max(t_min, x0)
            hi = min(t_max, x1)
            if hi > lo:
                ax.axvspan(lo, hi, color=color, alpha=alpha_s, label=label)

    if t_qpe_edge is not None:
        shade(t_min, t_qpe_edge, "#1f77b4", 0.08, "QPE-dominated")
    if t_qpe_edge is not None and t_qd_edge is not None and t_qd_edge > t_qpe_edge:
        shade(t_qpe_edge, t_qd_edge, "#ff7f0e", 0.10, "Crossover")
    if t_qd_edge is not None:
        shade(t_qd_edge, t_max, "#d62728", 0.06, "qDRIFT-dominated")

    # 6) Edge lines + numeric labels
    def vline_with_label(x, color, text):
        if x is not None and np.isfinite(x) and (t_min < x < t_max):
            ax.axvline(x, color=color, ls=":", lw=2.2)
            if cfg.show_annotations:
                ymin, ymax = ax.get_ylim()
                ax.text(x, ymax, text, color=color, rotation=90, va="top", ha="right",
                        fontsize=9, backgroundcolor="white", clip_on=True)

    vline_with_label(t_qpe_edge, "#1f77b4", rf"$t_{{\mathrm{{QPE\,edge}}}}={_fmt_t(t_qpe_edge)}$")
    vline_with_label(t_star,     "#ff7f0e", rf"$t^*={_fmt_t(t_star)}$")
    vline_with_label(t_qd_edge,  "#d62728", rf"$t_{{\mathrm{{qDRIFT\,edge}}}}={_fmt_t(t_qd_edge)}$")

    # 7) Selected points (if any)
    if selected_t is not None and selected_t.size > 0:
        y_vals = sum_error(selected_t, alpha, N, m)
        ax.scatter(selected_t, y_vals, color="k", s=24, zorder=5, label="Selected t")

    # 8) Final limits
    y_all = np.concatenate([ep, eq, esum, erms])
    y_min = np.min(y_all[np.isfinite(y_all)])
    y_max = np.max(y_all[np.isfinite(y_all)])
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(y_min * (1.0 - cfg.ypad_fraction), y_max * (1.0 + cfg.ypad_fraction))

    ax.set_xlabel("t")
    ax.set_ylabel("Error scale")
    ax.set_title("qDRIFT vs QPE error models and regimes\n"
                 + rf"$t_{{\mathrm{{QPE\,edge}}}}={_fmt_t(t_qpe_edge)}$, "
                 + rf"$t^*={_fmt_t(t_star)}$, "
                 + rf"$t_{{\mathrm{{qDRIFT\,edge}}}}={_fmt_t(t_qd_edge)}$")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()

    edges_dict = dict(
        t_qpe_edge=t_qpe_edge,
        t_star=t_star,
        t_qd_edge=t_qd_edge
    )
    return edges_dict, (t_min, t_max)

def sample_t_from_regimes(alpha: float, m: int, N: int,
                          total_samples: int,
                          cfg: RegimeConfig = RegimeConfig(),
                          sampling_allocation: Tuple[int, int, int] = None) -> Dict[str, np.ndarray]:
    """
    Randomly sample t's from each regime and return concatenation.
    Regimes:
      - QPE-dominated: (t_min_domain .. t_qpe_edge)
      - Crossover    : (t_qpe_edge .. t_qd_edge)
      - qDRIFT-dom.  : (t_qd_edge .. t_max_domain)
    If a regime interval is absent or degenerate, its quota is redistributed to available regimes.

    @param total_samples: total number of t samples to draw
    @param sampling_allocation: optional tuple of three integers specifying how many samples to draw from each regime.
                              If None, samples are split roughly equally among available regimes.
    """
    rng = np.random.default_rng(cfg.random_seed)
    edges, (t_min_dom, t_max_dom) = plot_errors_with_shading_and_edge_labels(alpha, m, N, cfg)

    t_qpe_edge = edges["t_qpe_edge"]
    t_qd_edge  = edges["t_qd_edge"]

    # Intervals (open on both ends for randomness; clamp safely)
    intervals = {
        "qpe_dominated": None,
        "crossover": None,
        "qdrift_dominated": None
    }

    # Build intervals if edges exist properly
    if t_qpe_edge is not None and t_qpe_edge > t_min_dom:
        intervals["qpe_dominated"] = (t_min_dom, t_qpe_edge)

    if (t_qpe_edge is not None) and (t_qd_edge is not None) and (t_qd_edge > t_qpe_edge):
        intervals["crossover"] = (t_qpe_edge, t_qd_edge)

    if t_qd_edge is not None and t_max_dom > t_qd_edge:
        intervals["qdrift_dominated"] = (t_qd_edge, t_max_dom)

    # Default allocation: split roughly equally
    if sampling_allocation is None:
        base = total_samples // 3
        rem = total_samples - 3 * base
        sampling_allocation = (base + (1 if rem > 0 else 0),
                               base + (1 if rem > 1 else 0),
                               base)

    # Re-map allocations if some intervals are missing
    names = ["qpe_dominated", "crossover", "qdrift_dominated"]
    quotas = dict(zip(names, sampling_allocation))
    available = [k for k, v in intervals.items() if v is not None]
    missing = [k for k in names if intervals[k] is None]

    if len(available) == 0:
        # fallback: sample uniformly in the plotting domain
        t_all = rng.uniform(low=t_min_dom, high=t_max_dom, size=total_samples)
        return dict(
            t_samples=np.sort(t_all),
            intervals=intervals,
            edges=edges,
            domain=(t_min_dom, t_max_dom)
        )

    # Redistribute missing quotas to available ones
    extra = sum(quotas[k] for k in missing)
    for k in missing:
        quotas[k] = 0
    # even split extra among available
    for i, k in enumerate(available):
        add = extra // len(available) + (1 if i < (extra % len(available)) else 0)
        quotas[k] += add

    # Now sample
    samples = []
    for k in names:
        q = quotas[k]
        if q > 0 and intervals[k] is not None:
            a, b = intervals[k]
            # sample log-uniform within regime to cover orders of magnitude
            log_a, log_b = np.log(a), np.log(b)
            u = rng.uniform(low=log_a, high=log_b, size=q)
            samples.append(np.exp(u))

    t_samples = np.sort(np.concatenate(samples)) if len(samples) else np.array([])
    return dict(
        t_samples=t_samples,
        intervals=intervals,
        edges=edges,
        domain=(t_min_dom, t_max_dom),
        quotas=quotas
    )

def _safe_clip(arr, max_val=1e300):
    """Clip array to avoid inf/nan from overflow for plotting."""
    arr = np.nan_to_num(arr, nan=max_val, posinf=max_val, neginf=1e-300)
    return np.clip(arr, 1e-300, max_val)  # keep positive, bounded


def plot_sim_results_with_regimes(df, alpha: float, m: int, N: int,
                                  dominance_ratio: float = 3.0,
                                  show_annotations: bool = True,
                                  include_rms: bool=False):
    """
    Plot simulation results + theoretical error models with regime shading.
    Less dependent on external config, more on actual data.
    """

    # --- 1. Regime boundaries (use data-driven min/max for search) ---
    t_min_data, t_max_data = df["time"].min(), df["time"].max()
    t_min_search = max(1e-12, t_min_data * 0.5)
    t_max_search = t_max_data * 2.0

    class SimpleCfg:
        def __init__(self):
            self.dominance_ratio = dominance_ratio
            self.t_min_search = t_min_search
            self.t_max_search = t_max_search
            self.grow_factor = 2.0
            self.max_bracket_iter = 2000
            self.show_annotations = show_annotations
    cfg = SimpleCfg()

    edges = find_regime_boundaries(alpha, m, N, cfg)
    t_qpe_edge, t_star, t_qd_edge = edges["t_qpe_edge"], edges["t_star"], edges["t_qd_edge"]

    # --- 2. Compute curves over data-driven domain ---
    t_min = min(t_min_data, min(x for x in edges.values() if x is not None))
    t_max = max(t_max_data, max(x for x in edges.values() if x is not None))
    t_grid = np.geomspace(t_min, t_max, 2000)

    ep = _safe_clip(qpe_error(t_grid, m))
    eq = _safe_clip(qdrift_error(t_grid, alpha, N))
    esum = _safe_clip(ep + eq)
    erms = _safe_clip(np.sqrt(ep**2 + eq**2))

    # --- 3. Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    ax.loglog(t_grid, ep,   color="#1f77b4", lw=2.0, label=r"$E_{\mathrm{QPE}}(t) = \frac{2 \pi}{2^{m}} \frac{1}{t}$")
    ax.loglog(t_grid, eq,   color="#d62728", lw=2.0, label=r"$E_{\mathrm{qDRIFT}}(t)$")
    ax.loglog(t_grid, esum, color="#2ca02c", lw=2.0, label=r"$E_{\mathrm{sum}}(t)$")
    if include_rms:
        ax.loglog(t_grid, erms, color="#9467bd", lw=1.8, ls="--", label=r"$E_{\mathrm{rms}}(t)$")

    # --- 4. Shading regimes ---
    def shade(x0, x1, color, alpha_s, label=None):
        if x0 is not None and x1 is not None and x1 > x0:
            ax.axvspan(x0, x1, color=color, alpha=alpha_s, label=label)

    if t_qpe_edge is not None:
        shade(t_min, t_qpe_edge, "#1f77b4", 0.08, "QPE-dominated")
    if t_qpe_edge is not None and t_qd_edge is not None and t_qd_edge > t_qpe_edge:
        shade(t_qpe_edge, t_qd_edge, "#ff7f0e", 0.10, "Crossover")
    if t_qd_edge is not None:
        shade(t_qd_edge, t_max, "#d62728", 0.06, "qDRIFT-dominated")

    # --- 5. Vertical lines ---
    def vline_with_label(x, color, text):
        if x is not None and np.isfinite(x) and (t_min < x < t_max):
            ax.axvline(x, color=color, ls=":", lw=2.2)
            if show_annotations:
                ymin, ymax = ax.get_ylim()
                ax.text(x, ymax, text, color=color, rotation=90, va="top", ha="right",
                        fontsize=9, backgroundcolor="white", clip_on=True)

    vline_with_label(t_qpe_edge, "#1f77b4", rf"$t_{{\mathrm{{QPE\,edge}}}}={_fmt_t(t_qpe_edge)}$")
    vline_with_label(t_star,     "#ff7f0e", rf"$t^*={_fmt_t(t_star)}$")
    vline_with_label(t_qd_edge,  "#d62728", rf"$t_{{\mathrm{{qDRIFT\,edge}}}}={_fmt_t(t_qd_edge)}$")

    # --- 6. Simulation scatter with errorbars ---
    y = df["estimation_error"].values
    yerr_lower = np.minimum(df["est_energy_std"].values, y * 0.999)
    yerr_upper = df["est_energy_std"].values
    yerr = [yerr_lower, yerr_upper]

    ax.errorbar(
        df["time"], y, yerr=yerr,
        fmt='o', ecolor="gray", elinewidth=1.0, capsize=3,
        alpha=0.9, label="Simulation"
    )

    # --- 7. Axis limits: union of data + curves ---
    all_y = np.concatenate([ep, eq, esum, erms, y])
    y_min = np.nanmin(all_y[all_y > 0])
    y_max = np.nanmax(all_y[np.isfinite(all_y)])
    ax.set_xlim(t_min * 0.8, t_max * 1.2)
    ax.set_ylim(y_min * 0.8, y_max * 1.2)

    ax.set_xlabel("t")
    ax.set_ylabel("Error scale")
    ax.set_title("Simulation results with qDRIFT vs QPE error models and regimes")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()

    return edges, (t_min, t_max)
