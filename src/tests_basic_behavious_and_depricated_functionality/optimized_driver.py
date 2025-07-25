"""
Parameter sweep for the *optimised* qDRIFT–QPE implementation.
Produces a CSV that contains the same rich set of statistical indicators
"""
from __future__ import annotations
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))  # add root to path
from typing import Dict, List, Optional, Tuple, Callable, Sequence
from functools import wraps, cache, cached_property, partial
import csv, datetime, itertools, json, os, pathlib, statisticxs, time
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count
from typing_extensions import runtime
from memory_profiler import memory_usage
import tracemalloc, time, psutil, threading, time, os

import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate

from src.algorithms.algos_optimized_qft_qpe_qdrift import prepare_eigenstate_circuit, make_pauli_gate_cache, build_template_circuit, build_qdrift_trajectory

from src.algorithms.algos_qft_qpe_qdrift_latest import generate_ising_hamiltonian

@dataclass
class QPEResult:
    ham             : str
    n_sys           : int
    n_anc           : int
    time            : float
    segments        : int
    replication_seed            : int
    n_circuits     : int
    n_shots           : int
    depth           : int
    size            : int
    peak_MB         : float  # peak memory usage in MB
    runtime         : float  # runtime in seconds
    exact_eig       : float
    most_likely_bs  : str
    est_energy_med  : float
    est_energy_mean : float
    est_energy_std  : float
    est_energy_min  : float
    est_energy_max  : float
    estimation_error : float

    counts          : str  # json-encoded

templateCircuitKey = tuple[str, int, Optional[bool]]

# ════════════════════════════════════════════════════════════════════════════
#  Parameter grid and constants
# ════════════════════════════════════════════════════════════════════════════
NUM_SYSTEM_QUBITS  = 2
ISING_J, ISING_G = 0.5, 0.4
PLACEHOLDER = "_I_"

HAMILTONIANS_TO_TEST: dict[str, SparsePauliOp] = {
    "ising" : generate_ising_hamiltonian(NUM_SYSTEM_QUBITS, ISING_J, ISING_G),
    "Z-field" : SparsePauliOp.from_list([("Z"*NUM_SYSTEM_QUBITS, np.pi/2)])
}

NUM_ANCILLA  = [12, 14, 16]  # number of ancilla qubits
TIMES     = np.linspace(0.001, 1, 100) # np.logspace(base=2, start=-10, stop=1, num=100, endpoint=True)
NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE  = [1]
RANDOM_CIRCUITS_PER_DATAPOINT = [100]
SHOTS_PER_CIRCUIT = [1]
REPLICATION_SEEDS = [42] # the same seed is used for all circuits in one data point. if more than 1 seed is given, the number of circuits is multiplied by the number of seeds.
ESTIMATE_GROUND_STATE = [False]  # whether to estimate the smallest eigenvalue (ground state). If False we pick the largest eigenvalue (excited state).

# ════════════════════════════════════════════════════════════════════════════
#  # memoised, fork-safe factories of circuit templates and PauliEvolutionGates
# ════════════════════════════════════════════════════════════════════════════

PAULI_CACHE : dict[str, dict[str,PauliEvolutionGate]] = {}
TEMPLATE_CIRCUITS    : dict[templateCircuitKey, QuantumCircuit]    = {}

'''
def _ensure_caches(ham_key: str, H: SparsePauliOp, n_anc   : int, ground_state: bool) -> None:
    if ham_key not in PAULI_CACHE:
        # cache of single–τ PauliEvolutionGates (controlled later on)
        PAULI_CACHE[ham_key], _ = make_pauli_gate_cache(H, PLACEHOLDER)

    if (ham_key, n_anc, ground_state) not in TEMPLATE_CIRCUITS:
        eigvals, eigvecs = np.linalg.eig(H.to_matrix())
        eigenstate_index = np.argmin(eigvals.real) if ground_state else np.argmax(eigvals.real)
        eigenstate = eigvecs[:, eigenstate_index]   # pick the ground state or excited state
        eigenstate_circuit = prepare_eigenstate_circuit(eigenstate)

        # one static template that still contains PLACEHOLDER gates
        TEMPLATE_CIRCUITS[(ham_key, n_anc, ground_state)] = build_template_circuit(
            n_anc               = n_anc,
            n_sys               = NUM_SYSTEM_QUBITS,
            placeholder_label   = PLACEHOLDER,
            eigenvalue_circuit  = eigenstate_circuit,
            gate_cache          = PAULI_CACHE[ham_key]
        )

        print(f"Prepared template circuit for {ham_key} with {n_anc} ancillas for process {os.getpid()}.")
'''
        

@cache # this decorator is fork-safe and works with multiprocessing
def pauli_cache(ham_key: str) -> dict[str, PauliEvolutionGate]:
    """
    Cache of single-τ PauliEvolutionGates for a given Hamiltonian.
    One instance per *process* thanks to functools.cache.
    """
    if ham_key not in PAULI_CACHE:
        H = HAMILTONIANS_TO_TEST[ham_key]
        PAULI_CACHE[ham_key], tau = make_pauli_gate_cache(H, PLACEHOLDER)
        print(f"Prepared PauliEvolutionGate cache for {ham_key} with {len(PAULI_CACHE[ham_key])} gates for process {os.getpid()}.")
    return PAULI_CACHE[ham_key]

@cache
def template_circuit(ham_key: str, n_anc: int, ground_state: bool) -> QuantumCircuit:
    """
    Heavy-weight template circuit that is re-used for every trajectory
    with identical (ham_key, n_anc, ground_state).
    """
    if (ham_key, n_anc, ground_state) not in TEMPLATE_CIRCUITS:
        H = HAMILTONIANS_TO_TEST[ham_key]
        eigvals, eigvecs = np.linalg.eig(H.to_matrix())
        eigenstate_index = np.argmin(eigvals.real) if ground_state else np.argmax(eigvals.real)
        eigenstate = eigvecs[:, eigenstate_index]   # pick the ground state or excited state
        eigenstate_circuit = prepare_eigenstate_circuit(eigenstate)
        # one static template that still contains PLACEHOLDER gates
        TEMPLATE_CIRCUITS[(ham_key, n_anc, ground_state)] = build_template_circuit(
            n_anc               = n_anc,
            n_sys               = NUM_SYSTEM_QUBITS,
            placeholder_label   = PLACEHOLDER,
            eigenvalue_circuit  = eigenstate_circuit,
            exponentiated_hamiltonian_terms_cache          = pauli_cache(ham_key)
        )
    else:
        print(f"Re-using template circuit for {ham_key} with {n_anc} ancillas for process {os.getpid()}.")
    return TEMPLATE_CIRCUITS[(ham_key, n_anc, ground_state)].copy()


# ════════════════════════════════════════════════════════════════════════════
# Local resources that should exist exactly once per worker
# ════════════════════════════════════════════════════════════════════════════
class LocalResources:
    @cached_property
    def backend(self):
        return AerSimulator(method="automatic")

_LOCAL = LocalResources()

# =========================================================================
#  Lightweight, multiprocessing-friendly memory profiler
# =========================================================================

def _peak_rss_during(fn, *, dt: float = 0.05):
    """
    Run `fn()` and return    (result, peak_RSS_in_MiB).

    The RSS (resident-set size) is sampled inside the *same* process,
    hence no fork is required – fully compatible with daemon workers.

    Parameters
    ----------
    fn : Callable[[], T]
        Workload whose memory profile we want to observe.
    dt : float, default 0.05
        Sampling period in seconds. 50 ms gives <1 % CPU overhead while
        detecting peaks that last a few scheduler quanta.

    Notes
    -----
    • We use a daemon `threading.Thread` because daemonic processes are
      not allowed to start child *processes*.
    • The value returned is the true high-water-mark of resident memory
      (not Python allocations only).  Interpreting RSS still requires
      caution, similar to how the term RSS can be mis-read in other
      domains such as solar-activity proxies [1].

    Returns
    -------
    (T, float)
        The original return value of `fn` and the peak RSS in MiB.
    """
    proc        = psutil.Process()
    peak_bytes  = 0
    stop_signal = threading.Event()

    def poll():
        nonlocal peak_bytes
        while not stop_signal.is_set():
            rss_now = proc.memory_info().rss
            if rss_now > peak_bytes:
                peak_bytes = rss_now
            time.sleep(dt)

    sampler = threading.Thread(target=poll, daemon=True)
    sampler.start()
    try:
        retval = fn()
    finally:
        stop_signal.set()
        sampler.join()

    return retval, peak_bytes / 1024**2   # bytes → MiB


def profile_mp(func):
    @wraps(func)
    def _w(*a, **k):
        tracemalloc.start()
        t0 = time.perf_counter()
        result, peak_psutil = _peak_rss_during(lambda: func(*a, **k))
        runtime = time.perf_counter() - t0

        # ─── extract top-N allocation sites ────────────────────────
        snapshot = tracemalloc.take_snapshot()
        stats    = snapshot.statistics("lineno")[:10]     # top-10
        top10    = "; ".join(f"{st.traceback[0]}: {st.size/1024:.1f} KiB"
                             for st in stats)
        result.top10_py_alloc  = top10
        result.peak_MB         = peak_psutil
        result.runtime_s       = runtime
        tracemalloc.stop()
        return result
    return _w

# ════════════════════════════════════════════════════════════════════════════
# post-processing helpers
# ════════════════════════════════════════════════════════════════════════════
def analyse_counts(counts: dict[str,int],
                   t: float,
                   m: int) -> tuple[str, float,float,float,float,float,float]:
    total      = sum(counts.values())
    ml_bitstr  = max(counts, key=counts.get)
    # translate bitstrings ↦ energies
    energies_weighted = []
    for bs, c in counts.items():
        phase = int(bs, 2) / 2**m
        if phase >= .5:                         # map to (-.5,.5]
            phase -= 1
        energy = 2*np.pi*phase / t
        energies_weighted += [energy]*c
    e_med  = statistics.median(energies_weighted)
    e_mean = statistics.mean  (energies_weighted)
    e_std  = statistics.stdev (energies_weighted) if len(energies_weighted)>1 else 0
    e_min, e_max = min(energies_weighted), max(energies_weighted)
    return ml_bitstr, e_med, e_mean, e_std, e_min, e_max


# =========================================================================
#   worker function – executed inside each worker process
# =========================================================================

@profile_mp
def run_simulation(experimental_conditions: dict[str, object]) -> QPEResult:
    """Run one data-point of the parameter sweep."""
    print("Running worker from process", os.getpid())

    # ─── shorthand variables ─────────────────────────────────────
    ham_key      = experimental_conditions["ham"]
    n_anc        = experimental_conditions["anc"]
    ground_state = experimental_conditions["ground_state"]
    n_circuits   = experimental_conditions["circuits"]
    shots        = experimental_conditions["shots"]
    total_time   = experimental_conditions["time"]

    # ─── static Hamiltonian info (used only for ground-truth) ────
    H       = HAMILTONIANS_TO_TEST[ham_key]
    eigvals = np.linalg.eigvals(H.to_matrix()).real
    exact_eig = float(np.min(eigvals) if ground_state else np.max(eigvals))

    # ─── random-seed hierarchy ───────────────────────────────────
    root_ss  = np.random.SeedSequence(experimental_conditions["replication_seed"])
    child_ss = root_ss.spawn(n_circuits)


    total_counts: dict[str, int] = {}
    circ_sample: QuantumCircuit | None = None

    for ss in child_ss:
        rng   = np.random.default_rng(ss)
        dummy_qc = template_circuit(ham_key, n_anc, ground_state) #just to ensure the template is built
        qc    = build_qdrift_trajectory(n_anc=n_anc,
                                        h_signature=ham_key,
                                        total_time=total_time,
                                        H=H,
                                        rng=rng,
                                        n_qdrift_segments=1,
                                        placeholder_label=PLACEHOLDER,
                                        template_circuit= TEMPLATE_CIRCUITS,
                                        exponentialed_hamiltonian_terms_cache= pauli_cache(ham_key),
                                        ground_state=ground_state
        )
        qc_t  = transpile(qc, backend=_LOCAL.backend)

        sim_seed = ss.generate_state(1)[0]
        counts_i = _LOCAL.backend.run(
            qc_t, shots=shots, seed_simulator=int(sim_seed)
        ).result().get_counts()

        # merge counts
        for k, v in counts_i.items():
            total_counts[k] = total_counts.get(k, 0) + v

        circ_sample = circ_sample or qc_t

    # ─── post-processing ─────────────────────────────────────────
    ml_bs, e_med, e_mean, e_std, e_min, e_max = analyse_counts(
        total_counts, total_time, n_anc
    )
    error = abs(exact_eig - e_med)

    # ─── package result ──────────────────────────────────────────
    return QPEResult(
        ham              = ham_key,
        n_sys            = NUM_SYSTEM_QUBITS,
        n_anc            = n_anc,
        time             = total_time,
        segments         = experimental_conditions["segments"],  # still useful meta-data
        replication_seed = experimental_conditions["replication_seed"],
        n_circuits       = n_circuits,
        n_shots          = shots,
        depth            = circ_sample.depth() if circ_sample else 0,
        size             = circ_sample.size()  if circ_sample else 0,
        peak_MB          = 0.0,     # overwritten by @profile_mp
        runtime          = 0.0,     # overwritten by @profile_mp
        exact_eig        = exact_eig,
        most_likely_bs   = ml_bs,
        est_energy_med   = e_med,
        est_energy_mean  = e_mean,
        est_energy_std   = e_std,
        est_energy_min   = e_min,
        est_energy_max   = e_max,
        estimation_error = error,
        counts           = json.dumps(total_counts, sort_keys=True),
    )

# =========================================================================
# driving script – build the grid and launch a Pool
# =========================================================================
def main() -> None:
    # full Cartesian product of all sweep parameters
    grid = itertools.product(
        HAMILTONIANS_TO_TEST.keys(),
        NUM_ANCILLA,
        TIMES,
        NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE,
        REPLICATION_SEEDS,            # outer repetition
        RANDOM_CIRCUITS_PER_DATAPOINT,
        SHOTS_PER_CIRCUIT,
        ESTIMATE_GROUND_STATE         # whether to estimate the ground state
    )
    # serialise each tuple into a plain dict for _run
    cfgs = [dict(ham              = g[0],
                 anc              = g[1],
                 time             = float(g[2]),
                 segments         = g[3],
                 replication_seed = g[4],
                 circuits         = g[5],
                 shots            = g[6],
                 ground_state     = g[7])
            for g in grid]

    csv_path = pathlib.Path(
        f"qdrift_qpe_stats_{datetime.datetime.today():%Y-%m-%d}.csv"
    )

    # write header once
    if not csv_path.exists():
        with csv_path.open("w", newline="") as fh:
            csv.DictWriter(
                fh,
                fieldnames = QPEResult.__dataclass_fields__.keys()
            ).writeheader()

    # run the sweep in parallel
    n_proc = min(os.cpu_count() or 1, 16)
    with Pool(processes=n_proc) as pool:
        print(f"Running {len(cfgs)} configurations in parallel on {pool._processes} workers.")
        for result in pool.imap_unordered(run_simulation, cfgs):
            with csv_path.open("a", newline="") as fh:
                csv.DictWriter(
                    fh,
                    fieldnames = QPEResult.__dataclass_fields__.keys()
                ).writerow(asdict(result))

# entry-point
if __name__ == "__main__":
    main()