"""
Parameter sweep for the *optimised* qDRIFT–QPE implementation.
Produces a CSV that contains the same rich set of statistical indicators
"""
from __future__ import annotations
from importlib import metadata
from re import M
import sys
import pathlib
import re

from zmq import device
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))  # add root to path
from typing import Dict, List, Optional, Tuple, Callable, Sequence, NamedTuple
from functools import wraps, cache, cached_property, partial
import csv, datetime, itertools, json, os, pathlib, statistics, time
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count
from memory_profiler import memory_usage
import tracemalloc, time, psutil, threading, time, os
from uuid import uuid4

import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import Parameter

from src.algorithms.optimized_qft_qpe_qdrift import prepare_eigenstate_circuit, make_pauli_gate_cache, build_template_circuit, build_qdrift_trajectory
from src.algorithms.unoptimized_qft_qpe_qdrift import generate_ising_hamiltonian
from src.algorithms.chebyshev import chebyshev_nodes


@dataclass
class QPEResult:
    ham : str
    num_system_qubits : int
    num_ancilla : int
    time : float
    segments : int
    replication_seed : int
    n_circuits : int
    n_shots : int
    peak_MB : float  # peak memory usage in MB
    runtime : float  # runtime in seconds
    exact_eig : float
    most_likely_bs : str
    est_energy_med : float
    est_energy_mean : float
    est_energy_std  : float
    est_energy_min  : float
    est_energy_max  : float
    estimation_error : float
    alpha : float # sum of Hamiltonian coefficients
    max_theoretical_qdrift_error: float

    counts          : str  # json-encoded

# ════════════════════════════════════════════════════════════════════════════
#  Parameter grid and constants
# ════════════════════════════════════════════════════════════════════════════
NUM_SYSTEM_QUBITS  = 2
ISING_J, ISING_G = 1.2, 1
PLACEHOLDER = "_I_"

HAMILTONIANS_TO_TEST: dict[str, SparsePauliOp] = {
    # "exact_ham_exact_qdritf" : SparsePauliOp(data="ZZ", coeffs=np.pi / 4),
    # "H_many_diag_terms" : SparsePauliOp(data=["ZI", "ZZ", "IZ", "II"], coeffs=np.array([1/10, -2/10, 3/10, 4/10])),
    # "H_many_diag_terms_1" : SparsePauliOp(data=["ZI", "ZZ", "IZ", "II"], coeffs=np.array([1/10, 2/10, 3/10, 4/10])),
    "H_many_diag_terms_3" : SparsePauliOp(data=["ZI", "ZZ", "IZ", "II"], coeffs=np.array([4/5, 2/3, 4/5, 4/5])),
    "H_many_diag_terms_0.5" : SparsePauliOp(data=["ZI", "ZZ", "IZ", "II"], coeffs=np.array([2/20, 1/20, 3/20, 4/20])),
    "H_ising" : generate_ising_hamiltonian(num_qubits=NUM_SYSTEM_QUBITS, J=ISING_J * 0.5, g=ISING_G * 0.5) 
}

NUM_ANCILLA  = [14]  # number of ancilla qubits

chebyshev_nodes = np.array(chebyshev_nodes(10))
scaled_nodes_pos = 0.000001 + (0.1 - 0.000001) * chebyshev_nodes[:5]
TIMES     = scaled_nodes_pos #np.logspace(-10, 1, base=2, num=20)
NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE  = [1]
RANDOM_CIRCUITS_PER_DATAPOINT = [100]
SHOTS_PER_CIRCUIT = [1024]
REPORT_PROTOCOL_RESULTS_FROM_ANY_RANDOM_CIRCUIT = [{"group": True, "group_by": "median"}]
REPLICATION_SEEDS = [42] # the same seed is used for all circuits in one data point. if more than 1 seed is given, the number of circuits is multiplied by the number of seeds.
ESTIMATE_GROUND_STATE = [False]  # whether to estimate the smallest eigenvalue (ground state). If False we pick the largest eigenvalue (excited state).
TEST_ID = uuid4()

# ════════════════════════════════════════════════════════════════════════════
#  # memoised, fork-safe factories of circuit templates and PauliEvolutionGates
# ════════════════════════════════════════════════════════════════════════════

class PauliGateCache(NamedTuple):
    gates : dict[str, PauliEvolutionGate]
    tau   : Parameter                     # the *same* symbol for all gates        

@cache # this decorator is fork-safe and works with multiprocessing
def pauli_cache(ham_key: str) -> PauliGateCache:
    """
    Cache of single-τ PauliEvolutionGates for a given Hamiltonian.
    One instance per *process* thanks to functools.cache.
    """
    H = HAMILTONIANS_TO_TEST[ham_key]
    gates, tau = make_pauli_gate_cache(H, PLACEHOLDER)
    return PauliGateCache(gates=gates, tau=tau)

@cache
def template_circuit(ham_key: str, n_anc: int, ground_state: bool) -> QuantumCircuit:
    """
    Heavy-weight template circuit that is re-used for every trajectory
    with identical (ham_key, n_anc, ground_state).
    """
    H = HAMILTONIANS_TO_TEST[ham_key]
    eigvals, eigvecs = np.linalg.eig(H.to_matrix())
    eigenstate_index = np.argmin(eigvals.real) if ground_state else np.argmax(eigvals.real)
    eigenstate = eigvecs[:, eigenstate_index]   # pick the ground state or excited state
    eigenstate_circuit = prepare_eigenstate_circuit(eigenstate)
    # one static template that still contains PLACEHOLDER gates
    qc = build_template_circuit(
        n_anc               = n_anc,
        n_sys               = NUM_SYSTEM_QUBITS,
        placeholder_label   = PLACEHOLDER,
        eigenvalue_circuit  = eigenstate_circuit,
        exponentiated_hamiltonian_terms_cache          = pauli_cache(ham_key)
    )
    return qc


# ════════════════════════════════════════════════════════════════════════════
# Local resources that should exist exactly once per worker
# ════════════════════════════════════════════════════════════════════════════
class LocalResources:
    @cached_property
    def backend(self):
        return AerSimulator(method="matrix_product_state", device="CPU")

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
        result.runtime       = runtime
        tracemalloc.stop()
        return result
    return _w

# ════════════════════════════════════════════════════════════════════════════
# post-processing helpers
# ════════════════════════════════════════════════════════════════════════════
def analyse_counts(counts: dict[str,int],
                   t: float,
                   m: int,
                   wrapp_around_correction: bool=True) -> tuple[str, float,float,float,float,float,float]:
    ml_bitstr  = max(counts, key=counts.get)
    # translate bitstrings ↦ energies
    energies_weighted = []
    for bs, c in counts.items():
        phase = int(bs, 2) / 2**m
        if phase > .5 and wrapp_around_correction:                         # map to (-.5,.5]
            phase -= 1
        energy = 2*np.pi*phase / t
        energies_weighted += [energy]*c
    e_med  = statistics.median(energies_weighted)
    e_mean = statistics.mean  (energies_weighted)
    e_std  = statistics.stdev (energies_weighted) if len(energies_weighted)>1 else 0
    e_min, e_max = min(energies_weighted), max(energies_weighted)
    return ml_bitstr, e_med, e_mean, e_std, e_min, e_max


def int_to_bitstring(value: int, m: int) -> str:
    """
    Convert an integer to a binary bitstring of fixed length `m`.

    Args:
        value (int): The integer to convert. Must be non-negative and less than 2**m.
        m (int): The desired length of the output bitstring.

    Returns:
        str: Binary representation of `value` as a zero-padded bitstring of length `m`.

    Raises:
        ValueError: If `value` is negative or cannot be represented with `m` bits.
    """
    if value < 0:
        raise ValueError("value must be non-negative")
    if value >= 2**m:
        raise ValueError(f"value {value} cannot be represented with {m} bits")

    return format(value, f"0{m}b")

# =========================================================================
#   worker function – executed inside each worker process
# =========================================================================

@profile_mp #custom decorator that profiles memory and runtime
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
    trajectory_report_protocol = experimental_conditions["trajectory_report_protocol"]

    # ─── static Hamiltonian info (used only for ground-truth) ────
    H       = HAMILTONIANS_TO_TEST[ham_key]
    eigvals = np.linalg.eigvals(H.to_matrix()).real
    exact_eig = float(np.min(eigvals) if ground_state else np.max(eigvals))

    # ─── random-seed hierarchy ───────────────────────────────────
    root_ss  = np.random.SeedSequence(experimental_conditions["replication_seed"])
    child_ss = root_ss.spawn(n_circuits)


    total_counts: dict[str, int] = {}
    circ_sample: QuantumCircuit | None = None
    template_circuit_cached = template_circuit(ham_key=ham_key, n_anc=n_anc, ground_state=ground_state)

    for ss in child_ss:
        rng   = np.random.default_rng(ss)
        qc    = build_qdrift_trajectory(n_anc=n_anc,
                                        h_signature=ham_key,
                                        total_time=total_time,
                                        H=H,
                                        rng=rng,
                                        n_qdrift_segments=1,
                                        placeholder_label=PLACEHOLDER,
                                        template_circuit= template_circuit_cached,
                                        exponentialed_hamiltonian_terms_cache=pauli_cache(ham_key),
                                        use_exp_ham_terms_cache=True #TODO: make something about this

        )
        qc_t  = transpile(qc, backend=_LOCAL.backend)

        sim_seed = ss.generate_state(1)[0]
        counts_i = _LOCAL.backend.run(
            qc_t, shots=shots, seed_simulator=int(sim_seed)
        ).result().get_counts()

        if (trajectory_report_protocol["group"] == True) and (shots > 1): # no shorthand pretencious syntax
            keys = np.array(list(counts_i.keys()))
            freqs = np.array(list(counts_i.values()))
            expanded_arrrr = np.repeat(keys, freqs) 
            print("DEBUG: sample values:", expanded_arrrr[:10])
            expanded_arrrr = np.array([int(bs, 2) for bs in expanded_arrrr]) # it's easier to calculate median from a list of ints than from a list of binary bitstrings

            if trajectory_report_protocol["group_by"] == "median":
                # just report the median measured bitstring 
                print("DEBUG: sample values:", expanded_arrrr[:10])
                median = np.median(expanded_arrrr)  # np.median sorts internally, for large number of shots this is inefficient 
                median_int = int(round(median))
                median_bin = int_to_bitstring(value=median_int, m=n_anc)
                counts_i = {median_bin : 1}
                
            elif trajectory_report_protocol["group_by"] == "mean":
                pass
            elif trajectory_report_protocol["group_by"] == "mode":
                pass

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
        num_system_qubits = NUM_SYSTEM_QUBITS,
        num_ancilla            = n_anc,
        time             = total_time,
        segments         = experimental_conditions["segments"],  # still useful meta-data
        replication_seed = experimental_conditions["replication_seed"],
        n_circuits       = n_circuits,
        n_shots          = shots,
        peak_MB          = 0.0,     # overwritten by @profile_mp
        runtime          = 0.0,     # overwritten by @profile_mp
        exact_eig        = exact_eig,
        max_theoretical_qdrift_error = 2 * sum(abs(H.coeffs)) ** 2 * total_time * np.exp(2 * sum(abs(H.coeffs)) * total_time),
        most_likely_bs   = ml_bs,
        est_energy_med   = e_med,
        est_energy_mean  = e_mean,
        est_energy_std   = e_std,
        est_energy_min   = e_min,
        est_energy_max   = e_max,
        estimation_error = error,
        alpha            = sum(abs(H.coeffs)),
        counts           = json.dumps(total_counts, sort_keys=True),
    )

# =========================================================================
# driving script – build the grid and launch a Pool
# =========================================================================
def main(verbose_export = False) -> None:
    # full Cartesian product of all sweep parameters
    grid = itertools.product(
        HAMILTONIANS_TO_TEST.keys(),
        NUM_ANCILLA,
        TIMES,
        NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE,
        REPLICATION_SEEDS,            # outer repetition
        RANDOM_CIRCUITS_PER_DATAPOINT,
        SHOTS_PER_CIRCUIT,
        ESTIMATE_GROUND_STATE,         # whether to estimate the ground state,
        REPORT_PROTOCOL_RESULTS_FROM_ANY_RANDOM_CIRCUIT
    )

    # serialise each tuple into a plain dict for _run
    cfgs = [dict(ham              = g[0],
                 anc              = g[1],
                 time             = float(g[2]),
                 segments         = g[3],
                 replication_seed = g[4],
                 circuits         = g[5],
                 shots            = g[6],
                 ground_state     = g[7],
                 trajectory_report_protocol = g[8]
                 )
            for g in grid]
    
    # dump the full experiment grid to JSON (once)
    basename = f"qdrift_qpe_fc_parameter_sweep_{datetime.datetime.today():%Y-%m-%d}"
    metadata_path = pathlib.Path(f"{basename}.json")
    if verbose_export:
        with metadata_path.open("w") as fh:
            json.dump(
                {"generated_utc": datetime.datetime.today().isoformat(timespec="seconds") + "Z",
                "num_configs": len(cfgs),
                "configs": cfgs},
                fh,
                indent=2,
                sort_keys=True
            )
    else:
        spec = {
        "generated_utc" : datetime.datetime.today().isoformat(timespec="seconds") + "Z",
        "script_version": os.getenv("GIT_COMMIT", "unknown"),
        "metadata": {
            "num_configs": len(cfgs),
            "static_parameters": {
                "num_system_qubits" : NUM_SYSTEM_QUBITS,
                "Random_seed(s)" : REPLICATION_SEEDS
            },
            "sweep_space" : {
                "t" : list(TIMES),
                "num_ancilla" : NUM_ANCILLA,
                "num_qdrift_segments_per_qdrift_channel_invocation" : NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE,
                "num_independent_stochastic_circuits_per_datapoint" : RANDOM_CIRCUITS_PER_DATAPOINT,
                "num_shots_per_circuit": SHOTS_PER_CIRCUIT,
                "Hamiltonians": [{"type" : ty, 
                                  "coeffs" : str(H.coeffs), 
                                  "paulis" : H.paulis.to_labels(),
                                  "eigenvalue to estimate" : np.linalg.eigvals(H.to_matrix()).real.tolist()} for ty, H in HAMILTONIANS_TO_TEST.items()],
                "calculate_ground_state": ESTIMATE_GROUND_STATE
            }
            }
        }
        with metadata_path.open("w") as fh:
            json.dump(
                spec,
                fh,
                indent=2,
                sort_keys=True
            )

    csv_path = pathlib.Path(f"{basename}.csv")
    if not csv_path.exists():
        with csv_path.open("w", newline="") as fh:
            csv.DictWriter(
                fh,
                fieldnames = QPEResult.__dataclass_fields__.keys() # write header only once (and only if the file does not exist)
            ).writeheader()

    # run the sweep in parallel
    n_proc = min(os.cpu_count() or 1, 16)
    with Pool(processes=n_proc - 2 ) as pool:
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