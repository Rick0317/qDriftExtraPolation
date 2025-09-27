
from __future__ import annotations
from importlib import metadata
from re import M
import tracemalloc, time, psutil, threading, sys, csv, datetime, itertools, json, os, pathlib, statistics
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))  # add root to path
from typing import Dict, List, Optional, Tuple, Callable, Sequence, NamedTuple
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count
from memory_profiler import memory_usage
from uuid import uuid4
from functools import cache, cached_property


import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import Parameter

from src.algorithms.optimized_qft_qpe_qdrift import prepare_eigenstate_circuit, make_pauli_gate_cache, build_template_circuit, build_qdrift_trajectory
from src.algorithms.chebyshev import chebyshev_nodes
from src.utils.generate_hamiltonians import calculate_minimum_evolution_time, create_h2_minimal_basis_hamiltonian, generate_ising_hamiltonian
from src.utils.memory_profiling_utils import *
from src.utils.qpe_postprocessing_utils import batch_process_counts, int_to_bitstring, analyse_counts_optimized
from src.utils.utils_io import init_csv, write_metadata, append_csv

@dataclass
class QPEResult:
    ham : str
    exact_eig : float
    num_system_qubits : int
    num_ancilla : int
    alpha : float # sum of Hamiltonian coefficients
    replication_seed : int
    n_circuits : int
    n_shots : int
    time : float
    segments : int
    most_likely_bs : str
    est_energy_med : float
    est_energy_mean : float
    est_energy_std  : float
    estimation_error : float
    max_theoretical_qdrift_error: float
    peak_MB : float  # peak memory usage in MB
    runtime : float  # runtime in seconds
    counts          : str  # json-encoded

# ════════════════════════════════════════════════════════════════════════════
#  Parameter grid and constants
# ════════════════════════════════════════════════════════════════════════════
NUM_SYSTEM_QUBITS  = 1
ISING_J, ISING_G = 1.2, 1
PLACEHOLDER = "<PLACEHOLDER>_I_"

HAMILTONIANS_TO_TEST: dict[str, SparsePauliOp] = {
    # "exact_ham_exact_qdritf" : SparsePauliOp(data="ZZZZ", coeffs=np.pi / 4),
    # "H2_minimal_basis": H_H2,
    # "H_ising" : generate_ising_hamiltonian(num_qubits=NUM_SYSTEM_QUBITS, J=ISING_J * 0.7, g=ISING_G * 0.7) 
    # "1 qubit test": SparsePauliOp.from_list([("X", 0.2), ("Z", 0.5), ("I", 0.3)], num_qubits=1),
    " diagonal 1 qubit": SparsePauliOp.from_list([("I", 0.1), ("Z", -0.2), ("I", 0.4), ("Z", 0.3)])
}

NUM_ANCILLA  = [8]  # number of ancilla qubits

qpe_resolution_limits = calculate_minimum_evolution_time(hamiltonians=HAMILTONIANS_TO_TEST, m=min(NUM_ANCILLA))
print(qpe_resolution_limits)
t_min_global = max(qpe_resolution_limits.values())
lower_bound = max(1e-10, t_min_global * 0.8)  # Don't go below 80% of t_min
upper_bound = min(1e1, t_min_global * 100)    # Don't exceed 100× t_min
TIMES = np.logspace(np.log2(lower_bound), np.log2(upper_bound), base=2, num=12)
NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE  = [1]
RANDOM_CIRCUITS_PER_DATAPOINT = [10, 100, 1000, 10000]
SHOTS_PER_CIRCUIT = [1, 10, 100]
REPORT_PROTOCOL_RESULTS_FROM_ANY_RANDOM_CIRCUIT = [{"group": True, "group_by": "median"}]
REPLICATION_SEEDS = [42] # the same seed is used for all circuits in one data point. if more than 1 seed is given, the number of circuits is multiplied by the number of seeds.
ESTIMATE_GROUND_STATE = [False]  # whether to estimate the smallest eigenvalue (ground state). If False we pick the largest eigenvalue (excited state).
TEST_ID = uuid4()
BATCH_SIZE = 100  # Configurable batch size
KET_0_AS_EIGENSTATE = [True]  # ignore everything else and initialize the circuits with ket 0 state

# ════════════════════════════════════════════════════════════════════════════
#  # memoised, fork-safe factories of circuit templates and PauliEvolutionGates
# ════════════════════════════════════════════════════════════════════════════

class PauliGateCache(NamedTuple):
    gates : dict[str, PauliEvolutionGate]
    tau   : Parameter # the symbolic parameter for evolution time, to be bound later
    pauli_labels : np.ndarray  # Pre-computed labels for fast access
    pmf : np.ndarray  # Pre-computed probability mass function

@cache
def pauli_cache(ham_key: str) -> PauliGateCache:
    """
    Enhanced cache with pre-computed sampling arrays for vectorized operations.
    """
    H = HAMILTONIANS_TO_TEST[ham_key]
    gates, tau = make_pauli_gate_cache(H, PLACEHOLDER)
    
    # Pre-compute sampling arrays
    pauli_labels = np.array(list(H.paulis.to_labels()))
    pmf = np.abs(H.coeffs) / np.sum(np.abs(H.coeffs))
    
    return PauliGateCache(
        gates=gates, 
        tau=tau, 
        pauli_labels=pauli_labels, 
        pmf=pmf
    )

@cache
def template_circuit(ham_key: str, n_anc: int, ground_state: bool, ket_0_as_eigenstate: bool) -> QuantumCircuit:
    """
    Heavy-weight template circuit cached for reuse.
    """
    H = HAMILTONIANS_TO_TEST[ham_key]
    eigvals, eigvecs = np.linalg.eig(H.to_matrix())
    eigenstate_index = np.argmin(eigvals.real) if ground_state else np.argmax(eigvals.real)
    eigenstate = eigvecs[:, eigenstate_index]
    eigenstate_circuit = prepare_eigenstate_circuit(eigenstate) if not ket_0_as_eigenstate else None
    
    qc = build_template_circuit(
        n_anc=n_anc,
        n_sys=NUM_SYSTEM_QUBITS,
        placeholder_label=PLACEHOLDER,
        eigenvalue_circuit=eigenstate_circuit,
        exponentiated_hamiltonian_terms_cache=pauli_cache(ham_key)
    )
    return qc


# ════════════════════════════════════════════════════════════════════════════
# Local resources that should exist exactly once per worker
# ════════════════════════════════════════════════════════════════════════════
class LocalResources:

    @cached_property
    def backend(self):
        # pick this based on what you set below for workers/threads
        max_threads = int(os.environ.get("AER_MAX_THREADS", "4"))
        sim = AerSimulator(method="matrix_product_state", device="CPU", max_parallel_threads=max_threads)
        return sim

_LOCAL = LocalResources()

# =========================================================================
#   worker function – executed inside each worker process
# =========================================================================


@profile_memory_and_time #custom decorator that profiles memory and runtime
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
    ket_0_as_eigenstate = experimental_conditions["exceptionally_stupid_eigenstate"]

    # ─── static info  ────
    H       = HAMILTONIANS_TO_TEST[ham_key]
    eigvals = np.linalg.eigvals(H.to_matrix()).real
    exact_eig = float(np.min(eigvals) if ground_state else np.max(eigvals))
    cache = pauli_cache(ham_key)
    template_circuit_cached = template_circuit(ham_key=ham_key, n_anc=n_anc, ground_state=ground_state, ket_0_as_eigenstate=ket_0_as_eigenstate)

    # ─── random-seed hierarchy ───────────────────────────────────
    root_ss  = np.random.SeedSequence(experimental_conditions["replication_seed"]) # ss stands for SeedSequence
    child_ss = root_ss.spawn(n_circuits)  # one child seed per random circuit
    
    for batch_start in range(0, n_circuits, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_circuits)
        batch_ss = child_ss[batch_start:batch_end]
        batch_circuits: List[QuantumCircuit] = []
        all_counts_from_all_batches: List[dict[str, int]] = []

        for ss in batch_ss:
            rng   = np.random.default_rng(ss)
            qc    = build_qdrift_trajectory(n_anc=n_anc,
                                            h_signature=ham_key,
                                            total_time=total_time,
                                            H=H,
                                            rng=rng,
                                            n_qdrift_segments=1,
                                            placeholder_label=PLACEHOLDER,
                                            template_circuit= template_circuit_cached,
                                            exponentialed_hamiltonian_terms_cache=cache     
            )
            batch_circuits.append(qc)
        
        # Transpile the entire batch at once for efficiency
        transpiled_batch = transpile( batch_circuits, backend=_LOCAL.backend, optimization_level=0, num_processes=1, approximation_degree=0)
        
        # Execute the entire batch at once
        seeds_for_batch = [ss.generate_state(1)[0] for ss in batch_ss]
        job = _LOCAL.backend.run(transpiled_batch, shots=shots, seed_simulation=seeds_for_batch)
        results = job.result()
        all_counts_from_all_batches.extend([results.get_counts(i) for i in range(len(transpiled_batch))])

    batch_merged_counts = batch_process_counts(counts_list=all_counts_from_all_batches,
                                               shots_per_circuit=shots,
                                               n_anc=n_anc,
                                               group_by=trajectory_report_protocol.get("group_by", "median") if trajectory_report_protocol.get("group", False) else "none"
                                               )
    # Analyze with wrap-around correction
    ml_bs, e_med, e_mean, e_std, e_min, e_max = analyse_counts_optimized(counts=batch_merged_counts,
                                                                          total_time=total_time,
                                                                          n_anc=n_anc,
                                                                          wrap_around_correction=True)
    error = abs(e_med - exact_eig)
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
        peak_MB          = 0.0,     # overwritten by @profile_memory_and_time
        runtime          = 0.0,     # overwritten by @profile_memory_and_time
        exact_eig        = exact_eig,
        max_theoretical_qdrift_error = 2 * sum(abs(H.coeffs)) ** 2 * total_time * np.exp(2 * sum(abs(H.coeffs)) * total_time),
        most_likely_bs   = ml_bs,
        est_energy_med   = e_med,
        est_energy_mean  = e_mean,
        est_energy_std   = e_std,
        estimation_error = error,
        alpha            = sum(abs(H.coeffs)),
        counts           = json.dumps(batch_merged_counts, sort_keys=True)
    )

# =========================================================================
# driving script – build the grid and launch a Pool
# =========================================================================
def main(verbose_export = False, parallel = False) -> None:
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
        REPORT_PROTOCOL_RESULTS_FROM_ANY_RANDOM_CIRCUIT,
        KET_0_AS_EIGENSTATE
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
                 trajectory_report_protocol = g[8],
                 exceptionally_stupid_eigenstate = g[9]
                 )
            for g in grid]
    
    # dump the full experiment grid to JSON (once)
    basename = f"qdrift_qpe_fc_parameter_sweep_{datetime.datetime.today():%Y-%m-%d}"
    metadata_path = pathlib.Path(f"{basename}.json")
    csv_path = pathlib.Path(f"{basename}.csv")

    # dump metadata JSON once (human-readable, reproducible)
    write_metadata(
        path=metadata_path,
        cfgs=cfgs,
        verbose=verbose_export,
        num_system_qubits=NUM_SYSTEM_QUBITS,
        replication_seeds=REPLICATION_SEEDS,
        times=list(TIMES),
        num_ancilla=NUM_ANCILLA,
        num_qdrift_segments=NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE,
        circuits_per_datapoint=RANDOM_CIRCUITS_PER_DATAPOINT,
        shots_per_circuit=SHOTS_PER_CIRCUIT,
        hamiltonians=HAMILTONIANS_TO_TEST,
        estimate_ground_state=ESTIMATE_GROUND_STATE,
        test_id=str(TEST_ID)
    )

    # ensure CSV header exists
    fieldnames = list(QPEResult.__dataclass_fields__.keys())
    init_csv(csv_path, fieldnames)

    if parallel:
        # run the sweep in parallel
        n_proc = min(os.cpu_count() or 1, 16)
        with Pool(processes=n_proc - 2) as pool:
            print(f"Running {len(cfgs)} configurations in parallel on {pool._processes} workers.")
            for result in pool.imap_unordered(run_simulation, cfgs):
                append_csv(csv_path, fieldnames, result)
    else:
        print(f"Running {len(cfgs)} configurations sequentially.")
        for cfg in cfgs:
            result = run_simulation(cfg)
            append_csv(csv_path, fieldnames, result)
# entry-point
if __name__ == "__main__":
    main()