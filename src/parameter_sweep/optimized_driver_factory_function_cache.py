
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
import warnings


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
from src.utils.qpe_postprocessing_utils import analyse_counts, int_to_bitstring

# Create the SparsePauliOp
H_H2 = create_h2_minimal_basis_hamiltonian()

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
    top10_py_allocs : str  # report from tracemalloc
    counts          : str  # json-encoded

# ════════════════════════════════════════════════════════════════════════════
#  Parameter grid and constants
# ════════════════════════════════════════════════════════════════════════════
NUM_SYSTEM_QUBITS  = 4
ISING_J, ISING_G = 1.2, 1
PLACEHOLDER = "_I_"

HAMILTONIANS_TO_TEST: dict[str, SparsePauliOp] = {
    # "exact_ham_exact_qdritf" : SparsePauliOp(data="ZZZZ", coeffs=np.pi / 4),
    "H2_minimal_basis": H_H2,
    "H_ising" : generate_ising_hamiltonian(num_qubits=NUM_SYSTEM_QUBITS, J=ISING_J * 0.5, g=ISING_G * 0.5) 
}

NUM_ANCILLA  = [15]  # number of ancilla qubits

# chebyshev_nodes = np.array(chebyshev_nodes(10))
# scaled_nodes_pos = 0.000001 + (0.1 - 0.000001) * chebyshev_nodes[:5]

qpe_resolution_limits = calculate_minimum_evolution_time(hamiltonians=HAMILTONIANS_TO_TEST, m=min(NUM_ANCILLA))
print(qpe_resolution_limits)
t_min_global = max(qpe_resolution_limits.values())
lower_bound = max(1e-10, t_min_global * 0.9)  # Don't go below 90% of t_min
upper_bound = min(1e1, t_min_global * 1000)    # Don't exceed 100× t_min
TIMES = np.logspace(np.log2(lower_bound), np.log2(upper_bound), base=2, num=20)
NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE  = [1]
RANDOM_CIRCUITS_PER_DATAPOINT = [100]
SHOTS_PER_CIRCUIT = [1, 100]
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
            # print("DEBUG: sample values:", expanded_arrrr[:10])
            expanded_arrrr = np.array([int(bs, 2) for bs in expanded_arrrr]) # it's easier to calculate median from a list of ints than from a list of binary bitstrings

            if trajectory_report_protocol["group_by"] == "median":
                # just report the median measured bitstring 
                # print("DEBUG: sample values:", expanded_arrrr[:10])
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
        counts           = json.dumps(total_counts, sort_keys=True),
        top10_py_allocs  = "",      # overwritten by @profile_memory_and_time
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
    with Pool(processes=n_proc//2) as pool:
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