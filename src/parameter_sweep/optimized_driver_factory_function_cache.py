"""
Optimized Stochastic QPE Parameter Sweep Driver

Key optimizations over original:
1. Module-level caching works correctly with joblib (was causing pickle errors)
2. Pre-computed eigenvalues cached alongside template circuits (avoids redundant np.linalg.eig calls)
3. Optimized AerSimulator configuration for your i7-10750H (2 threads per worker)
4. Reduced print() calls in hot path (I/O overhead in parallel)
5. Better batch size tuning for memory efficiency

USAGE: Do NOT run this file directly. Instead use:
    python run_sweep.py [--strategy BALANCED] [--sequential] [--no-progress]
"""
from __future__ import annotations
import time
import sys
import csv
import datetime
import itertools
import json
import os
import pathlib
from typing import Dict, List, NamedTuple
from dataclasses import dataclass, asdict
from uuid import uuid4
from functools import cache, cached_property
from enum import Enum

# Joblib for parallelization
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import Parameter

# Add project root to path
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.algorithms.optimized_qft_qpe_qdrift import (
    prepare_eigenstate_circuit, 
    make_pauli_gate_cache, 
    build_template_circuit, 
    build_qdrift_trajectory
)
from src.algorithms.chebyshev import chebyshev_nodes
from src.utils.generate_hamiltonians import calculate_minimum_evolution_time
from src.utils.memory_profiling_utils import profile_memory_and_time
from src.utils.qpe_postprocessing_utils import batch_process_counts, analyse_counts_optimized
from src.utils.utils_io import init_csv, append_csv
from src.utils.results_management import ResultsOrganizer


# ════════════════════════════════════════════════════════════════════════════
#  Result dataclass
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class QPEResult:
    ham: str
    exact_eig: float
    num_system_qubits: int
    num_ancilla: int
    alpha: float
    replication_seed: int
    n_circuits: int
    n_shots: int
    time: float
    segments: int
    most_likely_bs: str
    est_energy_med: float
    est_energy_mean: float
    est_energy_std: float
    estimation_error: float
    max_theoretical_qdrift_error: float
    peak_MB: float
    runtime: float
    counts: str


# ════════════════════════════════════════════════════════════════════════════
#  Parameter grid and constants
# ════════════════════════════════════════════════════════════════════════════
NUM_SYSTEM_QUBITS = 1  # Matches your "1 qubit test" Hamiltonian
PLACEHOLDER = "<PLACEHOLDER>_I_"

HAMILTONIANS_TO_TEST: dict[str, SparsePauliOp] = {
    "1 qubit test": SparsePauliOp.from_list([("X", 0.2), ("Z", 0.5), ("I", 0.3)], num_qubits=1)
}

NUM_ANCILLA = [7, 8, 9]
NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE = [1]
RANDOM_CIRCUITS_PER_DATAPOINT = [10, 100, 1000]
SHOTS_PER_CIRCUIT = [1, 100, 1000]
REPORT_PROTOCOL_RESULTS_FROM_ANY_RANDOM_CIRCUIT = [{"group": True, "group_by": "median"}]
REPLICATION_SEEDS = [42]
ESTIMATE_GROUND_STATE = [False]
KET_0_AS_EIGENSTATE = [False]
TEST_ID = uuid4()

# OPTIMIZATION: Batch size trades off transpile() overhead vs memory.
# 100 is conservative for 24GB RAM with 6 workers.
# Dynamically adjusted in run_simulation for very large circuit counts.
BATCH_SIZE = 100

# Compute time grid once at module load
_qpe_resolution_limits = calculate_minimum_evolution_time(
    hamiltonians=HAMILTONIANS_TO_TEST,
    m=min(NUM_ANCILLA)
)
_t_min_global = max(_qpe_resolution_limits.values())
_safety_margin = 0.85
_lower_bound = max(1e-10, _t_min_global * _safety_margin)
_upper_bound = _lower_bound * 400

TIMES = np.logspace(np.log10(_lower_bound), np.log10(_upper_bound), num=50)


# ════════════════════════════════════════════════════════════════════════════
#  Parallelization strategies (from your benchmarks)
# ════════════════════════════════════════════════════════════════════════════
class ParallelStrategy(Enum):
    """
    Parallelization strategies optimized for Intel i7-10750H (6 cores/12 threads).
    """
    MAXIMUM_SPEED = ("loky", 12, "Maximum speed (all logical cores)")
    BALANCED = ("loky", 6, "Balanced - physical cores only (recommended)")
    EFFICIENT = ("loky", 4, "Memory-efficient (fewer workers)")
    SEQUENTIAL = (None, 1, "Sequential execution (debugging)")

    def __init__(self, backend, n_jobs, description):
        self.backend = backend
        self.n_jobs = n_jobs
        self.description = description


DEFAULT_STRATEGY = ParallelStrategy.BALANCED


# ════════════════════════════════════════════════════════════════════════════
#  Cached objects - these work correctly when module is imported (not __main__)
# ════════════════════════════════════════════════════════════════════════════
class PauliGateCache(NamedTuple):
    """Immutable cache of Pauli evolution gates and sampling arrays."""
    gates: dict[str, PauliEvolutionGate]
    tau: Parameter
    pauli_labels: np.ndarray
    pmf: np.ndarray


class TemplateCircuitCache(NamedTuple):
    """
    OPTIMIZATION: Bundle template circuit with its exact eigenvalue.
    Avoids redundant np.linalg.eig() calls in run_simulation.
    """
    circuit: QuantumCircuit
    exact_eigenvalue: float


@cache
def get_pauli_cache(ham_key: str) -> PauliGateCache:
    """
    Cached Pauli evolution gates with pre-computed sampling arrays.
    Called once per Hamiltonian per worker process.
    """
    H = HAMILTONIANS_TO_TEST[ham_key]
    gates, tau = make_pauli_gate_cache(H, PLACEHOLDER)
    pauli_labels = np.array(list(H.paulis.to_labels()))
    pmf = np.abs(H.coeffs) / np.sum(np.abs(H.coeffs))
    return PauliGateCache(gates=gates, tau=tau, pauli_labels=pauli_labels, pmf=pmf)


@cache
def get_template_circuit(
    ham_key: str, 
    n_anc: int, 
    ground_state: bool, 
    ket_0_as_eigenstate: bool
) -> TemplateCircuitCache:
    """
    Cached template circuit AND exact eigenvalue.
    
    OPTIMIZATION: We compute eigenvalues here once, not in every run_simulation call.
    The eigendecomposition is O(n³) - expensive to repeat.
    """
    H = HAMILTONIANS_TO_TEST[ham_key]
    
    # Compute eigenvalues/eigenvectors once
    eigvals, eigvecs = np.linalg.eig(H.to_matrix())
    eigvals_real = eigvals.real
    
    eigenstate_index = np.argmin(eigvals_real) if ground_state else np.argmax(eigvals_real)
    exact_eigenvalue = float(eigvals_real[eigenstate_index])
    eigenstate = eigvecs[:, eigenstate_index]
    
    eigenstate_circuit = None if ket_0_as_eigenstate else prepare_eigenstate_circuit(eigenstate)
    
    # Get number of system qubits from Hamiltonian
    n_sys = H.num_qubits
    
    qc = build_template_circuit(
        n_anc=n_anc,
        n_sys=n_sys,
        placeholder_label=PLACEHOLDER,
        eigenvalue_circuit=eigenstate_circuit,
        exponentiated_hamiltonian_terms_cache=get_pauli_cache(ham_key)
    )
    
    return TemplateCircuitCache(circuit=qc, exact_eigenvalue=exact_eigenvalue)


# ════════════════════════════════════════════════════════════════════════════
#  Per-worker AerSimulator (lazy initialization)
# ════════════════════════════════════════════════════════════════════════════
class WorkerResources:
    """
    Lazy-initialized per-worker resources.
    
    OPTIMIZATION: Configure AerSimulator for your hardware:
    - max_parallel_threads=2: With 6 workers, uses all 12 logical cores
    - max_memory_mb=3000: ~3GB per worker, safe for 24GB total RAM
    """
    
    @cached_property
    def backend(self) -> AerSimulator:
        return AerSimulator(
            method="matrix_product_state",
            device="CPU",
            max_parallel_threads=2,  # 6 workers × 2 threads = 12 logical cores
            max_memory_mb=3000,      # Cap memory per worker
        )


# Each worker process gets its own _WORKER instance
_WORKER = WorkerResources()


# ════════════════════════════════════════════════════════════════════════════
#  Worker function - the actual simulation
# ════════════════════════════════════════════════════════════════════════════
@profile_memory_and_time
def run_simulation(cfg: dict) -> QPEResult:
    """
    Run one stochastic QPE experiment.
    
    This function is called in parallel by joblib workers.
    All heavy objects (Pauli cache, template circuit, AerSimulator) are
    lazily initialized and cached per-worker.
    """
    # Unpack config
    ham_key = cfg["ham"]
    n_anc = cfg["anc"]
    ground_state = cfg["ground_state"]
    n_circuits = cfg["circuits"]
    shots = cfg["shots"]
    total_time = cfg["time"]
    trajectory_report_protocol = cfg["trajectory_report_protocol"]
    ket_0_as_eigenstate = cfg["exceptionally_stupid_eigenstate"]

    # Get cached resources (computed once per unique key per worker)
    H = HAMILTONIANS_TO_TEST[ham_key]
    pauli_cache = get_pauli_cache(ham_key)
    template_cache = get_template_circuit(ham_key, n_anc, ground_state, ket_0_as_eigenstate)
    
    template_qc = template_cache.circuit
    exact_eig = template_cache.exact_eigenvalue

    # Seed hierarchy for reproducibility
    root_ss = np.random.SeedSequence(cfg["replication_seed"])
    child_ss = root_ss.spawn(n_circuits)
    
    # Dynamic batch size based on circuit count to prevent OOM
    # Large configs (10k circuits) need smaller batches
    if n_circuits >= 10000:
        batch_size = 50
    elif n_circuits >= 1000:
        batch_size = 75
    else:
        batch_size = BATCH_SIZE
    
    all_counts: List[dict] = []

    # Process in batches
    for batch_start in range(0, n_circuits, batch_size):
        batch_end = min(batch_start + batch_size, n_circuits)
        batch_ss = child_ss[batch_start:batch_end]
        
        # Build circuits for this batch
        batch_circuits = []
        for ss in batch_ss:
            rng = np.random.default_rng(ss)
            qc = build_qdrift_trajectory(
                n_anc=n_anc,
                h_signature=ham_key,
                total_time=total_time,
                H=H,
                rng=rng,
                n_qdrift_segments=1,
                placeholder_label=PLACEHOLDER,
                template_circuit=template_qc,
                exponentialed_hamiltonian_terms_cache=pauli_cache
            )
            batch_circuits.append(qc)

        # Transpile batch (vectorized is faster than one-by-one)
        transpiled = transpile(batch_circuits, backend=_WORKER.backend)

        # Execute batch
        seeds = [ss.generate_state(1)[0] for ss in batch_ss]
        job = _WORKER.backend.run(transpiled, shots=shots, seed_simulation=seeds)
        result = job.result()
        
        all_counts.extend([result.get_counts(i) for i in range(len(transpiled))])

    # Post-process counts
    group_by = trajectory_report_protocol.get("group_by", "median") if trajectory_report_protocol.get("group", False) else "none"
    merged_counts = batch_process_counts(
        counts_list=all_counts,
        shots_per_circuit=shots,
        n_anc=n_anc,
        group_by=group_by
    )

    # Analyze results
    ml_bs, e_med, e_mean, e_std, e_min, e_max = analyse_counts_optimized(
        counts=merged_counts,
        total_time=total_time,
        n_anc=n_anc,
        wrap_around_correction=True
    )

    # Compute theoretical error bound
    alpha = float(np.sum(np.abs(H.coeffs)))
    max_error = 2 * alpha**2 * total_time * np.exp(2 * alpha * total_time)

    return QPEResult(
        ham=ham_key,
        num_system_qubits=H.num_qubits,
        num_ancilla=n_anc,
        time=total_time,
        segments=cfg["segments"],
        replication_seed=cfg["replication_seed"],
        n_circuits=n_circuits,
        n_shots=shots,
        peak_MB=0.0,      # Filled by @profile_memory_and_time
        runtime=0.0,      # Filled by @profile_memory_and_time
        exact_eig=exact_eig,
        max_theoretical_qdrift_error=max_error,
        most_likely_bs=ml_bs,
        est_energy_med=e_med,
        est_energy_mean=e_mean,
        est_energy_std=e_std,
        estimation_error=abs(e_med - exact_eig),
        alpha=alpha,
        counts=json.dumps(merged_counts, sort_keys=True)
    )

# ════════════════════════════════════════════════════════════════════════════
#  Checkpointing helpers
# ════════════════════════════════════════════════════════════════════════════
def find_resumable_runs(
    organizer: 'ResultsOrganizer',
    experiment_type: str,
    metadata: dict,
    num_time_values: int,
    hours: int = 24
) -> List[pathlib.Path]:
    """
    Find CSV files from previous runs with matching metadata started within the last N hours.
    
    Uses ResultsOrganizer's directory structure:
    - Chebyshev (<10 time values): results/parameter_sweeps/chebyshev_interpolation/YYYY-MM/
    - Full sweep (>=10 time values): results/parameter_sweeps/full_sweeps/YYYY-MM/
    
    Matches files with same metadata prefix (ignoring timestamp).
    """
    from datetime import datetime, timedelta
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    # Determine which directory to search based on sweep type
    if num_time_values < 10:
        search_dir = organizer.dirs['chebyshev_interpolation']
    else:
        search_dir = organizer.dirs['full_sweeps']
    
    # Build the expected filename prefix using same logic as generate_filename
    # Pattern: {experiment_type}_{key}_{value}_{key}_{value}_..._{timestamp}.csv
    prefix_parts = [experiment_type]
    for key, value in sorted(metadata.items()):
        clean_value = str(value).replace(' ', '_').replace('/', '-')
        prefix_parts.append(f"{key}_{clean_value}")
    
    prefix = '_'.join(prefix_parts)
    
    resumable = []
    
    if search_dir.exists():
        # Search in all month subdirectories
        for csv_file in search_dir.rglob("*.csv"):
            # Check if filename starts with our prefix (excluding timestamp)
            if csv_file.name.startswith(prefix):
                # Check file modification time
                mtime = datetime.fromtimestamp(csv_file.stat().st_mtime)
                if mtime > cutoff_time:
                    # Verify it has some content (not just header)
                    try:
                        with open(csv_file, 'r') as f:
                            lines = sum(1 for _ in f)
                        if lines > 1:  # Has data beyond header
                            resumable.append(csv_file)
                    except:
                        pass
    
    # Sort by modification time (most recent first)
    resumable.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return resumable


def prompt_for_resume(resumable_runs: List[pathlib.Path]) -> Optional[pathlib.Path]:
    """
    Ask user if they want to resume a previous run.
    Returns the selected path or None for a new run.
    """
    from datetime import datetime
    
    print("\n" + "=" * 80)
    print("RESUMABLE RUNS FOUND")
    print("=" * 80)
    print()
    
    for i, csv_path in enumerate(resumable_runs[:5], 1):  # Show max 5
        mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
        size_kb = csv_path.stat().st_size / 1024
        
        # Count completed configs
        try:
            with open(csv_path, 'r') as f:
                completed = sum(1 for _ in f) - 1  # Subtract header
        except:
            completed = "?"
        
        time_ago = datetime.now() - mtime
        hours_ago = time_ago.total_seconds() / 3600
        
        # Extract timestamp from filename for display
        # Pattern: ...._YYYY-MM-DD_HH-MM-SS.csv
        filename = csv_path.stem  # Remove .csv
        
        print(f"  [{i}] {csv_path.name}")
        print(f"      Last modified: {hours_ago:.1f} hours ago ({mtime.strftime('%Y-%m-%d %H:%M')})")
        print(f"      Completed: {completed} configs, Size: {size_kb:.1f} KB")
        print()
    
    print(f"  [N] Start NEW run (ignore previous)")
    print()
    
    while True:
        choice = input("Enter choice [1/2/.../N]: ").strip().upper()
        
        if choice == 'N':
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(resumable_runs[:5]):
                selected = resumable_runs[idx]
                print(f"\n  → Resuming: {selected.name}")
                return selected
        except ValueError:
            pass
        
        print("  Invalid choice. Enter a number or 'N'.")


def get_completed_configs(csv_path: pathlib.Path) -> set:
    """Read existing CSV and return set of completed configuration tuples."""
    if not csv_path.exists():
        return set()

    completed = set()
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                config_tuple = (
                    row['ham'],
                    int(row['num_ancilla']),
                    float(row['time']),
                    int(row['segments']),
                    int(row['replication_seed']),
                    int(row['n_circuits']),
                    int(row['n_shots'])
                )
                completed.add(config_tuple)
        print(f"Found {len(completed)} completed configurations")
    except Exception as e:
        print(f"Warning: Could not read CSV: {e}")
        return set()

    return completed


def config_to_tuple(cfg: dict) -> tuple:
    """Convert config dict to tuple for comparison."""
    return (
        cfg['ham'],
        cfg['anc'],
        cfg['time'],
        cfg['segments'],
        cfg['replication_seed'],
        cfg['circuits'],
        cfg['shots']
    )


# ════════════════════════════════════════════════════════════════════════════
#  Main driver
# ════════════════════════════════════════════════════════════════════════════
def main(
    verbose_export: bool = False,
    parallel: bool = True,
    strategy: ParallelStrategy = DEFAULT_STRATEGY,
    show_progress: bool = True,
    auto_resume: bool = True
) -> None:
    """
    Main driver function for parameter sweep.
    
    Args:
        verbose_export: Include all configs in metadata JSON
        parallel: Enable parallel execution
        strategy: Parallelization strategy
        show_progress: Show progress bar (sequential mode only)
        auto_resume: Check for and offer to resume previous runs
    """
    print("=" * 80)
    print("STOCHASTIC QPE PARAMETER SWEEP")
    print("=" * 80)

    # Build parameter grid
    grid = itertools.product(
        HAMILTONIANS_TO_TEST.keys(),
        NUM_ANCILLA,
        TIMES,
        NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE,
        REPLICATION_SEEDS,
        RANDOM_CIRCUITS_PER_DATAPOINT,
        SHOTS_PER_CIRCUIT,
        ESTIMATE_GROUND_STATE,
        REPORT_PROTOCOL_RESULTS_FROM_ANY_RANDOM_CIRCUIT,
        KET_0_AS_EIGENSTATE
    )

    cfgs = [
        dict(
            ham=g[0],
            anc=g[1],
            time=float(g[2]),
            segments=g[3],
            replication_seed=g[4],
            circuits=g[5],
            shots=g[6],
            ground_state=g[7],
            trajectory_report_protocol=g[8],
            exceptionally_stupid_eigenstate=g[9]
        )
        for g in grid
    ]

    # Setup output paths
    organizer = ResultsOrganizer(pathlib.Path(__file__).parent.parent.parent / "results")
    metadata = {
        'hamiltonians': '_'.join([h.replace(' ', '') for h in HAMILTONIANS_TO_TEST.keys()]),
        'ancilla': f"{min(NUM_ANCILLA)}to{max(NUM_ANCILLA)}" if len(NUM_ANCILLA) > 1 else str(NUM_ANCILLA[0]),
        'nodes': len(TIMES),
        'type': 'chebyshev' if len(TIMES) < 10 else 'full_sweep'
    }
    
    # Check for resumable runs from the past 24 hours
    csv_path = None
    
    if auto_resume:
        resumable_runs = find_resumable_runs(
            organizer=organizer,
            experiment_type='parameter_sweep',
            metadata=metadata,
            num_time_values=len(TIMES),
            hours=24
        )
        
        if resumable_runs:
            # Ask user if they want to resume
            selected_path = prompt_for_resume(resumable_runs)
            if selected_path:
                csv_path = selected_path
    
    # Create new path if not resuming
    if csv_path is None:
        csv_path = organizer.get_save_path(
            experiment_type='parameter_sweep',
            metadata=metadata,
            organize_by_month=True,
            num_time_values=len(TIMES)
        )
    
    metadata_path = csv_path.with_suffix('.json')

    # Resume from checkpoint
    completed = get_completed_configs(csv_path)
    cfgs = [cfg for cfg in cfgs if config_to_tuple(cfg) not in completed]

    if not cfgs:
        print("All configurations already completed!")
        return

    # Write metadata
    metadata_content = {
        'num_system_qubits': NUM_SYSTEM_QUBITS,
        'replication_seeds': REPLICATION_SEEDS,
        'times': list(TIMES),
        'num_ancilla': NUM_ANCILLA,
        'num_qdrift_segments': NUM_QDRIFT_SEGMENTS_PER_CHANNEL_SAMPLE,
        'circuits_per_datapoint': RANDOM_CIRCUITS_PER_DATAPOINT,
        'shots_per_circuit': SHOTS_PER_CIRCUIT,
        'hamiltonians': {k: str(v) for k, v in HAMILTONIANS_TO_TEST.items()},
        'estimate_ground_state': ESTIMATE_GROUND_STATE,
        'test_id': str(TEST_ID),
        'num_configs': len(cfgs),
        'batch_size': BATCH_SIZE,
        'parallel_strategy': strategy.name,
        'generated_timestamp': datetime.datetime.now().isoformat()
    }

    if verbose_export:
        metadata_content['all_configs'] = cfgs

    with open(metadata_path, 'w') as f:
        json.dump(metadata_content, f, indent=2, default=str)

    # Initialize CSV
    fieldnames = list(QPEResult.__dataclass_fields__.keys())
    init_csv(csv_path, fieldnames)

    print(f"\nConfiguration:")
    print(f"  Remaining configs: {len(cfgs)}")
    print(f"  Parallel: {parallel}")
    if parallel:
        print(f"  Strategy: {strategy.name} ({strategy.n_jobs} workers)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Output: {csv_path}")
    print("-" * 80)

    start_time = time.time()

    if parallel and strategy != ParallelStrategy.SEQUENTIAL:
        print(f"\n🚀 Starting parallel execution with joblib...")
        
        from joblib import Parallel, delayed
        
        completed_count = 0
        failed_count = 0
        failed_configs = []
        
        def safe_run_simulation(cfg):
            """Wrapper that catches exceptions and returns them instead of raising."""
            try:
                return ("success", run_simulation(cfg))
            except Exception as e:
                import traceback
                return ("failed", cfg, type(e).__name__, str(e)[:500])
        
        # Process in chunks to get incremental CSV writes
        # Joblib's generator mode doesn't actually stream results well
        CHUNK_SIZE = 50  # Write to CSV every 50 tasks
        
        for chunk_start in range(0, len(cfgs), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(cfgs))
            chunk_cfgs = cfgs[chunk_start:chunk_end]
            
            print(f"\n  Processing chunk {chunk_start//CHUNK_SIZE + 1}: "
                  f"tasks {chunk_start+1}-{chunk_end} of {len(cfgs)}")
            
            # Process this chunk
            results = Parallel(
                n_jobs=strategy.n_jobs,
                backend='loky',
                verbose=5,
                batch_size='auto',
                pre_dispatch='2*n_jobs',
            )(delayed(safe_run_simulation)(cfg) for cfg in chunk_cfgs)
            
            # Write results from this chunk immediately
            chunk_completed = 0
            chunk_failed = 0
            for result_tuple in results:
                if result_tuple[0] == "success":
                    result = result_tuple[1]
                    append_csv(csv_path, fieldnames, result)
                    completed_count += 1
                    chunk_completed += 1
                else:
                    # Failed task
                    _, cfg, error_type, error_msg = result_tuple
                    failed_count += 1
                    chunk_failed += 1
                    failed_configs.append({"config": cfg, "error": error_type, "message": error_msg})
                    print(f"    ✗ FAILED ({error_type}): circuits={cfg['circuits']}, "
                          f"shots={cfg['shots']}")
            
            print(f"  ✓ Chunk complete: {chunk_completed} succeeded, {chunk_failed} failed")
            print(f"  📊 Total progress: {completed_count}/{len(cfgs)} "
                  f"({100*completed_count/len(cfgs):.1f}%)")
        
        # Summary of failures
        if failed_configs:
            print(f"\n⚠ {failed_count} tasks failed. Saving failed configs...")
            failed_path = csv_path.with_suffix('.failed.json')
            with open(failed_path, 'w') as f:
                json.dump(failed_configs, f, indent=2, default=str)
            print(f"  Failed configs saved to: {failed_path}")
    else:
        print("\n📝 Sequential mode...")
        iterator = tqdm(cfgs, desc="Processing") if show_progress else cfgs
        for cfg in iterator:
            result = run_simulation(cfg)
            append_csv(csv_path, fieldnames, result)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"  Configs: {len(cfgs)}")
    print(f"  Time: {elapsed / 60:.2f} min")
    print(f"  Avg: {elapsed / len(cfgs):.2f} s/config")
    print(f"  Output: {csv_path}")


# ════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ════════════════════════════════════════════════════════════════════════════
def run_cli():
    """CLI entry point - call this from run_sweep.py"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stochastic QPE Parameter Sweep"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.name for s in ParallelStrategy],
        default=DEFAULT_STRATEGY.name,
        help=f"Parallelization strategy (default: {DEFAULT_STRATEGY.name})"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose metadata"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Skip resume prompt, always start fresh"
    )

    args = parser.parse_args()

    strategy = ParallelStrategy.SEQUENTIAL if args.sequential else ParallelStrategy[args.strategy]

    main(
        verbose_export=args.verbose,
        parallel=(strategy != ParallelStrategy.SEQUENTIAL),
        strategy=strategy,
        show_progress=not args.no_progress,
        auto_resume=not args.no_resume
    )


# ════════════════════════════════════════════════════════════════════════════
#  DO NOT RUN DIRECTLY - This guard prevents the pickle error but reminds you
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 80)
    print("ERROR: Do not run this file directly!")
    print("=" * 80)
    print()
    print("Running this file directly causes pickle errors with joblib.")
    print()
    print("Instead, use the runner script:")
    print()
    print("    python run_sweep.py [options]")
    print()
    print("Options:")
    print("    --strategy BALANCED|MAXIMUM_SPEED|EFFICIENT|SEQUENTIAL")
    print("    --sequential    Run without parallelization")
    print("    --no-progress   Disable progress bar")
    print("    --verbose       Include all configs in metadata JSON")
    print()
    sys.exit(1)
