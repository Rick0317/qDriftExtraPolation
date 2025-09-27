#!/usr/bin/env python3
"""
Optimized QPE Simulation with qDrift for AWS/Container Deployment
Features:
- Container-aware resource detection
- Shared memory caching for multiprocessing
- Simple CSV output with progress tracking
- Proper thread configuration
- Fault tolerance and graceful interruption
"""

from __future__ import annotations
import os
import sys
import pathlib
import datetime
import itertools
import json
import csv
import threading
import signal
import time
import pickle
import fcntl
from typing import Dict, List, Optional, Tuple, Callable, Sequence, NamedTuple
from dataclasses import dataclass, fields, asdict
from uuid import uuid4
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Third-party imports
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import Parameter

# Add parent directory to path for imports
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

# Import your existing modules
from src.algorithms.optimized_qft_qpe_qdrift import (
    prepare_eigenstate_circuit, make_pauli_gate_cache, 
    build_template_circuit, build_qdrift_trajectory
)
from src.algorithms.chebyshev import chebyshev_nodes
from src.utils.generate_hamiltonians import (
    calculate_minimum_evolution_time, create_h2_minimal_basis_hamiltonian, 
    generate_ising_hamiltonian
)
from src.utils.qpe_postprocessing_utils import (
    batch_process_counts, int_to_bitstring, analyse_counts_optimized
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Data Classes
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class QPEResult:
    """Results from a single QPE simulation run."""
    ham: str
    exact_eig: float
    num_system_qubits: int
    num_ancilla: int
    alpha: float  # sum of Hamiltonian coefficients
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
    peak_MB: float  # peak memory usage in MB
    runtime: float  # runtime in seconds
    counts: str  # json-encoded

class PauliGateCache(NamedTuple):
    """Cached Pauli gates and sampling data."""
    gates: dict[str, PauliEvolutionGate]
    tau: Parameter
    pauli_labels: np.ndarray
    pmf: np.ndarray

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Container/AWS Resource Detection
# ╚═══════════════════════════════════════════════════════════════════════════╝

def get_container_limits():
    """
    Detect container resource limits (CPU and memory).
    Works with Docker, K8s, ECS, and other container runtimes.
    """
    cpu_limit = None
    memory_limit = None
    
    # Try to read cgroup v2 limits
    try:
        with open('/sys/fs/cgroup/cpu.max', 'r') as f:
            cpu_max = f.read().strip().split()
            if cpu_max[0] != 'max':
                cpu_limit = int(cpu_max[0]) / int(cpu_max[1])
    except:
        # Try cgroup v1
        try:
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
                quota = int(f.read().strip())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
                period = int(f.read().strip())
            if quota > 0:
                cpu_limit = quota / period
        except:
            pass
    
    # Memory limits
    try:
        with open('/sys/fs/cgroup/memory.max', 'r') as f:
            mem = f.read().strip()
            if mem != 'max':
                memory_limit = int(mem)
    except:
        try:
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                memory_limit = int(f.read().strip())
        except:
            pass
    
    # Fallback to environment variables
    if cpu_limit is None:
        cpu_limit = float(os.environ.get('CPU_LIMIT', os.cpu_count()))
    if memory_limit is None:
        memory_limit = int(os.environ.get('MEMORY_LIMIT', 
                                         os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')))
    
    return int(cpu_limit), memory_limit

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Shared Memory Cache for Multiprocessing
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SharedMemoryCache:
    """
    Cache that uses multiprocessing.Manager for inter-process communication.
    Solves the cache duplication problem with multiprocessing.
    """
    
    def __init__(self):
        self.manager = mp.Manager()
        self.cache_dict = self.manager.dict()
        self.lock = self.manager.Lock()
        
    def get(self, key):
        """Thread-safe get from cache."""
        return self.cache_dict.get(key)
    
    def put(self, key, value):
        """Thread-safe put to cache."""
        with self.lock:
            if key not in self.cache_dict:
                self.cache_dict[key] = pickle.dumps(value)
    
    def get_or_compute(self, key, compute_func, *args, **kwargs):
        """Get from cache or compute and store."""
        cached = self.get(key)
        if cached is not None:
            return pickle.loads(cached)
        
        value = compute_func(*args, **kwargs)
        self.put(key, value)
        return value

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Circuit Factory with Shared Caching
# ╚═══════════════════════════════════════════════════════════════════════════╝

class CircuitFactory:
    """
    Factory for building circuits with efficient shared caching.
    Pre-computes expensive operations once and shares across processes.
    """
    
    def __init__(self, hamiltonians: Dict[str, SparsePauliOp], 
                 placeholder: str = "<PLACEHOLDER>_I_"):
        self.hamiltonians = hamiltonians
        self.placeholder = placeholder
        self.cache = SharedMemoryCache()
        
        # Pre-compute all Pauli gates for all Hamiltonians
        print("Pre-computing Pauli gate caches...")
        self._precompute_pauli_caches()
        print("Cache initialization complete!")
    
    def _precompute_pauli_caches(self):
        """Pre-compute all Pauli gate caches once in main process."""
        for ham_key, H in self.hamiltonians.items():
            cache_key = f"pauli_cache_{ham_key}"
            if self.cache.get(cache_key) is None:
                gates, tau = make_pauli_gate_cache(H, self.placeholder)
                pauli_labels = np.array(list(H.paulis.to_labels()))
                pmf = np.abs(H.coeffs) / np.sum(np.abs(H.coeffs))
                
                cache_data = PauliGateCache(
                    gates=gates,
                    tau=tau,
                    pauli_labels=pauli_labels,
                    pmf=pmf
                )
                self.cache.put(cache_key, cache_data)
    
    def get_pauli_cache(self, ham_key: str) -> PauliGateCache:
        """Get Pauli cache from shared memory."""
        cache_key = f"pauli_cache_{ham_key}"
        return pickle.loads(self.cache.get(cache_key))
    
    def get_template_circuit(self, ham_key: str, n_anc: int, 
                            ground_state: bool, n_sys: int) -> QuantumCircuit:
        """Get or build template circuit with caching."""
        cache_key = f"template_{ham_key}_{n_anc}_{ground_state}"
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            return pickle.loads(cached)
        
        # Build template circuit if not cached
        H = self.hamiltonians[ham_key]
        eigvals, eigvecs = np.linalg.eig(H.to_matrix())
        eigenstate_index = np.argmin(eigvals.real) if ground_state else np.argmax(eigvals.real)
        eigenstate = eigvecs[:, eigenstate_index]
        eigenstate_circuit = prepare_eigenstate_circuit(eigenstate)
        
        pauli_cache = self.get_pauli_cache(ham_key)
        qc = build_template_circuit(
            n_anc=n_anc,
            n_sys=n_sys,
            placeholder_label=self.placeholder,
            eigenvalue_circuit=eigenstate_circuit,
            exponentiated_hamiltonian_terms_cache=pauli_cache
        )
        
        self.cache.put(cache_key, qc)
        return qc

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Thread-Safe CSV Writer
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ThreadSafeCSVWriter:
    """
    Simple, bulletproof CSV writer for concurrent writes from multiple processes.
    Perfect for long-running simulations where I/O is not the bottleneck.
    """
    
    def __init__(self, filepath: pathlib.Path, dataclass_type: type):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.dataclass_type = dataclass_type
        
        # Get field names from dataclass
        self.fieldnames = [f.name for f in fields(dataclass_type)]
        
        # Initialize CSV with headers
        self._init_csv()
    
    def _init_csv(self):
        """Create CSV file with headers if it doesn't exist."""
        if not self.filepath.exists():
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            print(f"📝 Created CSV file: {self.filepath}")
    
    def write_result(self, result):
        """Write a single result to CSV. Thread-safe and process-safe."""
        row_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result
        
        with self.lock:
            with open(self.filepath, 'a', newline='') as f:
                # Use file locking for process-safety on Unix/Linux
               try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writerow(row_dict)
                    f.flush()
                    os.fsync(f.fileno())
               except:
                   writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                   writer.writerow(row_dict)
                   f.flush()
                   os.fsync(f.fileno())
               finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
               

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Progress Tracking
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SimulationProgressTracker:
    """Track progress of long-running simulations with ETA calculation."""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.task_times = []
    
    def task_completed(self, task_time: float):
        """Record successful task completion."""
        with self.lock:
            self.completed_tasks += 1
            self.task_times.append(task_time)
            self._print_progress(task_time)
    
    def task_failed(self, error_msg: str):
        """Record task failure."""
        with self.lock:
            self.failed_tasks += 1
            print(f"❌ Task failed: {error_msg}")
            self._print_progress()
    
    def _print_progress(self, last_task_time: Optional[float] = None):
        """Print progress with ETA."""
        progress = (self.completed_tasks + self.failed_tasks) / self.total_tasks * 100
        elapsed = time.time() - self.start_time
        
        # Calculate ETA
        if self.task_times:
            avg_time = sum(self.task_times) / len(self.task_times)
            remaining = self.total_tasks - self.completed_tasks - self.failed_tasks
            eta_seconds = avg_time * remaining
            eta_str = f"{eta_seconds/60:.1f} min"
        else:
            eta_str = "calculating..."
        
        status = (
            f"📊 Progress: {progress:.1f}% "
            f"({self.completed_tasks}/{self.total_tasks} completed, "
            f"{self.failed_tasks} failed) "
            f"| Elapsed: {elapsed/60:.1f} min "
            f"| ETA: {eta_str}"
        )
        
        if last_task_time:
            status += f" | Last: {last_task_time:.1f}s"
        
        print(status)

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Optimized Simulator Pool
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SimulatorPool:
    """Manages simulators with optimized thread configuration."""
    
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        cpu_limit, _ = get_container_limits()
        
        # Calculate optimal threads per worker
        available_cpus = max(1, cpu_limit - 1)  # Leave 1 CPU for system
        self.threads_per_worker = max(1, available_cpus // n_workers)
        
        # Set environment variables for linear algebra libraries
        os.environ['OMP_NUM_THREADS'] = str(self.threads_per_worker)
        os.environ['MKL_NUM_THREADS'] = str(self.threads_per_worker)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.threads_per_worker)
        os.environ['AER_MAX_THREADS'] = str(self.threads_per_worker)
        
        print(f"⚙️  Configured {n_workers} workers with {self.threads_per_worker} threads each")
    
    def create_simulator(self):
        """Create a properly configured simulator instance."""
        return AerSimulator(
            method="matrix_product_state",
            device="CPU",
            max_parallel_threads=self.threads_per_worker,
            max_parallel_experiments=1,
            statevector_parallel_threshold=14
        )

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Worker Functions
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Global variables for worker processes
_worker_factory = None
_worker_simulator = None
_worker_csv_writer = None
_worker_progress = None

def worker_init(circuit_factory, simulator, csv_writer, progress):
    """Initialize worker process with necessary resources."""
    global _worker_factory, _worker_simulator, _worker_csv_writer, _worker_progress
    _worker_factory = circuit_factory
    _worker_simulator = simulator
    _worker_csv_writer = csv_writer
    _worker_progress = progress
    
    # Set thread configuration for this worker
    if hasattr(simulator, 'threads_per_worker'):
        os.environ['OMP_NUM_THREADS'] = str(simulator.threads_per_worker)
        os.environ['MKL_NUM_THREADS'] = str(simulator.threads_per_worker)

def run_simulation_worker(config: dict) -> Optional[QPEResult]:
    """
    Run a single simulation in a worker process.
    This is the main computational function.
    """
    task_start = time.time()
    
    try:
        # Extract configuration
        ham_key = config["ham"]
        n_anc = config["anc"]
        n_sys = config["n_sys"]
        ground_state = config["ground_state"]
        n_circuits = config["circuits"]
        shots = config["shots"]
        total_time = config["time"]
        segments = config["segments"]
        
        # Get Hamiltonian and exact eigenvalue
        H = _worker_factory.hamiltonians[ham_key]
        eigvals = np.linalg.eigvals(H.to_matrix()).real
        exact_eig = float(np.min(eigvals) if ground_state else np.max(eigvals))
        
        # Get cached resources
        pauli_cache = _worker_factory.get_pauli_cache(ham_key)
        template_circuit = _worker_factory.get_template_circuit(
            ham_key, n_anc, ground_state, n_sys
        )
        
        # Process circuits in batches
        batch_size = min(100, n_circuits)
        all_counts = []
        
        # Random seed hierarchy
        root_ss = np.random.SeedSequence(config["replication_seed"])
        child_ss = root_ss.spawn(n_circuits)
        
        for batch_start in range(0, n_circuits, batch_size):
            batch_end = min(batch_start + batch_size, n_circuits)
            batch_ss = child_ss[batch_start:batch_end]
            batch_circuits = []
            
            # Build batch of circuits
            for ss in batch_ss:
                rng = np.random.default_rng(ss)
                qc = build_qdrift_trajectory(
                    n_anc=n_anc,
                    h_signature=ham_key,
                    total_time=total_time,
                    H=H,
                    rng=rng,
                    n_qdrift_segments=segments,
                    placeholder_label=_worker_factory.placeholder,
                    template_circuit=template_circuit,
                    exponentialed_hamiltonian_terms_cache=pauli_cache
                )
                batch_circuits.append(qc)
            
            # Transpile batch
            transpiled = transpile(
                batch_circuits,
                backend=_worker_simulator,
                optimization_level=0,
                num_processes=1, 
                approximation_degree=0 
            )
            
            # Execute batch
            seeds_for_batch = [ss.generate_state(1)[0] for ss in batch_ss]
            job = _worker_simulator.run(
                transpiled,
                shots=shots,
                memory=False,
                seed_simulator=seeds_for_batch
            )
            results = job.result()
            
            # Collect counts
            for i in range(len(transpiled)):
                all_counts.append(results.get_counts(i))
        
        # Process results
        merged_counts = batch_process_counts(
            counts_list=all_counts,
            shots_per_circuit=shots,
            n_anc=n_anc,
            group_by=config.get("group_by", "median")
        )
        
        # Analyze with wrap-around correction
        ml_bs, e_med, e_mean, e_std, _, _ = analyse_counts_optimized(
            counts=merged_counts,
            total_time=total_time,
            n_anc=n_anc,
            wrap_around_correction=True
        )
        
        error = abs(e_med - exact_eig)
        
        # Calculate runtime and memory (simplified)
        task_time = time.time() - task_start
        
        # Create result
        result = QPEResult(
            ham=ham_key,
            exact_eig=exact_eig,
            num_system_qubits=n_sys,
            num_ancilla=n_anc,
            alpha=sum(abs(H.coeffs)),
            replication_seed=config["replication_seed"],
            n_circuits=n_circuits,
            n_shots=shots,
            time=total_time,
            segments=segments,
            most_likely_bs=ml_bs,
            est_energy_med=e_med,
            est_energy_mean=e_mean,
            est_energy_std=e_std,
            estimation_error=error,
            max_theoretical_qdrift_error=2 * sum(abs(H.coeffs)) ** 2 * total_time * 
                                          np.exp(2 * sum(abs(H.coeffs)) * total_time),
            peak_MB=0.0,  # Could add memory profiling
            runtime=task_time,
            counts=json.dumps(merged_counts, sort_keys=True)
        )
        
        # Write result immediately
        _worker_csv_writer.write_result(result)
        _worker_progress.task_completed(task_time)
        
        return result
        
    except Exception as e:
        _worker_progress.task_failed(str(e))
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Main Execution Function
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main(verbose_export: bool = False, parallel: bool = True):
    """
    Main execution function with AWS-optimized parallel processing.
    """
    
    # ─── Configuration ─────────────────────────────────────────────────────
    
    # Simulation parameters
    NUM_SYSTEM_QUBITS = 1
    PLACEHOLDER = "<PLACEHOLDER>_I_"
    
    # Define Hamiltonians to test
    HAMILTONIANS_TO_TEST = {
        "1 qubit test": SparsePauliOp.from_list([("X", 0.2), ("Z", 0.5), ("I", 0.3)], num_qubits=1),
        # Add more Hamiltonians here
    }
    
    # Parameter grid
    NUM_ANCILLA = [13, 15]
    NUM_QDRIFT_SEGMENTS = [1]
    RANDOM_CIRCUITS_PER_DATAPOINT = [10, 100, 1000, 10000]
    SHOTS_PER_CIRCUIT = [1, 10, 100]
    REPLICATION_SEEDS = [42]
    ESTIMATE_GROUND_STATE = [False]
    
    # Calculate time range
    qpe_resolution_limits = calculate_minimum_evolution_time(
        hamiltonians=HAMILTONIANS_TO_TEST, 
        m=min(NUM_ANCILLA)
    )
    print(f"QPE resolution limits: {qpe_resolution_limits}")
    
    t_min_global = max(qpe_resolution_limits.values())
    lower_bound = max(1e-10, t_min_global * 0.8)
    upper_bound = min(1e1, t_min_global * 100)
    TIMES = np.logspace(np.log2(lower_bound), np.log2(upper_bound), base=2, num=12)
    
    TEST_ID = uuid4()
    
    # ─── Container Resource Detection ──────────────────────────────────────
    
    cpu_limit, memory_limit = get_container_limits()
    print(f"🐳 Container limits - CPUs: {cpu_limit}, Memory: {memory_limit / (1024**3):.2f} GB")
    
    # ─── Build Parameter Grid ──────────────────────────────────────────────
    
    grid = itertools.product(
        HAMILTONIANS_TO_TEST.keys(),
        NUM_ANCILLA,
        TIMES,
        NUM_QDRIFT_SEGMENTS,
        REPLICATION_SEEDS,
        RANDOM_CIRCUITS_PER_DATAPOINT,
        SHOTS_PER_CIRCUIT,
        ESTIMATE_GROUND_STATE
    )
    
    configs = [
        {
            "ham": g[0],
            "anc": g[1],
            "time": float(g[2]),
            "segments": g[3],
            "replication_seed": g[4],
            "circuits": g[5],
            "shots": g[6],
            "ground_state": g[7],
            "n_sys": NUM_SYSTEM_QUBITS,
            "group_by": "median"
        }
        for g in grid
    ]
    
    print(f"📋 Total configurations to run: {len(configs)}")
    
    # ─── Setup Output Files ────────────────────────────────────────────────
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pathlib.Path(f"results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "results.csv"
    metadata_path = output_dir / "metadata.json"
    
    # Save metadata
    metadata = {
        "test_id": str(TEST_ID),
        "timestamp": timestamp,
        "num_configurations": len(configs),
        "num_system_qubits": NUM_SYSTEM_QUBITS,
        "container_limits": {
            "cpu": cpu_limit,
            "memory_gb": memory_limit / (1024**3) if memory_limit else None
        },
        "parameters": {
            "num_ancilla": NUM_ANCILLA,
            "times": TIMES.tolist(),
            "segments": NUM_QDRIFT_SEGMENTS,
            "circuits_per_point": RANDOM_CIRCUITS_PER_DATAPOINT,
            "shots_per_circuit": SHOTS_PER_CIRCUIT,
            "replication_seeds": REPLICATION_SEEDS,
            "estimate_ground_state": ESTIMATE_GROUND_STATE
        },
        "hamiltonians": {k: str(v) for k, v in HAMILTONIANS_TO_TEST.items()}
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadata saved to: {metadata_path}")
    
    # ─── Initialize Shared Resources ───────────────────────────────────────
    
    csv_writer = ThreadSafeCSVWriter(csv_path, QPEResult)
    progress = SimulationProgressTracker(len(configs))
    circuit_factory = CircuitFactory(HAMILTONIANS_TO_TEST, PLACEHOLDER)
    
    # ─── Setup Signal Handler ──────────────────────────────────────────────
    
    def signal_handler(signum, frame):
        print(f"\n⚠️  Interrupted! Results saved to: {csv_path}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # ─── Run Simulations ───────────────────────────────────────────────────
    
    if parallel:
        # Calculate optimal worker count
        # For long-running tasks, don't oversaturate
        n_workers = min(
            cpu_limit - 1,  # Leave 1 CPU for system
            len(configs),    # Don't have more workers than tasks
            16              # Reasonable upper limit
        )
        n_workers = max(1, n_workers)
        
        print(f"🚀 Starting {n_workers} parallel workers...")
        print("=" * 60)
        
        # Create simulator pool
        sim_pool = SimulatorPool(n_workers)
        
        # Create a single simulator instance to pass to workers
        # (each worker will configure its own threads)
        simulator = sim_pool.create_simulator()
        
        # Use ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=worker_init,
            initargs=(circuit_factory, simulator, csv_writer, progress)
        ) as executor:
            
            # Submit all tasks
            futures = {
                executor.submit(run_simulation_worker, config): config
                for config in configs
            }
            
            # Process as they complete
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=600)  # 10-minute timeout
                except Exception as e:
                    print(f"Task failed for config: {futures[future]}")
                    print(f"Error: {e}")
    
    else:
        print("🔄 Running simulations sequentially...")
        print("=" * 60)
        
        # Initialize worker resources for sequential execution
        sim_pool = SimulatorPool(1)
        simulator = sim_pool.create_simulator()
        worker_init(circuit_factory, simulator, csv_writer, progress)
        
        # Run sequentially
        for config in configs:
            run_simulation_worker(config)
    
    # ─── Finalize ──────────────────────────────────────────────────────────
    
    print("=" * 60)
    elapsed = (time.time() - progress.start_time) / 60
    print(f"✅ Complete! Total time: {elapsed:.1f} minutes")
    print(f"📊 Success rate: {progress.completed_tasks}/{progress.total_tasks}")
    print(f"💾 Results saved to: {csv_path}")
    print(f"📄 Metadata saved to: {metadata_path}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Entry Point
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimized QPE simulation with qDrift for AWS/Container deployment"
    )
    parser.add_argument(
        "--sequential", 
        action="store_true",
        help="Run sequentially instead of parallel"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output for debugging"
    )
    
    args = parser.parse_args()
    
    # Run main function
    main(verbose_export=args.verbose, parallel=not args.sequential)