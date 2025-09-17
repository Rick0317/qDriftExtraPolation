"""
Memory and performance profiling utilities for quantum simulation experiments.

This module provides lightweight, multiprocessing-friendly profiling tools
that can be used to monitor memory usage and execution time, plus realistic
parameter sweep planning based on empirical overhead measurements.
"""
from __future__ import annotations
import time
import threading
import tracemalloc
from functools import wraps
from typing import Callable, TypeVar, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import re
from typing import List, Dict


import psutil

T = TypeVar('T')


def peak_rss_during(fn: Callable[[], T], *, dt: float = 0.05) -> Tuple[T, float]:
    """
    Run `fn()` and return (result, peak_RSS_in_MiB).

    The RSS (resident-set size) is sampled inside the *same* process,
    hence no fork is required — fully compatible with daemon workers.

    Parameters
    ----------
    fn : Callable[[], T]
        Workload whose memory profile we want to observe.
    dt : float, default 0.05
        Sampling period in seconds. 50 ms gives <1% CPU overhead while
        detecting peaks that last a few scheduler quanta.

    Notes
    -----
    • We use a daemon `threading.Thread` because daemonic processes are
      not allowed to start child *processes*.
    • The value returned is the true high-water-mark of resident memory
      (not Python allocations only). Interpreting RSS still requires
      caution.

    Returns
    -------
    (T, float)
        The original return value of `fn` and the peak RSS in MiB.
    """
    proc = psutil.Process()
    peak_bytes = 0
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


def profile_memory_and_time(func: Callable) -> Callable:
    """
    Decorator to profile memory usage and execution time of a function.
    
    The decorated function must return an object that has the following
    attributes that will be set by this decorator:
    - peak_MB: float (peak memory usage in MB)
    - runtime: float (execution time in seconds)
    - top10_py_alloc: str (top 10 Python allocation sites)
    
    This decorator is designed to work with multiprocessing.
    
    Args:
        func: Function to be profiled
        
    Returns:
        Decorated function that adds profiling information to the result
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        t0 = time.perf_counter()
        
        # Run the function with memory monitoring
        result, peak_psutil = peak_rss_during(lambda: func(*args, **kwargs))
        runtime = time.perf_counter() - t0

        # Extract top-N allocation sites
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("lineno")[:10]     # top-10
        top10 = "; ".join(f"{st.traceback[0]}: {st.size/1024:.1f} KiB"
                         for st in stats)
        
        # Add profiling info to result
        result.top10_py_alloc = top10
        result.peak_MB = peak_psutil
        result.runtime = runtime
        
        tracemalloc.stop()
        return result
    
    return wrapper


class SystemMonitor:
    """
    Context manager for monitoring system resources during execution.
    
    Example:
        with SystemMonitor() as monitor:
            # do some work
            pass
        
        print(f"Peak memory: {monitor.peak_memory_mb:.1f} MB")
        print(f"Average CPU: {monitor.avg_cpu_percent:.1f}%")
    """
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize the system monitor.
        
        Args:
            sampling_interval: How often to sample system stats (seconds)
        """
        self.sampling_interval = sampling_interval
        self.peak_memory_mb = 0.0
        self.avg_cpu_percent = 0.0
        self.cpu_samples = []
        self._stop_event = None
        self._monitor_thread = None
        self._process = psutil.Process()
    
    def __enter__(self):
        """Start monitoring."""
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and calculate final statistics."""
        if self._stop_event:
            self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
        
        # Calculate average CPU usage
        if self.cpu_samples:
            self.avg_cpu_percent = sum(self.cpu_samples) / len(self.cpu_samples)
    
    def _monitor_loop(self):
        """Main monitoring loop running in separate thread."""
        while not self._stop_event.is_set():
            # Monitor memory
            memory_info = self._process.memory_info()
            current_memory_mb = memory_info.rss / 1024**2
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)
            
            # Monitor CPU (averaged over sampling interval)
            cpu_percent = self._process.cpu_percent()
            if cpu_percent != 0.0:  # Skip first measurement (always 0)
                self.cpu_samples.append(cpu_percent)
            
            time.sleep(self.sampling_interval)


def get_system_info() -> dict[str, Any]:
    """
    Get current system information useful for experiment metadata.
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import multiprocessing
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": multiprocessing.cpu_count(),
        "total_memory_gb": psutil.virtual_memory().total / 1024**3,
        "available_memory_gb": psutil.virtual_memory().available / 1024**3,
    }

# ════════════════════════════════════════════════════════════════════════════


def parse_allocation_string(alloc_string: str) -> List[Tuple[str, str, float]]:
    """
    Parse the top10_py_alloc string into structured data.
    
    Args:
        alloc_string: String from top10_py_alloc field
        
    Returns:
        List of (file_path, line_number, memory_kb) tuples
    """
    if not alloc_string or alloc_string == "":
        return []
    
    allocations = []
    # Pattern to match: "file.py:line_number: memory_amount KiB"
    pattern = r'([^:]+):(\d+):\s*([\d.]+)\s*KiB'
    
    for match in re.finditer(pattern, alloc_string):
        file_path = match.group(1).strip()
        line_number = match.group(2)
        memory_kb = float(match.group(3))
        allocations.append((file_path, line_number, memory_kb))
    
    return allocations


def analyze_memory_hotspots(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze memory allocation patterns across a parameter sweep.
    
    Args:
        df: DataFrame with experimental results including 'top10_py_alloc' column
        
    Returns:
        Dictionary with analysis results
    """
    all_allocations = []
    file_totals = defaultdict(float)
    line_totals = defaultdict(float)
    
    for alloc_string in df['top10_py_alloc'].dropna():
        allocations = parse_allocation_string(alloc_string)
        all_allocations.extend(allocations)
        
        for file_path, line_number, memory_kb in allocations:
            # Extract just the filename for cleaner analysis
            filename = file_path.split('/')[-1]
            file_totals[filename] += memory_kb
            line_totals[f"{filename}:{line_number}"] += memory_kb
    
    # Find the biggest memory consumers
    top_files = sorted(file_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    top_lines = sorted(line_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Count frequency of problematic files
    file_frequency = Counter(file_path.split('/')[-1] for file_path, _, _ in all_allocations)
    
    analysis = {
        "total_allocations_tracked": len(all_allocations),
        "top_files_by_memory": top_files,
        "top_lines_by_memory": top_lines, 
        "most_frequent_files": file_frequency.most_common(10),
        "average_allocation_kb": sum(mem for _, _, mem in all_allocations) / len(all_allocations) if all_allocations else 0
    }
    
    return analysis


def identify_memory_regression_patterns(df: pd.DataFrame) -> Dict[str, any]:
    """
    Identify if certain parameter combinations cause specific memory allocation patterns.
    
    Args:
        df: DataFrame with experimental results
        
    Returns:
        Analysis of memory patterns vs parameters
    """
    # Group by parameter combinations to find patterns
    memory_by_params = {}
    
    for _, row in df.iterrows():
        param_key = f"anc{row['num_ancilla']}_t{row['time']:.3f}_shots{row['n_shots']}"
        allocations = parse_allocation_string(row.get('top10_py_alloc', ''))
        
        if allocations:
            total_python_memory = sum(mem for _, _, mem in allocations)
            memory_by_params[param_key] = {
                'total_python_kb': total_python_memory,
                'peak_mb': row.get('peak_MB', 0),
                'allocations': allocations,
                'params': {
                    'num_ancilla': row['num_ancilla'],
                    'time': row['time'],
                    'shots': row['n_shots']
                }
            }
    
    # Find configurations with unusually high Python allocation
    if memory_by_params:
        avg_python_memory = sum(data['total_python_kb'] for data in memory_by_params.values()) / len(memory_by_params)
        
        high_memory_configs = {
            k: v for k, v in memory_by_params.items() 
            if v['total_python_kb'] > avg_python_memory * 1.5
        }
        
        regression_analysis = {
            "average_python_memory_kb": avg_python_memory,
            "high_memory_configurations": high_memory_configs,
            "memory_scaling_with_ancilla": analyze_memory_vs_parameter(memory_by_params, 'num_ancilla'),
            "memory_scaling_with_time": analyze_memory_vs_parameter(memory_by_params, 'time'),
            "memory_scaling_with_shots": analyze_memory_vs_parameter(memory_by_params, 'shots')
        }
    else:
        regression_analysis = {"error": "No allocation data found"}
    
    return regression_analysis


def analyze_memory_vs_parameter(memory_data: Dict, param_name: str) -> Dict:
    """Helper function to analyze memory scaling vs a specific parameter."""
    param_memory = defaultdict(list)
    
    for config_data in memory_data.values():
        param_value = config_data['params'][param_name]
        param_memory[param_value].append(config_data['total_python_kb'])
    
    scaling_analysis = {}
    for param_value, memory_values in param_memory.items():
        scaling_analysis[param_value] = {
            'avg_memory_kb': sum(memory_values) / len(memory_values),
            'max_memory_kb': max(memory_values),
            'min_memory_kb': min(memory_values),
            'sample_count': len(memory_values)
        }
    
    return scaling_analysis


def plot_memory_hotspots(analysis: Dict, save_path: str = None):
    """
    Create visualizations of memory allocation patterns.
    
    Args:
        analysis: Results from analyze_memory_hotspots()
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot top files by memory
    if analysis["top_files_by_memory"]:
        files, memories = zip(*analysis["top_files_by_memory"])
        ax1.barh(range(len(files)), memories)
        ax1.set_yticks(range(len(files)))
        ax1.set_yticklabels(files)
        ax1.set_xlabel('Total Memory Allocated (KB)')
        ax1.set_title('Top Files by Memory Allocation')
        ax1.invert_yaxis()
    
    # Plot most frequent problematic files
    if analysis["most_frequent_files"]:
        files, frequencies = zip(*analysis["most_frequent_files"])
        ax2.barh(range(len(files)), frequencies)
        ax2.set_yticks(range(len(files)))
        ax2.set_yticklabels(files)
        ax2.set_xlabel('Frequency of Appearance in Top 10')
        ax2.set_title('Most Frequently Problematic Files')
        ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_memory_report(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive text report of memory allocation patterns.
    
    Args:
        df: DataFrame with experimental results
        
    Returns:
        Formatted report string
    """
    hotspot_analysis = analyze_memory_hotspots(df)
    regression_analysis = identify_memory_regression_patterns(df)
    
    report = []
    report.append("=" * 60)
    report.append("MEMORY ALLOCATION ANALYSIS REPORT")
    report.append("=" * 60)
    
    report.append(f"\nTOTAL ALLOCATIONS TRACKED: {hotspot_analysis['total_allocations_tracked']}")
    report.append(f"AVERAGE ALLOCATION SIZE: {hotspot_analysis['average_allocation_kb']:.1f} KB")
    
    report.append("\n" + "─" * 40)
    report.append("TOP MEMORY-CONSUMING FILES:")
    report.append("─" * 40)
    for i, (filename, total_kb) in enumerate(hotspot_analysis["top_files_by_memory"][:5], 1):
        report.append(f"{i:2d}. {filename:<30} {total_kb:>8.1f} KB")
    
    report.append("\n" + "─" * 40)
    report.append("TOP MEMORY-CONSUMING CODE LINES:")
    report.append("─" * 40)
    for i, (line_info, total_kb) in enumerate(hotspot_analysis["top_lines_by_memory"][:5], 1):
        report.append(f"{i:2d}. {line_info:<40} {total_kb:>8.1f} KB")
    
    report.append("\n" + "─" * 40)
    report.append("MOST FREQUENTLY PROBLEMATIC FILES:")
    report.append("─" * 40)
    for i, (filename, frequency) in enumerate(hotspot_analysis["most_frequent_files"][:5], 1):
        report.append(f"{i:2d}. {filename:<30} appears {frequency:>3d} times")
    
    if "high_memory_configurations" in regression_analysis:
        report.append("\n" + "─" * 40)
        report.append("HIGH MEMORY CONFIGURATIONS:")
        report.append("─" * 40)
        for config_name, config_data in list(regression_analysis["high_memory_configurations"].items())[:3]:
            params = config_data['params']
            report.append(f"Config: {config_name}")
            report.append(f"  Parameters: {params['num_ancilla']} ancilla, t={params['time']:.3f}, {params['shots']} shots")
            report.append(f"  Python memory: {config_data['total_python_kb']:.1f} KB")
            report.append(f"  Total peak: {config_data['peak_mb']:.1f} MB")
            report.append("")
    
    # Add recommendations
    report.append("\n" + "─" * 40)
    report.append("OPTIMIZATION RECOMMENDATIONS:")
    report.append("─" * 40)
    
    top_file = hotspot_analysis["top_files_by_memory"][0][0] if hotspot_analysis["top_files_by_memory"] else "N/A"
    if "qiskit" in top_file.lower():
        report.append("• Qiskit operations dominate memory usage - consider:")
        report.append("  - Using matrix product state simulator for low-entanglement circuits")
        report.append("  - Implementing circuit chunking for large parameter sweeps")
        report.append("  - Upgrading to latest Qiskit version for memory optimizations")
    
    if "transpiler" in str(hotspot_analysis["top_files_by_memory"]):
        report.append("• Transpilation memory spikes detected - consider:")
        report.append("  - Pre-transpiling template circuits")
        report.append("  - Using lighter transpilation optimization levels")
        report.append("  - Caching transpiled circuits when possible")
    
    avg_python_kb = regression_analysis.get("average_python_memory_kb", 0)
    if avg_python_kb > 5000:  # > 5MB Python allocations
        report.append("• High Python memory usage detected - consider:")
        report.append("  - Reducing intermediate data structures")
        report.append("  - Implementing streaming for large result sets")
        report.append("  - Using generators instead of lists where possible")
    
    return "\n".join(report)


# Example usage function
def analyze_parameter_sweep_memory(csv_path: str):
    """
    Complete analysis of memory allocation patterns from a parameter sweep CSV.
    
    Args:
        csv_path: Path to the CSV file with experimental results
    """
    df = pd.read_csv(csv_path)
    
    # Generate comprehensive report
    report = generate_memory_report(df)
    print(report)
    
    # Create visualizations
    hotspot_analysis = analyze_memory_hotspots(df)
    plot_memory_hotspots(hotspot_analysis, save_path="memory_hotspots.png")
    
    # Save detailed analysis to file
    with open("memory_analysis_detailed.txt", "w") as f:
        f.write(report)
        f.write("\n\n" + "=" * 60)
        f.write("\nDETAILED REGRESSION ANALYSIS:")
        f.write("\n" + "=" * 60)
        
        regression_analysis = identify_memory_regression_patterns(df)
        for key, value in regression_analysis.items():
            f.write(f"\n{key}:\n{value}\n")
    
    return {
        "report": report,
        "hotspot_analysis": hotspot_analysis,
        "regression_analysis": identify_memory_regression_patterns(df)
    }