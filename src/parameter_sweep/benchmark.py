"""
Ready-to-Use Benchmark Script for Quantum Simulation Parallelization

SETUP INSTRUCTIONS:
1. Make sure this file is in the same directory as your main quantum simulation script
2. Update the import on line 23 to match your actual script filename
3. Run: python benchmark_my_quantum_sim.py

The script will:
- Test multiple parallelization methods (sequential, multiprocessing, joblib, etc.)
- Try different worker counts
- Generate detailed performance reports
- Create visualizations
- Provide specific recommendations for your hardware
"""

import multiprocessing as mp
import sys
import pathlib

# ============================================================================
# CONFIGURATION - EDIT THESE TO MATCH YOUR SETUP
# ============================================================================

# IMPORTANT: Update this import to match your actual script name
# If your script is called "my_quantum_script.py", change line below to:
# from my_quantum_script import run_simulation, HAMILTONIANS_TO_TEST, NUM_SYSTEM_QUBITS
try:
    # Try importing from the document you provided
    # You'll need to rename your script or update this import
    from optimized_driver_factory_function_cache import run_simulation, HAMILTONIANS_TO_TEST, NUM_SYSTEM_QUBITS
except ImportError:
    print("ERROR: Could not import from your quantum simulation script!")
    print("Please update the import statement in this file (line 23)")
    print("to match your actual script filename.")
    sys.exit(1)

from parallelization_benchmarker import benchmark_quantum_simulation
    

# Benchmark configuration
N_SAMPLE_CONFIGS = 20  # Number of test configurations (start small!)
TEST_METHODS = [
    'sequential',      # Baseline - no parallelization
    'mp_pool',         # Your current implementation
    'process',         # ProcessPoolExecutor
    'joblib_loky',     # Joblib with loky backend (often best)
    # 'thread',        # Uncomment to test threading (usually poor for CPU-bound)
    # 'joblib_mp',     # Uncomment to test joblib with multiprocessing backend
]

# Worker counts to test
# Will automatically use: 1, 2, 4, 8, half of CPU count, and full CPU count
AUTO_WORKER_COUNTS = True

# Manual worker counts (only used if AUTO_WORKER_COUNTS = False)
MANUAL_WORKER_COUNTS = [1, 2, 4, 8, 12]


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def main():
    """Run the comprehensive parallelization benchmark"""
    
    print("="*80)
    print("QUANTUM SIMULATION PARALLELIZATION BENCHMARK")
    print("="*80)
    print()
    
    # System information
    cpu_count = mp.cpu_count()
    print(f"System Information:")
    print(f"  • CPU cores: {cpu_count}")
    print(f"  • Python: {sys.version.split()[0]}")
    print()
    
    # Determine worker counts to test
    if AUTO_WORKER_COUNTS:
        workers = [1, 2, 4, 8, cpu_count // 2, cpu_count]
        workers = sorted(list(set([w for w in workers if w > 0 and w <= cpu_count])))
    else:
        workers = MANUAL_WORKER_COUNTS
    
    print(f"Benchmark Configuration:")
    print(f"  • Sample size: {N_SAMPLE_CONFIGS} configurations")
    print(f"  • Methods: {', '.join(TEST_METHODS)}")
    print(f"  • Worker counts: {workers}")
    print()
    
    # Estimate runtime
    # Very rough estimate: 1-2 seconds per config per worker count
    estimated_tests = N_SAMPLE_CONFIGS * len(workers) * len(TEST_METHODS)
    estimated_time = estimated_tests * 1.5 / 60  # Convert to minutes
    print(f"Estimated runtime: {estimated_time:.1f} - {estimated_time*2:.1f} minutes")
    print()
    
    input("Press Enter to start the benchmark (or Ctrl+C to cancel)...")
    print()
    
    # Generate test configurations
 
    # Fallback: create configs manually
    configs = []
    for ham_key in list(HAMILTONIANS_TO_TEST.keys())[:1]:  # Just use first Hamiltonian
        for _ in range(N_SAMPLE_CONFIGS):
            cfg = {
                'ham': ham_key,
                'anc': 10, 
                'time': 0.5,
                'segments': 1,
                'replication_seed': 42,
                'circuits': 20,  # Small for fast benchmarking
                'shots': 100,
                'ground_state': False,
                'trajectory_report_protocol': {"group": True, "group_by": "median"},
                'exceptionally_stupid_eigenstate': False
            }
            configs.append(cfg)
    
    print(f"✓ Generated {len(configs)} test configurations")
    print()
    
    # Run the benchmark
    try:
        results_df = benchmark_quantum_simulation(
            run_simulation_func=run_simulation,
            sample_configs=configs,
            worker_counts=workers,
            test_methods=TEST_METHODS
        )
        
        print()
        print("="*80)
        print("BENCHMARK COMPLETE!")
        print("="*80)
        print()
        
        # Display top recommendations
        print("TOP 3 RECOMMENDATIONS:")
        print()
        for i, (idx, row) in enumerate(results_df.head(3).iterrows(), 1):
            print(f"{i}. {row['Method']}")
            print(f"   • Speedup: {row['Speedup']:.2f}x faster")
            print(f"   • Time: {row['Time_Minutes']:.2f} minutes (for this test size)")
            print(f"   • Efficiency: {row['Efficiency_Percent']:.1f}%")
            print(f"   • Workers: {row['Workers']}")
            print()
        
        # Specific recommendation
        best = results_df.iloc[0]
        print("="*80)
        print("RECOMMENDED CONFIGURATION FOR YOUR SCRIPT")
        print("="*80)
        print()
        print(f"Method: {best['Method_Type']}")
        print(f"Workers: {best['Workers']}")
        print()
        
        # Show code example
        if 'mp_pool' in best['Method']:
            print("In your main script, use:")
            print(f"    with mp.Pool(processes={best['Workers']}) as pool:")
            print("        results = pool.map(run_simulation, configs)")
        elif 'process' in best['Method']:
            print("In your main script, use:")
            print("    from concurrent.futures import ProcessPoolExecutor")
            print(f"    with ProcessPoolExecutor(max_workers={best['Workers']}) as executor:")
            print("        results = list(executor.map(run_simulation, configs))")
        elif 'joblib' in best['Method']:
            print("In your main script, use:")
            print("    from joblib import Parallel, delayed")
            print(f"    results = Parallel(n_jobs={best['Workers']}, backend='loky')(")
            print("        delayed(run_simulation)(cfg) for cfg in configs")
            print("    )")
        
        print()
        print("Check the generated CSV and PNG files for detailed analysis!")
        print()
        
        return results_df
        
    except Exception as e:
        print(f"ERROR during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_memory_check():
    """Quick check of available memory before running"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        
        print(f"Memory Check:")
        print(f"  • Total: {total_gb:.2f} GB")
        print(f"  • Available: {available_gb:.2f} GB")
        print(f"  • Usage: {mem.percent}%")
        print()
        
        if available_gb < 2:

            
            print("⚠️  WARNING: Low available memory!")
            print("   Consider reducing N_SAMPLE_CONFIGS or worker counts")
            print()
    except ImportError:
        print("(psutil not available - skipping memory check)")
        print()


if __name__ == "__main__":
    # Quick pre-flight checks
    quick_memory_check()
    
    # Run the benchmark
    results = main()
    
    if results is not None:
        print("✓ Benchmark completed successfully!")
    else:
        print("✗ Benchmark encountered errors")
        sys.exit(1)