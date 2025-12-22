
import time
import random
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Any, Callable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
import pathlib


def save_benchmark_to_csv(results: Dict[str, float], filename: str = 'quantum_benchmark_results.csv') -> pd.DataFrame:
    """
    Save benchmark results to CSV with all metrics
    
    Args:
        results: Dictionary mapping method name to execution time
        filename: Output CSV filename
    
    Returns:
        DataFrame with comprehensive metrics
    """
    data = []
    
    # Get baseline time for speedup calculation
    baseline_time = results.get('sequential', results[list(results.keys())[0]])
    
    for method, exec_time in results.items():
        # Extract worker count
        if 'workers_' in method:
            n_workers = int(method.split('_')[-1]) if method.split('_')[-1].isdigit() else mp.cpu_count()
        elif 'jobs_' in method:
            workers_str = method.split('_')[-1]
            n_workers = mp.cpu_count() if workers_str == '-1' else int(workers_str)
        else:
            n_workers = 1
        
        speedup = baseline_time / exec_time if exec_time > 0 else 0
        efficiency = (speedup / n_workers * 100) if n_workers > 0 else 0
        
        # Categorize method type
        if 'process' in method.lower() or 'mp_pool' in method:
            method_type = 'Process'
        elif 'thread' in method.lower():
            method_type = 'Thread'
        elif 'joblib' in method:
            if 'loky' in method:
                method_type = 'Joblib-Loky'
            elif 'multiprocessing' in method:
                method_type = 'Joblib-Multiprocessing'
            else:
                method_type = 'Joblib-Threading'
        else:
            method_type = 'Sequential'
        
        data.append({
            'Method': method,
            'Method_Type': method_type,
            'Workers': n_workers,
            'Time_Seconds': exec_time,
            'Time_Minutes': exec_time / 60,
            'Speedup': speedup,
            'Efficiency_Percent': efficiency,
            'Throughput_Tasks_Per_Sec': len(results) / exec_time if exec_time > 0 else 0
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Time_Seconds')
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"\n✓ Results saved to {filename}")
    
    return df


def visualize_benchmark_results(df: pd.DataFrame, save_prefix: str = 'quantum_benchmark') -> plt.Figure:
    """
    Create comprehensive visualizations of benchmark results
    
    Args:
        df: DataFrame with benchmark results
        save_prefix: Prefix for saved figure filename
    
    Returns:
        matplotlib Figure object
    """
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Execution Time Comparison (Top 15)
    ax1 = plt.subplot(2, 3, 1)
    top_15 = df.nsmallest(15, 'Time_Seconds')
    colors = sns.color_palette("RdYlGn_r", len(top_15))
    bars = ax1.barh(range(len(top_15)), top_15['Time_Seconds'], color=colors)
    ax1.set_yticks(range(len(top_15)))
    ax1.set_yticklabels(top_15['Method'], fontsize=9)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_title('Execution Time - Top 15 Methods', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add time labels on bars
    for i, (idx, row) in enumerate(top_15.iterrows()):
        ax1.text(row['Time_Seconds'] * 1.02, i, f"{row['Time_Seconds']:.1f}s", 
                va='center', fontsize=8)
    
    # 2. Speedup vs Workers
    ax2 = plt.subplot(2, 3, 2)
    for method_type in df['Method_Type'].unique():
        subset = df[df['Method_Type'] == method_type].sort_values('Workers')
        if len(subset) > 1:  # Only plot if multiple data points
            ax2.plot(subset['Workers'], subset['Speedup'], 
                    marker='o', label=method_type, linewidth=2, markersize=8)
    
    # Add ideal speedup line
    max_workers = df['Workers'].max()
    ax2.plot([1, max_workers], [1, max_workers], 
            'k--', alpha=0.3, label='Ideal (linear)', linewidth=1)
    
    ax2.set_xlabel('Number of Workers', fontsize=11)
    ax2.set_ylabel('Speedup', fontsize=11)
    ax2.set_title('Speedup vs Number of Workers', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_workers + 1)
    
    # 3. Efficiency Heatmap
    ax3 = plt.subplot(2, 3, 3)
    
    # Pivot data for heatmap
    pivot_data = df.pivot_table(
        values='Efficiency_Percent', 
        index='Method_Type', 
        columns='Workers', 
        aggfunc='max'
    )
    
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                    center=50, vmin=0, vmax=100, ax=ax3, cbar_kws={'label': 'Efficiency %'})
        ax3.set_title('Efficiency Heatmap (% of ideal)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Number of Workers', fontsize=11)
        ax3.set_ylabel('Method Type', fontsize=11)
    
    # 4. Speedup Comparison (Top 10)
    ax4 = plt.subplot(2, 3, 4)
    top_10_speedup = df.nlargest(10, 'Speedup')
    colors_speedup = sns.color_palette("viridis", len(top_10_speedup))
    bars = ax4.bar(range(len(top_10_speedup)), top_10_speedup['Speedup'], color=colors_speedup)
    ax4.set_xticks(range(len(top_10_speedup)))
    ax4.set_xticklabels(top_10_speedup['Method'], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Speedup', fontsize=11)
    ax4.set_title('Top 10 Methods by Speedup', fontsize=12, fontweight='bold')
    ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax4.legend()
    
    # Add speedup labels on bars
    for i, (idx, row) in enumerate(top_10_speedup.iterrows()):
        ax4.text(i, row['Speedup'] + 0.1, f"{row['Speedup']:.2f}x", 
                ha='center', fontsize=8, fontweight='bold')
    
    # 5. Efficiency Distribution
    ax5 = plt.subplot(2, 3, 5)
    for method_type in df['Method_Type'].unique():
        subset = df[df['Method_Type'] == method_type]
        ax5.scatter(subset['Workers'], subset['Efficiency_Percent'], 
                   label=method_type, s=100, alpha=0.6)
    
    ax5.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='100% Efficient')
    ax5.axhline(y=75, color='orange', linestyle='--', alpha=0.3, label='75% Efficient')
    ax5.axhline(y=50, color='r', linestyle='--', alpha=0.3, label='50% Efficient')
    ax5.set_xlabel('Number of Workers', fontsize=11)
    ax5.set_ylabel('Efficiency (%)', fontsize=11)
    ax5.set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8, loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, min(120, df['Efficiency_Percent'].max() + 10))
    
    # 6. Method Type Summary
    ax6 = plt.subplot(2, 3, 6)
    method_summary = df.groupby('Method_Type').agg({
        'Time_Seconds': 'min',
        'Speedup': 'max',
        'Efficiency_Percent': 'max'
    }).reset_index()
    
    x = np.arange(len(method_summary))
    width = 0.25
    
    bars1 = ax6.bar(x - width, method_summary['Speedup'], width, 
                    label='Max Speedup', color='skyblue')
    bars2 = ax6.bar(x, method_summary['Efficiency_Percent']/20, width, 
                    label='Max Efficiency/20', color='lightcoral')
    
    ax6.set_xlabel('Method Type', fontsize=11)
    ax6.set_ylabel('Value', fontsize=11)
    ax6.set_title('Best Performance by Method Type', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(method_summary['Method_Type'], rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{save_prefix}_visualization_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {filename}")
    
    plt.show()
    
    return fig


def analyze_benchmark_results(df: pd.DataFrame) -> None:
    """
    Provide detailed analysis of benchmark results
    
    Args:
        df: DataFrame with benchmark results
    """
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF QUANTUM SIMULATION BENCHMARK")
    print("="*80)
    
    if len(df) == 0:
        print("No results to analyze!")
        return
    
    best = df.iloc[0]
    worst = df.iloc[-1]
    
    print(f"\n🏆 WINNER: {best['Method']}")
    print(f"   Time: {best['Time_Seconds']:.1f}s ({best['Time_Minutes']:.2f} min)")
    print(f"   Speedup: {best['Speedup']:.2f}x")
    print(f"   Efficiency: {best['Efficiency_Percent']:.1f}%")
    print(f"   Workers: {best['Workers']}")

    print(f"\n📊 KEY INSIGHTS:")
    
    # 1. Best method type
    best_by_type = df.groupby('Method_Type')['Speedup'].max().sort_values(ascending=False)
    print(f"\n1. Best Method Types (by max speedup):")
    for method_type, speedup in best_by_type.head(3).items():
        best_method = df[df['Method_Type'] == method_type].loc[df['Speedup'].idxmax()]
        print(f"   - {method_type}: {speedup:.2f}x (with {best_method['Workers']} workers)")
    
    # 2. Worker count analysis
    print(f"\n2. Optimal Worker Count Analysis:")
    for method_type in df['Method_Type'].unique():
        subset = df[df['Method_Type'] == method_type]
        if len(subset) > 0:
            best_workers = subset.loc[subset['Speedup'].idxmax()]
            print(f"   - {method_type}:")
            print(f"     • Best: {best_workers['Workers']} workers ({best_workers['Speedup']:.2f}x speedup)")
            print(f"     • Efficiency: {best_workers['Efficiency_Percent']:.1f}%")
    
    # 3. Efficiency analysis
    print(f"\n3. Efficiency Analysis:")
    high_eff = df[df['Efficiency_Percent'] > 70]
    medium_eff = df[(df['Efficiency_Percent'] >= 50) & (df['Efficiency_Percent'] <= 70)]
    low_eff = df[df['Efficiency_Percent'] < 50]
    
    print(f"   - {len(high_eff)} methods achieved >70% efficiency (excellent)")
    print(f"   - {len(medium_eff)} methods achieved 50-70% efficiency (good)")
    print(f"   - {len(low_eff)} methods achieved <50% efficiency (poor)")
    
    if len(high_eff) > 0:
        best_eff = high_eff.iloc[0]
        print(f"   - Most efficient: {best_eff['Method']} ({best_eff['Efficiency_Percent']:.1f}%)")
    
    # 4. Warnings - methods slower than sequential
    slower = df[df['Speedup'] < 1.0]
    if len(slower) > 0:
        print(f"\n4. ⚠️  WARNING: {len(slower)} methods were SLOWER than sequential:")
        for _, row in slower.head(5).iterrows():
            overhead = (row['Time_Seconds'] / df[df['Method'] == 'sequential']['Time_Seconds'].iloc[0] - 1) * 100
            print(f"   - {row['Method']}: {row['Speedup']:.2f}x ({overhead:.1f}% slower due to overhead)")
    
    # 5. Recommendations
    print(f"\n5. 💡 RECOMMENDATIONS:")
    
    # Find sweet spot (best speedup with good efficiency)
    good_methods = df[(df['Speedup'] > 1.5) & (df['Efficiency_Percent'] > 60)]
    if len(good_methods) > 0:
        recommended = good_methods.iloc[0]
        print(f"   ✓ RECOMMENDED: {recommended['Method']}")
        print(f"     • {recommended['Speedup']:.2f}x speedup with {recommended['Efficiency_Percent']:.1f}% efficiency")
        print(f"     • Projected time for full run: {recommended['Time_Minutes']:.2f} minutes")
    else:
        print(f"   ✓ RECOMMENDED: {best['Method']}")
        print(f"     • Best overall performance despite lower efficiency")
    
    # Memory considerations
    print(f"\n   ℹ️  QUANTUM SIMULATION SPECIFIC NOTES:")
    print(f"   • Your code uses Qiskit AerSimulator with matrix_product_state")
    print(f"   • Each worker needs memory for quantum state simulation")
    print(f"   • ProcessPoolExecutor/mp.Pool avoid GIL but use more memory")
    print(f"   • Consider memory usage when choosing worker count")
    print(f"   • CPU count: {mp.cpu_count()} cores available")


def benchmark_quantum_simulation(
    run_simulation_func: Callable[[Dict[str, Any]], Any],
    sample_configs: List[Dict[str, Any]],
    worker_counts: List[int] = None,
    test_methods: List[str] = None
) -> pd.DataFrame:
    """
    Comprehensive benchmark for quantum simulation parallelization
    
    Args:
        run_simulation_func: The simulation function to benchmark (typically run_simulation)
        sample_configs: List of configuration dictionaries to test
        worker_counts: List of worker counts to test (default: [1, 2, 4, 8, cpu_count])
        test_methods: List of methods to test (default: all)
    
    Returns:
        DataFrame with benchmark results
    """
    if worker_counts is None:
        cpu_count_val = mp.cpu_count()
        worker_counts = [1, 2, 4, 8, cpu_count_val]
        # Remove duplicates and sort
        worker_counts = sorted(list(set(worker_counts)))
    
    if test_methods is None:
        test_methods = ['sequential', 'thread', 'process', 'mp_pool', 'joblib_loky']
    
    print(f"\n{'='*80}")
    print(f"QUANTUM SIMULATION PARALLELIZATION BENCHMARK")
    print(f"{'='*80}")
    print(f"\nTesting {len(sample_configs)} configurations")
    print(f"Worker counts: {worker_counts}")
    print(f"Methods: {test_methods}")
    print(f"CPU cores available: {mp.cpu_count()}\n")
    
    results = {}
    
    # 1. Sequential baseline
    if 'sequential' in test_methods:
        print("Testing Sequential (baseline)...")
        start = time.time()
        for cfg in sample_configs:
            run_simulation_func(cfg)
        results['sequential'] = time.time() - start
        print(f"✓ {results['sequential']:.1f}s ({results['sequential']/60:.2f} min)\n")
    
    # 2. ThreadPoolExecutor
    if 'thread' in test_methods:
        print("Testing ThreadPoolExecutor...")
        for n_workers in worker_counts:
            try:
                start = time.time()
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    list(executor.map(run_simulation_func, sample_configs))
                results[f'thread_workers_{n_workers}'] = time.time() - start
                print(f"  {n_workers} workers: {results[f'thread_workers_{n_workers}']:.1f}s")
            except Exception as e:
                print(f"  {n_workers} workers: FAILED ({str(e)[:50]})")
        print()
    
    # 3. ProcessPoolExecutor
    if 'process' in test_methods:
        print("Testing ProcessPoolExecutor...")
        for n_workers in worker_counts:
            try:
                start = time.time()
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    list(executor.map(run_simulation_func, sample_configs))
                results[f'process_workers_{n_workers}'] = time.time() - start
                print(f"  {n_workers} workers: {results[f'process_workers_{n_workers}']:.1f}s")
            except Exception as e:
                print(f"  {n_workers} workers: FAILED ({str(e)[:50]})")
        print()
    
    # 4. multiprocessing.Pool (current implementation)
    if 'mp_pool' in test_methods:
        print("Testing multiprocessing.Pool (current implementation)...")
        for n_workers in worker_counts:
            try:
                start = time.time()
                with mp.Pool(processes=n_workers) as pool:
                    pool.map(run_simulation_func, sample_configs)
                results[f'mp_pool_workers_{n_workers}'] = time.time() - start
                print(f"  {n_workers} workers: {results[f'mp_pool_workers_{n_workers}']:.1f}s")
            except Exception as e:
                print(f"  {n_workers} workers: FAILED ({str(e)[:50]})")
        print()
    
    # 5. joblib with loky backend
    if 'joblib_loky' in test_methods:
        print("Testing joblib (loky backend)...")
        for n_workers in worker_counts:
            try:
                start = time.time()
                Parallel(n_jobs=n_workers, backend='loky')(
                    delayed(run_simulation_func)(cfg) for cfg in sample_configs
                )
                results[f'joblib_loky_jobs_{n_workers}'] = time.time() - start
                print(f"  {n_workers} workers: {results[f'joblib_loky_jobs_{n_workers}']:.1f}s")
            except Exception as e:
                print(f"  {n_workers} workers: FAILED ({str(e)[:50]})")
        print()
    
    # 6. joblib with multiprocessing backend
    if 'joblib_mp' in test_methods:
        print("Testing joblib (multiprocessing backend)...")
        for n_workers in worker_counts:
            try:
                start = time.time()
                Parallel(n_jobs=n_workers, backend='multiprocessing')(
                    delayed(run_simulation_func)(cfg) for cfg in sample_configs
                )
                results[f'joblib_mp_jobs_{n_workers}'] = time.time() - start
                print(f"  {n_workers} workers: {results[f'joblib_mp_jobs_{n_workers}']:.1f}s")
            except Exception as e:
                print(f"  {n_workers} workers: FAILED ({str(e)[:50]})")
        print()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save and analyze results
    df = save_benchmark_to_csv(results, f'quantum_benchmark_{timestamp}.csv')
    analyze_benchmark_results(df)
    visualize_benchmark_results(df, save_prefix=f'quantum_benchmark_{timestamp}')
    
    return df



if __name__ == "__main__":
    print(__doc__)
    print("\nThis is a benchmarking module for quantum simulation code.")
    print("\nTo use it:")
    print("1. Import it in your main quantum simulation script")
    print("2. Call benchmark_quantum_simulation() with your run_simulation function")
    print("3. Or use the example function: run_benchmark_on_quantum_code()")
    print("\nExample:")
    print("  from quantum_parallel_benchmark import benchmark_quantum_simulation")
    print("  from your_script import run_simulation")
    print("  ")
    print("  # Generate test configs")
    print("  configs = generate_test_configs_from_main_grid(n_sample=20)")
    print("  ")
    print("  # Run benchmark")
    print("  results = benchmark_quantum_simulation(run_simulation, configs)")