#!/usr/bin/env python
"""
Runner script for Stochastic QPE Parameter Sweep

This script exists to ensure the main module is IMPORTED (not run as __main__),
which is required for joblib's loky backend to pickle the cached functions correctly.

Usage:
    python run_sweep.py                        # Default: BALANCED strategy (6 workers)
    python run_sweep.py --strategy MAXIMUM_SPEED   # 12 workers (all logical cores)
    python run_sweep.py --strategy EFFICIENT       # 4 workers (memory-efficient)
    python run_sweep.py --sequential               # No parallelization (debugging)
    python run_sweep.py --no-progress              # Disable progress bar
    python run_sweep.py --verbose                  # Include all configs in metadata

Why this file exists:
    When you run `python some_module.py`, Python sets that module's __name__ to '__main__'.
    Any functions defined in that file live in the '__main__' namespace.
    
    Joblib's loky backend pickles functions by reference: ('module_name', 'function_name').
    Worker processes try to import '__main__' to find those functions, but their '__main__'
    is the loky bootstrap code, not your script. Result: AttributeError.
    
    By importing the module from this runner script, the functions live in 
    'optimized_driver_factory_function_cache' (a real module name), and workers can find them.
"""

from optimized_driver_factory_function_cache import run_cli

if __name__ == '__main__':
    run_cli()