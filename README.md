# qDRIFT Extrapolation

A comprehensive quantum simulation framework for studying quantum phase estimation (QPE) combined with qDRIFT time-evolution algorithms, with a focus on zero-noise extrapolation techniques.

## Overview

This project implements optimized quantum algorithms for studying Hamiltonian simulation through qDRIFT-QPE (quantum Drift - Quantum Phase Estimation) protocols. The main goal is to develop and benchmark extrapolation techniques that can improve the accuracy of quantum phase estimation by correcting for qDRIFT discretization errors.

### Key Features

- **Optimized qDRIFT-QPE Implementation**: High-performance quantum circuits combining qDRIFT time evolution with quantum phase estimation
- **Chebyshev Node Sampling**: Strategic time point selection for optimal extrapolation
- **Memory-Efficient Parameter Sweeps**: Comprehensive parameter space exploration with memory profiling
- **Zero-Noise Extrapolation**: Polynomial extrapolation techniques to estimate exact eigenvalues
- **Multiple Hamiltonian Support**: Built-in support for H₂ molecule, Ising models, and custom Hamiltonians
- **Comprehensive Analysis Tools**: Statistical analysis, visualization, and error characterization

## Project Structure

```
qDriftExtraPolation/
├── src/
│   ├── algorithms/              # Core quantum algorithms
│   │   ├── optimized_qft_qpe_qdrift.py    # Main optimized implementation
│   │   ├── chebyshev.py                   # Chebyshev node generation
│   │   └── ...
│   ├── parameter_sweep/         # Parameter sweep drivers
│   │   ├── optimized_driver_factory_function_cache.py
│   │   └── ...
│   ├── utils/                   # Utility functions
│   │   ├── generate_hamiltonians.py       # Hamiltonian construction
│   │   ├── qpe_postprocessing_utils.py    # QPE result analysis
│   │   ├── memory_profiling_utils.py      # Performance monitoring
│   │   └── ...
│   ├── error_analysis/          # Error analysis tools
│   └── tests_basic_behavious_and_deprecated_functionality/
├── notebooks/                   # Jupyter notebooks
│   ├── experiments/             # Research experiments
│   ├── numerical_tests/         # Algorithm validation
│   └── demos/                   # Tutorial notebooks
├── results/                     # Experimental data and figures
│   ├── figures/                 # Generated plots
│   ├── parameter_sweep/         # Parameter sweep results
│   └── interp/                  # Extrapolation analysis
└── requirements.txt             # Python dependencies
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Rick0317/qDriftExtraPolation.git
cd qDriftExtraPolation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add project root to Python path (if running scripts directly):
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Basic Usage

#### Running Parameter Sweeps

```python
from src.parameter_sweep.optimized_driver_factory_function_cache import main

# Run a parameter sweep with default settings
main(verbose_export=True, parallel=False)
```

#### Analyzing Results

```python
import pandas as pd
from results.interp.extrapolation import plot_cheby_extrapolation

# Load and analyze results
df = pd.read_csv("qdrift_qpe_fc_parameter_sweep_2025-10-02.csv")
plot_cheby_extrapolation(groupbys=["ham"], df=df, show_stdevs=True)
```

#### Custom Hamiltonian Example

```python
from qiskit.quantum_info import SparsePauliOp
from src.utils.generate_hamiltonians import calculate_minimum_evolution_time

# Define a custom Hamiltonian
H = SparsePauliOp.from_list([("X", 0.2), ("Z", 0.5), ("I", 0.3)], num_qubits=1)

# Calculate minimum evolution time for QPE
t_min = calculate_minimum_evolution_time({"custom": H}, m=14)
print(f"Minimum evolution time: {t_min['custom']:.6f}")
```

## Algorithms

### qDRIFT-QPE Algorithm

The core algorithm combines:
1. **qDRIFT Time Evolution**: Stochastic compilation of Hamiltonian evolution into random Pauli rotations
2. **Quantum Phase Estimation**: Estimation of eigenvalues through quantum Fourier transform
3. **Chebyshev Node Sampling**: Optimal time point selection for polynomial extrapolation

### Zero-Noise Extrapolation

The extrapolation process:
1. Sample evolution times using Chebyshev nodes
2. Run qDRIFT-QPE at each time point
3. Fit polynomial to eigenvalue estimates vs. time
4. Extrapolate to t→0 to obtain error-corrected eigenvalue

## Key Parameters

- **NUM_ANCILLA**: Number of ancilla qubits for phase precision (14-16 recommended)
- **TIMES**: Evolution times generated using Chebyshev nodes
- **NUM_QDRIFT_SEGMENTS**: Number of qDRIFT segments per channel invocation
- **RANDOM_CIRCUITS_PER_DATAPOINT**: Statistical sampling for each parameter point
- **SHOTS_PER_CIRCUIT**: Measurement shots per quantum circuit

## Performance Features

- **Function Caching**: Expensive operations cached for reuse across parameter sweeps
- **Memory Profiling**: Built-in memory usage tracking and optimization
- **Parallel Processing**: Multi-core parameter sweep execution
- **Vectorized Operations**: Numba-optimized numerical computations

## Examples and Tutorials

See the `notebooks/` directory for:
- `experiments/h_2_minimal_basis.ipynb`: H₂ molecule simulation
- `experiments/iterative_phase_estimation.ipynb`: QPE algorithm analysis  
- `numerical_tests/`: Algorithm validation and benchmarking
- `demos/`: Basic usage tutorials


## Troubleshooting

´´´python


ubuntu@ip-172-31-42-128:~/qDriftExtraPolation$ pip install -r requirements.txt
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.

    If you wish to install a non-Debian-packaged Python package,
    create a virtual environment using python3 -m venv path/to/venv.
    Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
    sure you have python3-full installed.

    If you wish to install a non-Debian packaged Python application,
    it may be easiest to use pipx install xyz, which will manage a
    virtual environment for you. Make sure you have pipx installed.

    See /usr/share/doc/python3.12/README.venv for more information.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
´´´
To resolve the "externally-managed-environment" error when trying to install packages using pip, you have a few options:
1. **Use a Virtual Environment**:
   Create a virtual environment to isolate your Python packages from the system-wide installation. This is the recommended approach.
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```
   