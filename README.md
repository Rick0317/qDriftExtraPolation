# qDRIFT Extrapolation

A comprehensive quantum simulation framework for studying quantum phase estimation (QPE) combined with qDRIFT time-evolution algorithms, with a focus on zero-noise extrapolation techniques.

## Overview

This project implements optimized quantum algorithms for studying Hamiltonian simulation through qDRIFT-QPE (quantum Drift - Quantum Phase Estimation) protocols. The main goal is to develop and benchmark extrapolation techniques that can improve the accuracy of quantum phase estimation by correcting for qDRIFT discretization errors.

## Theoretical Background

### Quantum Phase Estimation (QPE)

Quantum Phase Estimation is a fundamental quantum algorithm for estimating eigenvalues of unitary operators [1,2]. Given a unitary operator $U$ with eigenvector $|\psi\rangle$ such that:

$$U|\psi\rangle = e^{2\pi i\varphi}|\psi\rangle$$

QPE estimates the phase $\varphi \in [0,1)$ with precision $2^{-n}$ using $n$ ancilla qubits.

#### QFT-based QPE Algorithm

The standard QFT-based QPE circuit [1] consists of three main steps:

1. **Hadamard Layer**: Initialize $n$ ancilla qubits in superposition
   $$|0\rangle^{\otimes n} \xrightarrow{H^{\otimes n}} \frac{1}{2^{n/2}}\sum_{k=0}^{2^n-1}|k\rangle$$

2. **Controlled Unitary Evolution**: Apply controlled-$U^{2^j}$ operations
   $$|\psi\rangle \otimes \frac{1}{2^{n/2}}\sum_{k=0}^{2^n-1}|k\rangle \xrightarrow{\text{ctrl-}U} |\psi\rangle \otimes \frac{1}{2^{n/2}}\sum_{k=0}^{2^n-1}e^{2\pi i\varphi k}|k\rangle$$

3. **Inverse QFT**: Extract phase information via quantum Fourier transform
   $$\text{QFT}^{-1}: |k\rangle \rightarrow \frac{1}{2^{n/2}}\sum_{j=0}^{2^n-1}e^{-2\pi ijk/2^n}|j\rangle$$

The measurement outcome approximates $\tilde{\varphi} = k/2^n$ where $k$ is the measured bit string interpreted as an integer.

#### Kitaev's Iterative QPE

Kitaev's approach [3] uses a single ancilla qubit and iterative measurements, reducing space complexity from $O(n)$ to $O(1)$ qubits at the cost of increased circuit depth. The phase is reconstructed bit-by-bit through adaptive measurements.

### Hamiltonian Simulation via qDRIFT

For Hamiltonian eigenvalue estimation, we need to simulate the time evolution operator:

$$U(t) = e^{-iHt}$$

where $H = \sum_{j=1}^{L} h_j$ is a sum of Pauli operators. The qDRIFT algorithm [4] provides a randomized product formula approach.

#### qDRIFT Product Formula

The qDRIFT channel approximates $e^{-iHt}$ as a random product [4]:

$$\mathcal{U}_{\text{qDrift}}(t) = \prod_{k=1}^{N} e^{-i\tau H_{j_k}}$$

where:
- $N$ is the number of segments
- $\tau = \lambda t / N$ with $\lambda = \sum_{j=1}^{L}|h_j|$ (1-norm of Hamiltonian)
- Each $j_k$ is sampled independently from distribution $p_j = |h_j|/\lambda$

**Theorem (qDRIFT Error Bound)** [4]: The diamond distance between the ideal evolution and qDRIFT channel satisfies:

$$\mathbb{E}\left[\left\|\mathcal{U}_{\text{qDrift}}(t) - e^{-iHt}\right\|_{\diamond}\right] \leq \frac{\lambda^2 t^2}{2N}$$

This quadratic scaling in time motivates our extrapolation approach.

### Chebyshev Node Sampling

To minimize interpolation error, we sample evolution times at Chebyshev nodes of the second kind [5]:

$$t_k = T_{\max} \cos\left(\frac{\pi k}{n}\right), \quad k = 0, 1, \ldots, n$$

where $T_{\max}$ is the maximum evolution time. Chebyshev nodes minimize the Runge phenomenon in polynomial interpolation [6], making them optimal for our extrapolation scheme.

**Theorem (Chebyshev Interpolation Error)** [5]: For a function $f \in C^{n+1}[a,b]$, interpolation at Chebyshev nodes satisfies:

$$\|f - p_n\|_{\infty} \leq \frac{(b-a)^{n+1}}{2^{2n+1}(n+1)!}\|f^{(n+1)}\|_{\infty}$$

where $p_n$ is the interpolating polynomial of degree $n$.

### Zero-Noise Extrapolation

Our extrapolation protocol leverages the known error scaling of qDRIFT to estimate the true eigenvalue $\lambda$ from noisy estimates $\tilde{\lambda}(t)$:

$$\tilde{\lambda}(t) = \lambda + c_1 t + c_2 t^2 + O(t^3)$$

By fitting a polynomial to measurements at multiple times and extrapolating to $t \rightarrow 0$, we recover an improved estimate $\lambda_{\text{ext}}$ with reduced systematic error.

## References

[1] M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press (2010).

[2] A. Yu. Kitaev, "Quantum measurements and the Abelian Stabilizer Problem," arXiv:quant-ph/9511026 (1995).

[3] A. Yu. Kitaev, A. Shen, and M. Vyalyi, *Classical and Quantum Computation*, American Mathematical Society (2002).

[4] E. Campbell, "Random Compiler for Fast Hamiltonian Simulation," Phys. Rev. Lett. **123**, 070503 (2019). [arXiv:1811.08017](https://arxiv.org/abs/1811.08017)

[5] L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM (2013).

[6] C. Runge, "Über empirische Funktionen und die Interpolation zwischen äquidistanten Ordinaten," Zeitschrift für Mathematik und Physik **46**, 224-243 (1901).

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

The core algorithm combines quantum phase estimation with qDRIFT Hamiltonian simulation [4,7]:

#### Algorithm Overview

**Input**: 
- Hamiltonian $H = \sum_{j=1}^{L} h_j P_j$ (sum of weighted Pauli operators)
- Eigenstate $|\psi\rangle$ (or approximation)
- Number of ancilla qubits $m$
- Evolution time $t$
- Number of qDRIFT segments $N$

**Output**: Estimate $\tilde{\lambda}$ of eigenvalue $\lambda$ where $H|\psi\rangle = \lambda|\psi\rangle$

**Procedure**:

1. **Initialization**: Prepare state $|0\rangle^{\otimes m} \otimes |\psi\rangle$

2. **Hadamard Layer**: Apply $H^{\otimes m}$ to ancilla qubits

3. **Controlled qDRIFT Evolution**: For each ancilla qubit $k \in \{0, \ldots, m-1\}$, apply controlled-$\mathcal{U}_{\text{qDrift}}(t)^{2^k}$:
   
   $$\text{ctrl-}\mathcal{U}_{\text{qDrift}}(t)^{2^k} = \prod_{r=1}^{2^k} \prod_{s=1}^{N} e^{-i\tau H_{j_{r,s}}}$$
   
   where each $j_{r,s}$ is sampled i.i.d. from $p_j = |h_j|/\lambda$

4. **Inverse QFT**: Apply $\text{QFT}^{-1}$ to ancilla register

5. **Measurement**: Measure ancilla qubits to obtain bit string $b$

6. **Post-processing**: Estimate eigenvalue as:
   
   $$\tilde{\lambda} = \frac{2\pi}{t} \cdot \frac{\text{int}(b)}{2^m}$$

#### Error Analysis

The total error in eigenvalue estimation has three main contributions:

1. **QPE Resolution Error** [1]: $O(2^{-m})$ from finite ancilla qubits

2. **qDRIFT Sampling Error** [4]: $O(\lambda^2 t^2 / N)$ from stochastic compilation

3. **Statistical Error**: $O(1/\sqrt{S})$ where $S$ is number of shots

The combined error scaling motivates the extrapolation strategy to eliminate the dominant $O(t^2)$ systematic error.

### Zero-Noise Extrapolation Protocol

Our extrapolation method extends Richardson extrapolation [8] to quantum algorithms:

#### Extrapolation Procedure

1. **Sample Chebyshev Nodes**: Generate $n+1$ evolution times
   $$\{t_k = T_{\max}\cos(\pi k/n)\}_{k=0}^{n}$$

2. **Execute qDRIFT-QPE**: Run algorithm at each $t_k$ with $N_k$ segments and $S_k$ shots to obtain estimates $\{\tilde{\lambda}_k\}$

3. **Polynomial Fitting**: Fit polynomial of degree $d$ (typically $d=2$):
   $$p(t) = a_0 + a_1 t + a_2 t^2 + \cdots + a_d t^d$$
   to data points $\{(t_k, \tilde{\lambda}_k)\}$

4. **Extrapolation**: Evaluate $\lambda_{\text{ext}} = p(0) = a_0$ as the extrapolated eigenvalue estimate

#### Theoretical Justification

The qDRIFT error bound [4] implies that the estimated eigenvalue has the expansion:

$$\tilde{\lambda}(t) = \lambda + \frac{\alpha^2 t^2}{2N} + O(t^4)$$

where $\alpha$ characterizes the Hamiltonian's non-commutativity. By extrapolating to $t \rightarrow 0$, we eliminate the leading-order $O(t^2)$ error, achieving:

$$|\lambda_{\text{ext}} - \lambda| = O(t^4) + O(2^{-m}) + O(N^{-1})$$

This can yield exponential improvement over standard qDRIFT-QPE when $N \gg \lambda^2 t^2$.

### Implementation Optimizations

Our implementation includes several performance enhancements:

1. **Function Caching**: Pre-compilation of Pauli evolution gates using Qiskit's `PauliEvolutionGate`

2. **Template Circuits**: Static circuit scaffolding with parameter binding for rapid trajectory generation

3. **Vectorized Post-processing**: NumPy-based eigenvalue extraction from measurement statistics

4. **Memory-Mapped Storage**: Efficient handling of large parameter sweep datasets

## Key Parameters

- **NUM_ANCILLA** ($m$): Number of ancilla qubits for phase precision (14-16 recommended)
  - Phase resolution: $\Delta\varphi = 2^{-m}$
  - Eigenvalue resolution: $\Delta\lambda = 2\pi/(t \cdot 2^m)$
  
- **TIMES** ($\{t_k\}$): Evolution times generated using Chebyshev nodes
  - Minimizes polynomial interpolation error
  - Typically 4-8 nodes for quadratic extrapolation
  
- **NUM_QDRIFT_SEGMENTS** ($N$): Number of qDRIFT segments per channel invocation
  - Controls qDRIFT error: $\varepsilon_{\text{qDrift}} \sim \lambda^2 t^2 / N$
  - Higher $N$ reduces error but increases circuit depth
  
- **RANDOM_CIRCUITS_PER_DATAPOINT** ($R$): Statistical sampling for each parameter point
  - Reduces variance in stochastic qDRIFT channel
  - Standard error scales as $1/\sqrt{R}$
  
- **SHOTS_PER_CIRCUIT** ($S$): Measurement shots per quantum circuit
  - QPE statistical error: $\varepsilon_{\text{stat}} \sim 1/\sqrt{S}$
  - Typical values: 1024-8192 shots

### Heisenberg Limit Scaling

The Heisenberg limit [11,12] provides the fundamental quantum limit for parameter estimation. For eigenvalue estimation with total query complexity $Q$ (number of oracle calls to $e^{-iHt}$), the best achievable precision scales as:

$$\Delta\lambda \geq \frac{1}{Q}$$

In our qDRIFT-QPE implementation:
- Query complexity: $Q \sim N \cdot (2^m - 1) \cdot R \cdot S$
- Each qDRIFT segment requires $N$ Hamiltonian term evolutions
- Each QPE circuit uses $(2^m - 1)$ controlled time evolution operations

Our extrapolation approach aims to approach this Heisenberg scaling by eliminating systematic errors that would otherwise dominate at high query counts.

## Performance Features

- **Function Caching**: Expensive operations cached for reuse across parameter sweeps
- **Memory Profiling**: Built-in memory usage tracking and optimization
- **Parallel Processing**: Multi-core parameter sweep execution
- **Vectorized Operations**: Numba-optimized numerical computations

## Examples and Tutorials

See the `notebooks/` directory for:
- `experiments/h_2_minimal_basis.ipynb`: H₂ molecule simulation and eigenvalue estimation
- `experiments/iterative_phase_estimation.ipynb`: Comparison of QFT-based and Kitaev's QPE [3]
- `numerical_tests/`: Algorithm validation and error analysis
- `demos/`: Basic usage tutorials

## Additional References

[7] Y. Dong, X. Meng, K. B. Whaley, and L. Lin, "Efficient phase-factor evaluation in quantum signal processing," Phys. Rev. A **103**, 042419 (2021).

[8] L. F. Richardson and J. A. Gaunt, "The Deferred Approach to the Limit," Phil. Trans. R. Soc. Lond. A **226**, 299-361 (1927).

[9] S. Endo, S. C. Benjamin, and Y. Li, "Practical Quantum Error Mitigation for Near-Future Applications," Phys. Rev. X **8**, 031027 (2018).

[10] D. W. Berry, A. M. Childs, R. Cleve, R. Kothari, and R. D. Somma, "Simulating Hamiltonian dynamics with a truncated Taylor series," Phys. Rev. Lett. **114**, 090502 (2015).

[11] V. Giovannetti, S. Lloyd, and L. Maccone, "Quantum Metrology," Phys. Rev. Lett. **96**, 010401 (2006).

[12] V. Giovannetti, S. Lloyd, and L. Maccone, "Advances in quantum metrology," Nature Photonics **5**, 222-229 (2011).

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qdrift_extrapolation,
  title = {qDRIFT Extrapolation: Zero-Noise Extrapolation for Quantum Phase Estimation},
  author = {Reyes, Rodrigo},
  year = {2025},
  url = {https://github.com/Rick0317/qDriftExtraPolation}
}
```


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
   