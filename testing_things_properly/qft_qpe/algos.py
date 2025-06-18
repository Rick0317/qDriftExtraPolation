import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Union, Optional
import scipy
import random
from functools import reduce
from tqdm import tqdm

from qiskit import transpile
from qiskit_aer import AerSimulator  # as of 25Mar2025
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, UnitaryGate, PhaseGate, RZGate
from qiskit.quantum_info import Operator
from sympy import Matrix, latex
from IPython.display import display, Math
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.circuit.library import Initialize
from line_profiler import profile

# import basic plot tools
from qiskit.visualization import plot_histogram

# Type aliases
coefficient = float
Hamiltonian = list[tuple[coefficient, Pauli]]




# ---------------------------------------------------- utils ----------------------------------------------------

@profile
def generate_ising_hamiltonian(num_qubits: int, J, g) -> SparsePauliOp:
    z_terms = []
    z_coeffs = []
    
    # ZZ interaction terms
    for j in range(num_qubits):
        pauli_string = ['I'] * num_qubits
        pauli_string[j] = 'Z'
        pauli_string[(j + 1) % num_qubits] = 'Z'  # Periodic boundary conditions
        z_terms.append("".join(pauli_string))
        z_coeffs.append(-J)  # Coefficient for ZZ interaction

    x_terms = []
    x_coeffs = []
    
    # X field terms
    for j in range(num_qubits):
        pauli_string = ['I'] * num_qubits
        pauli_string[j] = 'X'
        x_terms.append("".join(pauli_string))
        x_coeffs.append(-g)  # Coefficient for X term

    # Combine the Z and X terms into a single Hamiltonian
    all_terms = z_terms + x_terms
    all_coeffs = z_coeffs + x_coeffs

    return SparsePauliOp(all_terms, coeffs=all_coeffs)


def exponentiate_hamiltonian(hamiltonian: SparsePauliOp, time: float) -> Operator:
    """Exponentiates the Hamiltonian to obtain U = e^(-i H t)."""
    matrix = hamiltonian.to_matrix()
    unitary_matrix = scipy.linalg.expm(1j * time * matrix)
    return Operator(unitary_matrix)


def calculate_ground_state_and_energy(H: SparsePauliOp) -> List[complex]:
    """Calculates the eigenvalues of the Hamiltonian."""
    matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # Sort eigenvalues and eigenvectors
    ground_state = eigenvectors[:, np.argmin(eigenvalues)]
    ground_energy = np.min(eigenvalues)
    return ground_state, ground_energy

@profile
def prepare_eigenstate_circuit(ground_state: np.ndarray) -> QuantumCircuit:
    """
    Prepare a quantum circuit that initializes the ground state.
    
    Args:
        ground_state (np.ndarray): State vector representing the ground state.

    Returns:
        QuantumCircuit: Circuit that prepares the ground state.
    """
    num_qubits = int(np.log2(len(ground_state)))
    if 2 ** num_qubits != len(ground_state):
        raise ValueError("The length of the state vector must be a power of 2.")

    # Normalize the state vector
    ground_state = ground_state / np.linalg.norm(ground_state)

    # Initialize circuit
    qc = QuantumCircuit(num_qubits)
    init_gate = Initialize(ground_state)
    qc.append(init_gate, range(num_qubits))

    # Optional: Simplify the circuit using transpile
    # from qiskit import transpile
    # qc = transpile(qc, optimization_level=3)

    qc.barrier()
    return qc

def generate_random_hamiltonian_with_pauli_tensor_structure(num_qubits: int, num_terms: int) -> SparsePauliOp:
    """
    Generates a random Hamiltonian with a specified number of Pauli terms.
    
    Args:
        num_qubits (int): Number of qubits in the Hamiltonian.
        num_terms (int): Number of Pauli terms in the Hamiltonian.

    Returns:
        SparsePauliOp: Random Hamiltonian with Pauli tensor structure.
    """
    paulis = []
    coeffs = []

    while len(paulis) < num_terms:
        # Generate a random Pauli string (I, X, Y, Z) of length num_qubits
        term = ''.join(np.random.choice(['I', 'X', 'Z'], size=num_qubits))
        if term not in paulis:
            paulis.append(term)
            coeffs.append(np.random.uniform(-1, 1))  # real coefficient

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))

# -------------------------------------------------------------------------- Non-stochastic QPE -----------------------------
def generate_qpe_circuit_simple(total_qubits, phase):
    """
    Assumptions: 
    - The target unitary acts on *one* qubit (the last one)
    - The target unitary is a phase gate:
        P(\theta) =
                \begin{pmatrix}
                    1 & 0 \\
                    0 & e^{i\theta}
                \end{pmatrix}
    """
    num_ancilla = total_qubits-1
    qpe = QuantumCircuit(total_qubits, num_ancilla) # num qubits, num classical bits (to store meaurements)
    qpe.x(num_ancilla) # because ket(1) is an eigenvector of the phase gate

    for qubit in range(num_ancilla):
        qpe.h(qubit)
        
    repetitions = 1
    for counting_qubit in range(num_ancilla):
        for i in range(repetitions):
            qpe.cp(phase, counting_qubit, num_ancilla); # Apply C-PhaseGate to last qubit (target qubit) controlled by counting qubit
        repetitions *= 2
        
    # Apply the inverse QFT
    list_of_ancilla_qubits = [i for i in range(num_ancilla)]
    qpe.append(QFT(3, inverse=True), list_of_ancilla_qubits) 

    qpe.measure(list_of_ancilla_qubits, list_of_ancilla_qubits) # Measure the ancilla qubits
    return qpe

@profile
def standard_qpe(unitary: Operator, eigenstate: QuantumCircuit, num_ancilla: int) -> QuantumCircuit:
    """Constructs a standard Quantum Phase Estimation (QPE) circuit using repeated controlled-U applications."""
    num_qubits = unitary.num_qubits
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    # Prepare eigenstate on system qubits
    qc.append(eigenstate, range(num_ancilla, num_ancilla + num_qubits))

    # Apply Hadamard gates to ancilla qubits
    qc.h(range(num_ancilla))

    # Apply controlled-U^(2^k) using repeated controlled applications of U
    for k in tqdm(range(num_ancilla), desc="Applying controlled-U powers"):
        controlled_U = UnitaryGate(unitary.data).control(1, label=f"U")
        
        # Apply controlled-U 2^k times
        for _ in range(2**k):  
            qc.append(controlled_U, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))

    # Apply inverse QFT on ancilla qubits
    qc.append(QFT(num_ancilla, inverse=True, do_swaps=True), range(num_ancilla))

    # Measure ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))

    return qc

# ---------------------------------------------------- qDRIFT-QPE ----------------------------------------------------
def qdrift_sample_naive(hamiltonian: SparsePauliOp, time: float, num_samples: int) -> Tuple[List[SparsePauliOp], List[str]]:
    # Extract coefficients and Pauli strings
    coeffs_absolute_values = np.abs(hamiltonian.coeffs)
    paulis = hamiltonian.paulis.to_labels()
    
    # Calculate lambda and tau
    lam = np.sum(coeffs_absolute_values)
    tau = time * lam / num_samples
    
    # Define sampling distribution
    pmf = coeffs_absolute_values / lam
    
    sampled_unitaries = []
    labels = []
    
    # Sample based on the distribution
    for _ in range(num_samples):
        idx = random.choices(population=range(len(paulis)), weights=pmf, k=1)[0]
        pauli_string = paulis[idx]
        original_coeff = hamiltonian.coeffs[idx]

        if original_coeff < 0:
            h_j = SparsePauliOp([pauli_string], [-1.0])
        else:
            h_j = SparsePauliOp([pauli_string], [1.0])

        unitary = exponentiate_hamiltonian(h_j, tau)

        sampled_unitaries.append(unitary)
        
        # Label for visualization
        labels.append(f"$e^{{i \\tau {pauli_string}}}$")
    
    return sampled_unitaries, labels

# Function to perform qDRIFT-based QPE
def qdrift_qpe(hamiltonian: SparsePauliOp, time: float, eigenstate, num_qubits: int, num_ancilla: int):
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    # Initialize the eigenstate
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circuit = QuantumCircuit(num_qubits, name='Eigenstate')
        eigenstate_circuit.initialize(eigenstate)
    else:
        eigenstate_circuit = eigenstate

    qc.append(eigenstate_circuit, range(num_ancilla, num_ancilla + num_qubits))
    
    # Apply QFT to ancilla qubits
    qc.append(QFT(num_ancilla), range(num_ancilla))
    k = 0
    # Controlled qDRIFT unitaries
    for k in tqdm(range(num_ancilla), desc=f"Ancilla layer (k={k})"):
        for _ in range(2 ** k):
            # Sample unitaries using the new qdrift_sample function
            sampled_unitaries, labels = qdrift_sample_naive(hamiltonian, time, num_samples=1)
            for unitary, label in zip(sampled_unitaries, labels):
                controlled_unitary = UnitaryGate(unitary, label=label).control(1)
                qc.append(controlled_unitary, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))
    
    # Apply inverse QFT
    qc.append(QFT(num_ancilla, inverse=True), range(num_ancilla))
    
    # Measure the ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))
    
    return qc


def qdrift_qpe_chat_gpts_take(hamiltonian: SparsePauliOp, time: float, eigenstate, num_qubits: int, num_ancilla: int):
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    # Initialize the eigenstate
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circuit = QuantumCircuit(num_qubits, name='Eigenstate')
        eigenstate_circuit.initialize(eigenstate)
    else:
        eigenstate_circuit = eigenstate

    qc.append(eigenstate_circuit, range(num_ancilla, num_ancilla + num_qubits))
    
    # Apply QFT to ancilla qubits
    qc.append(QFT(num_ancilla), range(num_ancilla))
    
    # Loop over ancilla bits
    for k in range(num_ancilla):
        # Sample 2^k QDrift steps once
        total_time = (2 ** k) * time
        sampled_unitaries, labels = qdrift_sample_naive(hamiltonian, total_time, num_samples=2**k)  # Tune num_samples

        # Compose into one unitary
        full_unitary = Operator(np.eye(2 ** num_qubits, dtype=complex))
        for u in sampled_unitaries:
            full_unitary = u @ full_unitary  # Compose left-to-right

        controlled_U = UnitaryGate(full_unitary, label=f"QDrift^{2**k}").control(1)
        qc.append(controlled_U, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))

    
    # Apply inverse QFT
    qc.append(QFT(num_ancilla, inverse=True), range(num_ancilla))
    
    # Measure the ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))
    
    return qc


def qdrift_qpe_extra_random(hamiltonian: SparsePauliOp, 
                            eigenstate:Union[QuantumCircuit, np.array], 
                            num_qubits: int, num_ancilla: int, 
                            total_simulation_time, 
                            num_samples: int = 1000000) -> QuantumCircuit:
    
    qc = QuantumCircuit(num_ancilla + num_qubits, num_ancilla)

    # Initialize the eigenstate
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circuit = QuantumCircuit(num_qubits, name='Eigenstate')
        eigenstate_circuit.initialize(eigenstate)
    else:
        eigenstate_circuit = eigenstate
    qc.append(eigenstate_circuit, range(num_ancilla, num_ancilla + num_qubits))
    
    # Apply QFT to ancilla qubits
    qc.append(QFT(num_ancilla), range(num_ancilla))

    # random sampling of unitaries
    lam = np.sum(np.abs(hamiltonian.coeffs))
    tau = total_simulation_time * lam / num_samples
    direct_access_table_of_unitaries = [] # this is so we don't have to recompute the same matrix exponential multiple times

    # Note: the direct_access_table optimization was not incorporated before bcause a) I didn't think of it 
    # and b) if we only sample a few unitaries (or just one), the overhead of computing the matrix exponentials is not that high.

    for coeff, pauli in zip(hamiltonian.coeffs, hamiltonian.paulis.to_labels()):
        if coeff < 0:
            h_j = SparsePauliOp([pauli], [-1.0])
        else:
            h_j = SparsePauliOp([pauli], [1.0])
        unitary = exponentiate_hamiltonian(h_j, tau)
        direct_access_table_of_unitaries.append(unitary)
    
    qdrift_pmf = np.abs(hamiltonian.coeffs) / lam
    for k in tqdm(range(num_ancilla), desc="Applying qDRIFT unitaries"):
        for _ in range(2 ** k):
            idx = random.choices(population=range(len(direct_access_table_of_unitaries)), weights=qdrift_pmf, k=1)[0]
            unitary = direct_access_table_of_unitaries[idx]
            controlled_unitary = UnitaryGate(unitary, label=f"QDrift_{k}").control(1)
            qc.append(controlled_unitary, [k] + list(range(num_ancilla, num_ancilla + num_qubits)))
    
    # Apply inverse QFT
    qc.append(QFT(num_ancilla, inverse=True), range(num_ancilla))

    # Measure the ancilla qubits
    qc.measure(range(num_ancilla), range(num_ancilla))

    return qc


def compute_qdrift_Nj_list(
    hamiltonian: SparsePauliOp,
    epsilon_total: float,
    m: int
) -> List[int]:
    """
    Compute the number of qDRIFT steps N_j for each controlled‐U^{2^j} layer,
    following Appendix E of the qDRIFT paper.  We allocate the diamond‐norm error
    budget equally:
       ε_j = ε_total / m,  for j=0,1,...,m-1.
    Then from Eq. (E5) one has
       N_j = ceil( (2^j)^2 / ε_j )  =  ceil( 4^j * (m / ε_total) ).
    Args:
        hamiltonian (SparsePauliOp):  H = ∑_k c_k P_k
        epsilon_total (float):  total diamond‐norm error budget (⌘_tot).
        m (int):  number of ancilla qubits (= number of controlled‐U layers).
    Returns:
        List[int]:  [N_0, N_1, ..., N_{m-1}].
    """
    if m <= 0:
        raise ValueError("Number of ancilla (m) must be ≥ 1.")
    if epsilon_total <= 0:
        raise ValueError("epsilon_total must be > 0.")
    N_list = []
    # Because λ = ∑ |c_k| only enters into choosing τ_j = (2^j) * (λ / N_j),
    # but cancels in the ratio (2^j)^2 / ε_j, we do not need λ explicitly here.
    #
    # We simply do: ε_j = ε_total/m, then N_j = ceil( (2^j)^2 / ε_j ) = ceil( 4^j * m / ε_total ).
    for j in range(m):
        Nj = math.ceil((4**j) * (m / epsilon_total))
        N_list.append(Nj)
    return N_list


@profile
def generate_fixed_qdrift_unitary_for_layer(
    hamiltonian: SparsePauliOp,
    layer_index: int,
    Nj: int
) -> Operator:
    """
    Given H, for the layer j (which corresponds to U^{2^j}):
      - We set t_j = 2^j.
      - We take N_j samples from {P_k} with probability |c_k|/λ, and each step evolves
        for τ_j = (2^j) * (λ / N_j).
      - We compose those N_j single‐term exponentials into a single Operator U_j ≈ e^{i H * 2^j}.
    Args:
        hamiltonian (SparsePauliOp):  H = ∑_k c_k P_k
        layer_index (int):   j (0‐based), meaning a target evolution time t_j = 2^j.
        Nj (int):            number of qDRIFT mini‐steps at layer j.
    Returns:
        Operator:  U_j ≈ exp(i H * 2^j) from a single fixed qDRIFT sampling.
    """
    num_qubits = hamiltonian.num_qubits
    # Extract |c_k| and Pauli labels
    coeffs = np.abs(hamiltonian.coeffs)
    pauli_labels = hamiltonian.paulis.to_labels()

    lam = float(np.sum(coeffs))            # λ = ∑_k |c_k|
    t_j = 2 ** layer_index                  # desired total time for this layer

    # Each mini‐step runs for τ_j := t_j * λ / N_j
    tau_j = t_j * lam / Nj

    # Build the pmf over terms {0,...,L-1}
    pmf = coeffs / lam

    # Precompute all single‐term unitaries e^{i τ_j * sign(c_k) P_k}
    single_term_table: List[Operator] = []
    for idx, label in enumerate(pauli_labels):
        sign = np.sign(float(hamiltonian.coeffs[idx]))  # ±1 for P_k
        h_j = SparsePauliOp([label], [sign])
        U_jk = exponentiate_hamiltonian(h_j, tau_j)
        single_term_table.append(U_jk)

    # Sample **exactly Nj times** up front
    sampled_indices = random.choices(population=range(len(pauli_labels)), weights=pmf, k=Nj)

    # Compose U_j = U_{i_{N_j}} ⋯ U_{i_2} ⋅ U_{i_1}
    U_layer = Operator(np.eye(2**num_qubits, dtype=complex))
    for i in sampled_indices:
        U_layer = single_term_table[i] @ U_layer

    return U_layer


# -------------------------------------------------------------------------- QPE builder ------------------------------------------------------------

@profile
def deterministic_qpe_qdrift_with_error_budget(
    hamiltonian: SparsePauliOp,
    epsilon_total: float,
    m: int,
    eigenstate: Union[QuantumCircuit, np.ndarray],
    num_qubits: int
) -> QuantumCircuit:
    """
    Build a deterministic‐qDRIFT + QPE circuit **using per‐layer N_j** from Appendix E.

    Steps:
      1. Compute [N_0, N_1, ..., N_{m-1}] via `compute_qdrift_Nj_list(hamiltonian, ε_total, m)`.
      2. For each j in 0..(m-1):
         a) Build a **fixed** qDRIFT approximation U_j ≈ exp(i H * 2^j) by sampling N_j terms.
         b) Promote U_j to a controlled‐unitary in the j-th ancilla slot.
      3. Wrap everything in a standard QPE pattern:
         - Prepare ansatz |ψ⟩ on the last `num_qubits`.
         - Apply H‐gates on all m ancilla qubits.
         - For each j = 0..m-1, apply controlled‐U_j (with control = ancilla j, target = system).
         - Inverse QFT on ancilla.
         - Measure ancilla.

    Args:
        hamiltonian (SparsePauliOp):  target H = ∑ c_k P_k.
        epsilon_total (float):  total diamond‐norm error budget (ε_tot).
        m (int):  number of ancilla qubits (bits of precision).
        eigenstate (QuantumCircuit or np.ndarray):  circuit or statevector for |ψ⟩.
        num_qubits (int):  number of system qubits.

    Returns:
        QuantumCircuit:  the full QPE circuit with deterministic qDRIFT.
    """
    # 1) Compute per‐layer sample counts N_j
    Nj_list = compute_qdrift_Nj_list(hamiltonian, epsilon_total, m)

    # 2) Build one fixed qDRIFT unitary U_j per layer
    U_fixed_list: List[Operator] = []
    for j in range(m):
        Uj = generate_fixed_qdrift_unitary_for_layer(hamiltonian, layer_index=j, Nj=Nj_list[j])
        U_fixed_list.append(Uj)

    # 3) Build the QPE circuit
    qc = QuantumCircuit(m + num_qubits, m)

    # 3a) Initialize eigenstate on the last num_qubits
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circ = QuantumCircuit(num_qubits, name="Eigenstate")
        eigenstate_circ.initialize(eigenstate)
    else:
        eigenstate_circ = eigenstate
    qc.append(eigenstate_circ, range(m, m + num_qubits))

    # 3b) Apply Hadamards to all m ancilla
    qc.h(range(m))

    # 3c) For each ancilla bit j, apply controlled‐U_j
    for j in range(m):
        controlled_Uj = UnitaryGate(U_fixed_list[j].data, label=f"U_qd_j={j}").control(1)
        ctrl_q = j
        tgt_qubits = list(range(m, m + num_qubits))
        qc.append(controlled_Uj, [ctrl_q] + tgt_qubits)

    # 3d) Inverse QFT on ancilla
    qc.append(QFT(m, inverse=True, do_swaps=True), range(m))

    # 3e) Measure ancilla
    qc.measure(range(m), range(m))

    return qc






