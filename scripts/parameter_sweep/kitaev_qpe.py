from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PauliEvolutionGate
import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
import random


def get_kitaev_result(H, eigen_circuit, m, shots_per_estimation=1024):
    """
    Perform Kitaev's phase estimation algorithm.

    Args:
        H: The Hamiltonian (PauliSumOp).
        eigen_circuit: A circuit that prepares an eigenstate of H.
        m: The number of bits of precision for the phase estimation.
        shots_per_estimation: The number of shots for each circuit execution.

    Returns:
        float: The estimated phase.
    """
    simulator = AerSimulator()
    rhos = {}

    # Part 1: Estimate rho_j for j = 1 to m
    for j in range(1, m + 1):
        M = 2**(j - 1)

        # Estimate cos(M * lambda)
        qc_cos = generate_kitaev_qpe_circuit(H, M, eigen_circuit, theta=0)
        compiled_cos = transpile(qc_cos, simulator)
        result_cos = simulator.run(compiled_cos, shots=shots_per_estimation).result()
        counts_cos = result_cos.get_counts()
        cos_val = (counts_cos.get('0', 0) - counts_cos.get('1', 0)) / shots_per_estimation

        # Estimate sin(M * lambda) by measuring in a different basis
        qc_sin = generate_kitaev_qpe_circuit(H, M, eigen_circuit, theta=np.pi/2)
        compiled_sin = transpile(qc_sin, simulator)
        result_sin = simulator.run(compiled_sin, shots=shots_per_estimation).result()
        counts_sin = result_sin.get_counts()
        # This gives cos(M*lambda - pi/2) = sin(M*lambda)
        sin_val = (counts_sin.get('0', 0) - counts_sin.get('1', 0)) / shots_per_estimation

        M_lambda = np.arctan2(sin_val, cos_val)
        rhos[j] = (M_lambda / (2 * np.pi)) % 1.0

    # Part 2: Classical post-processing to determine phase bits
    alphas = np.zeros(m + 3, dtype=int)

    # Step 4: Determine alpha_m, alpha_{m+1}, alpha_{m+2}
    rho_m = rhos[m]
    k = int(round(8 * rho_m)) % 8
    alphas[m] = (k >> 2) & 1
    alphas[m+1] = (k >> 1) & 1
    alphas[m+2] = k & 1

    # Steps 5-7: Infer alpha_j for j = m-1 down to 1
    for j in range(m - 1, 0, -1):
        rho_j = rhos[j]

        # Estimate based on 0
        val0 = (alphas[j+1] / 4.0 + alphas[j+2] / 8.0)
        # Estimate based on 1
        val1 = (1.0 / 2.0 + alphas[j+1] / 4.0 + alphas[j+2] / 8.0)

        dist0 = min(abs(val0 - rho_j), 1 - abs(val0 - rho_j))
        dist1 = min(abs(val1 - rho_j), 1 - abs(val1 - rho_j))

        if dist0 < dist1:
            alphas[j] = 0
        else:
            alphas[j] = 1

    # Step 8: Combine bits to form the phase
    phase = 0.0
    for i in range(1, m + 3):
        phase += alphas[i] / (2**i)

    return phase



def get_qDrift_kitaev_result(hamilotnian, eigen_circuit, m, qDrift_invocation):
    simulator = AerSimulator()
    rhos = {}

    # Part 1: Estimate rho_j for j = 1 to m
    for j in range(1, m + 1):
        M = 2 ** (j - 1)

        # Create all circuits upfront for parallel execution
        cos_circuits = []
        sin_circuits = []

        for _ in range(qDrift_invocation):
            H = qDrift_sample(hamilotnian)
            cos_circuits.append(generate_kitaev_qpe_circuit(H, M, eigen_circuit, theta=0))

            H = qDrift_sample(hamilotnian)  # Sample again for sin estimation
            sin_circuits.append(generate_kitaev_qpe_circuit(H, M, eigen_circuit, theta=np.pi/2))

        # Compile all circuits
        all_circuits = cos_circuits + sin_circuits
        compiled_circuits = transpile(all_circuits, simulator)

        # Run all circuits in parallel
        results = simulator.run(compiled_circuits, shots=1).result()

        # Process cos results
        cos_value = 0
        for i in range(qDrift_invocation):
            counts = results.get_counts(i)
            cos_value += counts.get('0', 0) - counts.get('1', 0)
        cos_val = cos_value / qDrift_invocation

        # Process sin results
        sin_value = 0
        for i in range(qDrift_invocation, 2 * qDrift_invocation):
            counts = results.get_counts(i)
            sin_value += counts.get('0', 0) - counts.get('1', 0)
        sin_val = sin_value / qDrift_invocation

        M_lambda = np.arctan2(sin_val, cos_val)
        rhos[j] = (M_lambda / (2 * np.pi)) % 1.0

    # Part 2: Classical post-processing to determine phase bits
    alphas = np.zeros(m + 3, dtype=int)

    # Step 4: Determine alpha_m, alpha_{m+1}, alpha_{m+2}
    rho_m = rhos[m]
    k = int(round(8 * rho_m)) % 8
    alphas[m] = (k >> 2) & 1
    alphas[m + 1] = (k >> 1) & 1
    alphas[m + 2] = k & 1

    # Steps 5-7: Infer alpha_j for j = m-1 down to 1
    for j in range(m - 1, 0, -1):
        rho_j = rhos[j]

        # Estimate based on 0
        val0 = (alphas[j + 1] / 4.0 + alphas[j + 2] / 8.0)
        # Estimate based on 1
        val1 = (1.0 / 2.0 + alphas[j + 1] / 4.0 + alphas[j + 2] / 8.0)

        dist0 = min(abs(val0 - rho_j), 1 - abs(val0 - rho_j))
        dist1 = min(abs(val1 - rho_j), 1 - abs(val1 - rho_j))

        if dist0 < dist1:
            alphas[j] = 0
        else:
            alphas[j] = 1

    # Step 8: Combine bits to form the phase
    phase = 0.0
    for i in range(1, m + 3):
        phase += alphas[i] / (2 ** i)

    return phase

def generate_kitaev_qpe_circuit(H, M, eigen_circuit, theta):
    """
    Generates a single circuit for Kitaev's QPE algorithm.
    """
    num_state_qubits = eigen_circuit.num_qubits

    state_reg = QuantumRegister(num_state_qubits, name='state')
    ancilla_reg = QuantumRegister(1, name='ancilla')
    classical_reg = ClassicalRegister(1, name='c')

    qc = QuantumCircuit(ancilla_reg, state_reg, classical_reg)

    # Prepare the eigenvector on the state register
    qc.append(eigen_circuit.to_instruction(), state_reg)

    # Start of the Kitaev QPE ancilla operations
    qc.h(ancilla_reg)

    # Phase correction Z(-theta). In qiskit, P(lambda) is the phase gate [[1, 0], [0, e^(i*lambda)]].
    qc.p(-theta, ancilla_reg)

    # Controlled Unitary
    # U is defined as e^(iH), which with an evolution time M gives e^(iHM).
    # Qiskit's PauliEvolutionGate implements e^(-iHt), so we use time = -M.
    evolution_op = PauliEvolutionGate(H, time=-M)
    controlled_U = evolution_op.control(1)
    qc.append(controlled_U, [ancilla_reg[0]] + list(state_reg))


    # Final Hadamard and measurement
    qc.h(ancilla_reg)
    qc.measure(ancilla_reg, classical_reg)

    return qc




def qDrift_sample(hamiltonian: SparsePauliOp):
        # First, simplify the Hamiltonian to combine duplicate terms
        # This is important for Ising Hamiltonians with periodic boundary conditions
        simplified_hamiltonian = hamiltonian.simplify()

        # Extract coefficients and Pauli strings from simplified Hamiltonian
        coeffs_absolute_values = np.abs(simplified_hamiltonian.coeffs)
        paulis = simplified_hamiltonian.paulis.to_labels()


        # Calculate lambda and tau
        lam = np.sum(coeffs_absolute_values)

        # Define sampling distribution
        pmf = coeffs_absolute_values / lam

        sampled_unitaries = []
        labels = []

        # Sample based on the distribution
        idx = random.choices(population=range(len(paulis)), weights=pmf, k=1)[0]
        pauli_string = paulis[idx]
        original_coeff = simplified_hamiltonian.coeffs[idx]

        # H_j should be normalized with coefficient ±1.0 based on sign of original coefficient
        normalized_coeff = 1.0 if original_coeff >= 0 else -1.0
        h_j = SparsePauliOp([pauli_string], [normalized_coeff])
        return h_j


def generate_kitaev_qpe_circuit_qdrift(hamiltonian, M, eigen_circuit, N, theta):
    """
    Generates a single circuit for Kitaev's QPE algorithm using qDrift approximation.

    Instead of using the full Hamiltonian H, this function:
    1. Samples num_qdrift_samples Pauli terms from the Hamiltonian
    2. Composes their exponentials as a product: U ≈ e^{iH₁τ} · e^{iH₂τ} · ... · e^{iHₙτ}
    3. Uses this product as the controlled unitary in QPE

    Args:
        hamiltonian: The full Hamiltonian (SparsePauliOp)
        M: Evolution time parameter (2^(j-1) for layer j)
        eigen_circuit: Circuit that prepares the eigenstate
        theta: Phase correction parameter
        num_qdrift_samples: Number of qDrift samples to use for approximation

    Returns:
        QuantumCircuit: QPE circuit with qDrift approximation
    """
    num_state_qubits = eigen_circuit.num_qubits

    # Get lambda (sum of absolute coefficients) for proper qDrift scaling
    simplified_hamiltonian = hamiltonian.simplify()
    lam = float(np.sum(np.abs(simplified_hamiltonian.coeffs)))


    state_reg = QuantumRegister(num_state_qubits, name='state')
    ancilla_reg = QuantumRegister(1, name='ancilla')
    classical_reg = ClassicalRegister(1, name='c')

    qc = QuantumCircuit(ancilla_reg, state_reg, classical_reg)

    # Prepare the eigenvector on the state register
    qc.append(eigen_circuit.to_instruction(), state_reg)

    # Start of the Kitaev QPE ancilla operations
    qc.h(ancilla_reg)

    # Phase correction Z(-theta)
    qc.p(-theta, ancilla_reg)

    # qDrift approximation: Apply sequence of controlled exponentials
    # U^M ≈ e^{iH₁τ} · e^{iH₂τ} · ... · e^{iHₙτ}
    for _ in range(M):
        for _ in range(N):
            # Sample a Pauli term from the Hamiltonian
            sampled_H = qDrift_sample(hamiltonian)

            # Create controlled evolution for this sampled term
            # Note: We use -tau because PauliEvolutionGate implements e^(-iHt)
            evolution_op = PauliEvolutionGate(sampled_H, time= -lam / N)
            controlled_U_sample = evolution_op.control(1)

            # Apply the controlled evolution
            qc.append(controlled_U_sample, [ancilla_reg[0]] + list(state_reg))

    # Final Hadamard and measurement
    qc.h(ancilla_reg)
    qc.measure(ancilla_reg, classical_reg)

    return qc



def generate_kitaev_qpe_circuit_qdrift_v2(hamiltonian, M, eigen_circuit, N, theta):
    """
    Generates a single circuit for Kitaev's QPE algorithm using qDrift approximation.

    Instead of using the full Hamiltonian H, this function:
    1. Samples num_qdrift_samples Pauli terms from the Hamiltonian
    2. Composes their exponentials as a product: U ≈ e^{iH₁τ} · e^{iH₂τ} · ... · e^{iHₙτ}
    3. Uses this product as the controlled unitary in QPE

    Args:
        hamiltonian: The full Hamiltonian (SparsePauliOp)
        M: Evolution time parameter (2^(j-1) for layer j)
        eigen_circuit: Circuit that prepares the eigenstate
        theta: Phase correction parameter
        num_qdrift_samples: Number of qDrift samples to use for approximation

    Returns:
        QuantumCircuit: QPE circuit with qDrift approximation
    """
    num_state_qubits = eigen_circuit.num_qubits

    # Get lambda (sum of absolute coefficients) for proper qDrift scaling
    simplified_hamiltonian = hamiltonian.simplify()
    lam = float(np.sum(np.abs(simplified_hamiltonian.coeffs)))


    state_reg = QuantumRegister(num_state_qubits, name='state')
    ancilla_reg = QuantumRegister(1, name='ancilla')
    classical_reg = ClassicalRegister(1, name='c')

    qc = QuantumCircuit(ancilla_reg, state_reg, classical_reg)

    # Prepare the eigenvector on the state register
    qc.append(eigen_circuit.to_instruction(), state_reg)

    # Start of the Kitaev QPE ancilla operations
    qc.h(ancilla_reg)

    # Phase correction Z(-theta)
    qc.p(-theta, ancilla_reg)

    # qDrift approximation: Apply sequence of controlled exponentials
    # U^M ≈ e^{iH₁τ} · e^{iH₂τ} · ... · e^{iHₙτ}
    for _ in range(M):
        # Sample a Pauli term from the Hamiltonian
        sampled_H = qDrift_sample(hamiltonian)

        # Create controlled evolution for this sampled term
        # Note: We use -tau because PauliEvolutionGate implements e^(-iHt)
        evolution_op = PauliEvolutionGate(sampled_H, time= -lam / N)
        controlled_U_sample = evolution_op.control(1)

        # Apply the controlled evolution
        qc.append(controlled_U_sample, [ancilla_reg[0]] + list(state_reg))

    # Final Hadamard and measurement
    qc.h(ancilla_reg)
    qc.measure(ancilla_reg, classical_reg)

    return qc



def get_qDrift_kitaev_result_v2(hamiltonian, eigen_circuit, m, qDrift_invocation, shots_per_estimation=1024):
    """
    Perform Kitaev's phase estimation algorithm using proper qDrift approximation.

    This version uses the correct qDrift approach where each circuit contains
    a product of exponentials: U^M ≈ e^{iH₁τ} · e^{iH₂τ} · ... · e^{iHₙτ}

    Args:
        hamiltonian: The full Hamiltonian (SparsePauliOp)
        eigen_circuit: A circuit that prepares an eigenstate of H
        m: The number of bits of precision for the phase estimation
        num_qdrift_samples: Number of qDrift samples per circuit
        shots_per_estimation: The number of shots for each circuit execution

    Returns:
        float: The estimated phase
    """
    simulator = AerSimulator()
    rhos = {}

    # Part 1: Estimate rho_j for j = 1 to m
    num_qdrift_circuits = 100  # Number of different random qDrift circuits to average over

    for j in range(1, m + 1):
        M = 2**(j - 1)

        cos_val = 0
        for _ in range(num_qdrift_circuits):
            # Generate a NEW random qDrift circuit (randomness from qDrift_sample in generate_kitaev_qpe_circuit_qdrift)
            qc_cos = generate_kitaev_qpe_circuit_qdrift(hamiltonian, M, eigen_circuit, qDrift_invocation, theta=0)
            compiled_cos = transpile(qc_cos, simulator)
            result_cos = simulator.run(compiled_cos, shots=1).result()  # 1 shot per circuit
            counts_cos = result_cos.get_counts()
            cosine_val = (counts_cos.get('0', 0) - counts_cos.get('1', 0))
            cos_val += cosine_val

        cos_val /= num_qdrift_circuits

        sin_val = 0
        for _ in range(num_qdrift_circuits):
            # Generate a NEW random qDrift circuit
            qc_sin = generate_kitaev_qpe_circuit_qdrift(hamiltonian, M, eigen_circuit, qDrift_invocation, theta=np.pi/2)
            compiled_sin = transpile(qc_sin, simulator)
            result_sin = simulator.run(compiled_sin, shots=1).result()  # 1 shot per circuit
            counts_sin = result_sin.get_counts()
            sine_val = (counts_sin.get('0', 0) - counts_sin.get('1', 0))
            sin_val += sine_val

        sin_val /= num_qdrift_circuits

        M_lambda = np.arctan2(sin_val, cos_val)
        rhos[j] = (M_lambda / (2 * np.pi)) % 1.0

    # Part 2: Classical post-processing to determine phase bits
    alphas = np.zeros(m + 3, dtype=int)

    # Step 4: Determine alpha_m, alpha_{m+1}, alpha_{m+2}
    rho_m = rhos[m]
    k = int(round(8 * rho_m)) % 8
    alphas[m] = (k >> 2) & 1
    alphas[m+1] = (k >> 1) & 1
    alphas[m+2] = k & 1

    # Steps 5-7: Infer alpha_j for j = m-1 down to 1
    for j in range(m - 1, 0, -1):
        rho_j = rhos[j]

        # Estimate based on 0
        val0 = (alphas[j+1] / 4.0 + alphas[j+2] / 8.0)
        # Estimate based on 1
        val1 = (1.0 / 2.0 + alphas[j+1] / 4.0 + alphas[j+2] / 8.0)

        dist0 = min(abs(val0 - rho_j), 1 - abs(val0 - rho_j))
        dist1 = min(abs(val1 - rho_j), 1 - abs(val1 - rho_j))

        if dist0 < dist1:
            alphas[j] = 0
        else:
            alphas[j] = 1

    # Step 8: Combine bits to form the phase
    phase = 0.0
    for i in range(1, m + 3):
        phase += alphas[i] / (2**i)

    return phase


def get_qDrift_continuous_phase_estimate(hamiltonian, eigen_circuit, qDrift_invocation,
                                          M=1, num_qdrift_circuits=100):
    """
    Estimate the continuous phase (before Kitaev discretization) using qDrift.

    This directly estimates cos(M*λ) and sin(M*λ) to get M*λ, avoiding
    the discrete binning of Kitaev QPE. Suitable for extrapolation.

    Args:
        hamiltonian: The Hamiltonian (SparsePauliOp)
        eigen_circuit: Circuit that prepares an eigenstate of H
        qDrift_invocation: Number of qDrift samples (N)
        M: Evolution time multiplier (default=1 for single estimation)
        num_qdrift_circuits: Number of random qDrift circuits to average

    Returns:
        float: Continuous phase estimate (M*λ)/(2π) mod 1
    """
    simulator = AerSimulator()

    # Estimate cos(M * lambda)
    cos_val = 0
    for _ in range(num_qdrift_circuits):
        qc_cos = generate_kitaev_qpe_circuit_qdrift(
            hamiltonian, M, eigen_circuit, qDrift_invocation, theta=0
        )
        compiled_cos = transpile(qc_cos, simulator)
        result_cos = simulator.run(compiled_cos, shots=1).result()
        counts_cos = result_cos.get_counts()
        cosine_val = (counts_cos.get('0', 0) - counts_cos.get('1', 0))
        cos_val += cosine_val

    cos_val /= num_qdrift_circuits

    # Estimate sin(M * lambda)
    sin_val = 0
    for _ in range(num_qdrift_circuits):
        qc_sin = generate_kitaev_qpe_circuit_qdrift(
            hamiltonian, M, eigen_circuit, qDrift_invocation, theta=np.pi/2
        )
        compiled_sin = transpile(qc_sin, simulator)
        result_sin = simulator.run(compiled_sin, shots=1).result()
        counts_sin = result_sin.get_counts()
        sine_val = (counts_sin.get('0', 0) - counts_sin.get('1', 0))
        sin_val += sine_val

    sin_val /= num_qdrift_circuits

    # Compute continuous phase estimate
    M_lambda = np.arctan2(sin_val, cos_val)
    phase = (M_lambda / (2 * np.pi)) % 1.0

    return phase
