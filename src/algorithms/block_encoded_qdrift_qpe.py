from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
import numpy as np

# Define the Hamiltonian terms H_j (Pauli matrices for simplicity)
H_terms = [("X", 0.3), ("Y", 0.5), ("Z", 0.2)]  # Example: H = 0.3X + 0.5Y + 0.2Z
tau = 0.1  # Small time step
n_qpe = 3  # Number of ancilla qubits for QPE

# Quantum registers
phase_reg = QuantumRegister(n_qpe, name="phase")  # QPE ancillas
state_reg = QuantumRegister(1, name="state")  # Target state
sample_reg = QuantumRegister(2, name="sample")  # Sampling register (log2 of # terms)
classical_reg = ClassicalRegister(n_qpe, name="c_phase")  # Classical result storage

# Circuit initialization
qc = QuantumCircuit(phase_reg, state_reg, sample_reg, classical_reg)

# Step 1: Apply Hadamard to phase register for QPE superposition
qc.h(phase_reg)

# Step 2: Prepare the sampling superposition (PREP)
p_norm = np.sqrt([term[1] / sum(t[1] for t in H_terms) for term in H_terms])
qc.initialize(p_norm, sample_reg)  # PREP

# Step 3: Controlled Unitaries for QPE
for k in range(n_qpe):
    qc.barrier()
    qc.measure(sample_reg, [0, 1])  # Collapse to one Hamiltonian term
    qc.barrier()

    for idx, (pauli, coeff) in enumerate(H_terms):
        theta = tau * coeff * 2**k  # QPE scaling factor

        if pauli == "X":
            qc.cx(phase_reg[k], state_reg[0])  # Controlled-X
        elif pauli == "Y":
            qc.cy(phase_reg[k], state_reg[0])  # Controlled-Y
        elif pauli == "Z":
            qc.cz(phase_reg[k], state_reg[0])  # Controlled-Z

        qc.p(theta, state_reg[0])  # Phase rotation

    qc.barrier()

# Step 4: Apply Quantum Fourier Transform (QFT)
qc.append(QFT(n_qpe).inverse(), phase_reg)

# Step 5: Measure the phase register
qc.measure(phase_reg, classical_reg)

# Run the simulation
"""
backend = Aer.get_backend("qasm_simulator")
job = execute(qc, backend, shots=1024)
result = job.result()
counts = result.get_counts()


# Output results
print("Phase Estimation Results:", counts)
"""

# Draw circuit
qc.draw("mpl")
