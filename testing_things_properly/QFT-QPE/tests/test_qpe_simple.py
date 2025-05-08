import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from algos import generate_qpe_circuit_simple

def test_simple_qpe_phase_pi_4():
    total_qubits = 4
    phase = np.pi / 4  # θ = π/4 → binary ≈ 0.001
    shots = 1024
    expected_bin = '001'

    qc = generate_qpe_circuit_simple(total_qubits, phase)
    
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=1024)
    result = job.result()
    counts = result.get_counts(qc)
    most_probable = max(counts, key=counts.get)
    assert most_probable.startswith(expected_bin), f"Expected binary prefix {expected_bin}, got {most_probable}"