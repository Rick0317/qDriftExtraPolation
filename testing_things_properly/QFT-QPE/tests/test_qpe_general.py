
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from qiskit.quantum_info import Operator
from qiskit.circuit.library import PhaseGate
from algos import standard_qpe
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.visualization import plot_histogram
import pytest
import math

# Setup for the test environment
LOG_FILE = "qpe_histograms_log.txt"
if os.path.exists(LOG_FILE): # Clear log file at start of test run
    os.remove(LOG_FILE)

# Test function for Quantum Phase Estimation with a simple phase gate
@pytest.mark.parametrize("phase, expected_bin", [
    (math.pi / 4, '001'),
    (math.pi / 2,  '010'),
    (3 * math.pi / 4, '011'),
    (math.pi,   '100'),
    (5 * math.pi / 4, '101'),
    (3 * math.pi / 2,  '110'),
    (7 * math.pi / 4, '111'),
])
def test_general_qpe_with_parametrized_phase(phase, expected_bin):
    unitary = Operator(PhaseGate(phase))
    
    eigenstate = QuantumCircuit(1)
    eigenstate.x(0)

    num_ancilla = 3
    shots = 1024
    qc = standard_qpe(unitary, eigenstate, num_ancilla)

    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    most_probable = max(counts, key=counts.get)
    estimated_decimal = int(most_probable, 2) / (2 ** num_ancilla)
    estimated_phase = 2 * math.pi * estimated_decimal
    assert most_probable.startswith(expected_bin), f"Expected prefix {expected_bin}, got {most_probable}"

    plot_histogram(counts).savefig(f"output_general_phase_{round(phase, 3)}.png")
    # Generate filename and save histogram
    filename = f"output_general_phase_{round(phase, 3)}_a{num_ancilla}.png"
    plot_histogram(counts).savefig(filename)

    # save circuit diagram
    qc.draw("mpl").savefig(f"circuit_general_phase_{round(phase, 3)}_a{num_ancilla}.png")

    # Log data
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{filename},{phase},{num_ancilla},{most_probable},{estimated_phase}\n")

def test_general_qpe_with_n_fold_tensor_prod_of_paulis():
# test that result is the best n-bit approximation of the phase



def test_general_qpe_with_known_hamiltonian():
    # e^{-iHt}