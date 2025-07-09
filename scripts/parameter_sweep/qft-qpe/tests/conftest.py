# sane_applications/QFT-QPE/tests/conftest.py
import sys, os
import gc
import pytest
from qiskit_aer import AerSimulator

# Navigate from tests/ → QFT-QPE/ → sane_applications/ → playing_with_quantum_computing/
root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # …/tests
        "..",                        # …/QFT-QPE
        "..",                        # …/sane_applications
        ".."                         # …/playing_with_quantum_computing
    )
)
sys.path.insert(0, root)



# =============================
# Garbage Collection Hook
# =============================

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Trigger GC before each test to free memory."""
    gc.collect()


# =============================
# Parametrization-Aware Mark Injection
# =============================

def pytest_collection_modifyitems(config, items):
    """
    Dynamically mark heavy tests based on (qubits, randomness).
    """
    for item in items:
        name = item.name

        # Extract parameters from test ID
        num_qubits = None
        num_circuits = None
        num_shots = None

        # Safely parse parameterized test IDs
        for part in name.split('['):
            if 'qubit' in part:
                try:
                    num_qubits = int(part.split('qubit')[-1].split('-')[0])
                except ValueError:
                    continue
            if '1024' in part:
                num_circuits = 1024
            if '-10' in part:
                num_shots = 10

        # Decide on mark
        if num_qubits == 10:
            item.add_marker(pytest.mark.xdist_group("heavy_sim"))  # isolate in one worker