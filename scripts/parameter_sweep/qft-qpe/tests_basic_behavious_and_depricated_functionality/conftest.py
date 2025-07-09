# sane_applications/QFT-QPE/tests/conftest.py
import sys, os

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
