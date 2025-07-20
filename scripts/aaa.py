import numpy as np
from typing import Dict, List, Tuple, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, PauliEvolutionGate
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


# ────────────────────────────────────────────────────────────────────
# 0)  Hamiltonian & helper objects  (2-qubit toy model)
# ────────────────────────────────────────────────────────────────────
H = SparsePauliOp.from_list([('ZZ', 0.5), ('XZ', 1.0), ('YZ', 2.0)])
λ = np.abs(H.coeffs).sum()                       # Σ|h_j|
SAMPLE_LABELS = H.paulis.to_labels()             # ['ZZ','XZ','YZ']
PLACEHOLDER_LABEL = 'dummy'
PMF = np.abs(H.coeffs) / λ

tau = Parameter('tau')

def _make_pauli_gate_cache(ham: SparsePauliOp,
                           placeholder_label: str) -> Dict[str, PauliEvolutionGate]:
    cache = {}
    for coeff, lab in zip(ham.coeffs, ham.paulis.to_labels()):
        cache[lab] = PauliEvolutionGate(SparsePauliOp([lab], [np.sign(coeff)]), tau)
    # one-qubit identity for placeholders
    n = ham.num_qubits
    cache[placeholder_label] = PauliEvolutionGate(SparsePauliOp([('I' * n, 1.0)]), tau)
    return cache

PAULI_CACHE = _make_pauli_gate_cache(H, PLACEHOLDER_LABEL)


# exact ground state just for completeness (2 qubits ⇒ cheap)
evals, evecs = np.linalg.eigh(H.to_matrix())
GROUND = evecs[:, evals.argmin()]
def prepare_eigen_circuit(state):
    circ = QuantumCircuit(int(np.log2(len(state))))
    circ.initialize(state)
    return circ

EIG_CIRC = prepare_eigen_circuit(GROUND)
N_SYSTEM = H.num_qubits


# ────────────────────────────────────────────────────────────────────
# 1)  template circuit with *all* placeholders
# ────────────────────────────────────────────────────────────────────
def _template_circuit(n_anc: int,
                      placeholder_label: str,
                      eig_circ: QuantumCircuit) -> QuantumCircuit:
    qc = QuantumCircuit(n_anc + N_SYSTEM, n_anc, name='qDRIFT-QPE')
    qc.append(eig_circ, range(n_anc, n_anc+N_SYSTEM))
    qc.h(range(n_anc))

    sys = range(n_anc, n_anc+N_SYSTEM)
    for k in range(n_anc):
        for _ in range(2**k):
            ph = PAULI_CACHE[placeholder_label].control(1)
            ph.name = f'ctrl-evolution-{k}-{placeholder_label}'
            qc.append(ph, [k, *sys])

    qc.append(QFT(n_anc, inverse=True), range(n_anc))
    qc.measure(range(n_anc), range(n_anc))
    return qc


# ────────────────────────────────────────────────────────────────────
# 2)  trajectory builder
# ────────────────────────────────────────────────────────────────────
TEMPLATE_CACHE: Dict[int, QuantumCircuit] = {}

def build_qdrift_trajectory(n_anc: int,
                            total_time: float,
                            rng: np.random.Generator,
                            pmf: np.ndarray,
                            n_qdrift_segments: int = 1,
                            placeholder_label: str = PLACEHOLDER_LABEL
) -> QuantumCircuit:
    if n_anc not in TEMPLATE_CACHE:
        TEMPLATE_CACHE[n_anc] = _template_circuit(n_anc,
                                                  placeholder_label,
                                                  EIG_CIRC)

    qc = TEMPLATE_CACHE[n_anc].copy()

    n_ph = 2**n_anc - 1
    words: List[str] = rng.choice(SAMPLE_LABELS, p=pmf, size=n_ph)

    τval = λ * total_time / (n_ph * n_qdrift_segments)

    word_it = iter(words)
    for inst in qc.data:
        if inst.operation.name.startswith('ctrl-evolution'):
            lab = next(word_it)
            inst.operation = (
                PAULI_CACHE[lab]
                .assign_parameters({'tau': τval})
                .control(1)
            )
    return qc


# ────────────────────────────────────────────────────────────────────
# 3)  demo run   (n_anc = 6 , 10 time-points)
# ────────────────────────────────────────────────────────────────────
times = np.logspace(-2, 1, 10)
rng   = np.random.default_rng(2024)

sim   = AerSimulator(method='statevector')

for t in times:
    circ = build_qdrift_trajectory(6, t, rng, PMF)
    res  = sim.run(transpile(circ, sim), shots=512).result()
    bit  = max(res.get_counts(), key=res.get_counts().get)
    print(f"t={t:5.3f}   depth={circ.depth():4d}   most_probable={bit}")

# visualise first circuit
circ.draw('mpl')