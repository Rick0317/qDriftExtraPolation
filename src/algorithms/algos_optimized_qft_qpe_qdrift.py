from calendar import c
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator  # as of 25Mar2025
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, UnitaryGate, PhaseGate, RZGate
from qiskit.quantum_info import Operator
from IPython.display import display, Math
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.circuit.library import Initialize
from qiskit.circuit import Parameter, CircuitInstruction
import numpy as np
from typing import Dict, Optional, List


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
    return qc


def make_pauli_gate_cache(H: SparsePauliOp,
                          placeholder_label: str = "⋯") \
                          -> Dict[str, PauliEvolutionGate]:
    """
    Return dictionary  label ↦ *uncontrolled* PauliEvolutionGate
    carrying the COMMON Parameter tau.  A dummy I^{⊗n} placeholder is added.
    """
    tau = Parameter("tau")                   
    cache: Dict[str, PauliEvolutionGate] = {}
    n_sys = H.num_qubits

    for coeff, label in zip(H.coeffs, H.paulis.to_labels()):
        sign     = np.sign(coeff).item() or 1.0      # ±1
        pauli_op = SparsePauliOp.from_list([(label, sign)],  # minus → e^{-i …}
                                           num_qubits=n_sys)
        cache[label] = PauliEvolutionGate(pauli_op, tau).control(1)  # controlled evolution gate

    # never actually sampled, but convenient for the template
    n_q = H.num_qubits
    cache[placeholder_label] = PauliEvolutionGate(
        SparsePauliOp.from_list([("I"*n_q, -1.0)], num_qubits=n_q), tau
    )
    return cache, tau


def build_template_circuit(n_anc: int, 
                           n_sys: int, 
                           placeholder_label: str, 
                           eigenvalue_circuit: QuantumCircuit, 
                           gate_cache: Dict[str, PauliEvolutionGate]) -> QuantumCircuit:
    """
    Static scaffold that already contains *all*  controlled placeholders:
    layer k gets 2**k placeholders so the stochastic product can be realised.
    """
    qc = QuantumCircuit(n_anc + n_sys, n_anc, name="qDRIFT-QPE")
    qc.append(eigenvalue_circuit, range(n_anc, n_anc+n_sys))
    qc.h(range(n_anc))

    # --- allocate the necessary number of placeholders -----------------
    ctrl_gates = []
    for k in range(n_anc):
        for _ in range(2**k):
            dummy_gate = gate_cache[placeholder_label].control(1) 
            dummy_gate.name = f"ctrl-evolution-{k}-{placeholder_label}" # explicitly name the placeholder
            ctrl_gates.append((k, dummy_gate))  # store the layer and gate

    # add them in the *same* order every time  →  topology  is fixed
    sys = range(n_anc, n_anc+n_sys)
    for k, dummy in ctrl_gates:
        qc.append(dummy, [k, *sys])

    qc.append(QFT(n_anc, inverse=True), range(n_anc))
    qc.measure(range(n_anc), range(n_anc))
    return qc


def build_qdrift_trajectory(
        n_anc: int,
        h_signature: Optional[str],
        total_time: float,
        H: SparsePauliOp,
        rng: np.random.Generator,
        n_qdrift_segments: Optional[int],
        placeholder_label: str,
        template_circuit_cache: Dict[str, QuantumCircuit],
        exponentialed_hamiltonian_terms_cache: Dict[str, PauliEvolutionGate],
        ground_state: bool
) -> QuantumCircuit:
    """
    Return ONE qDRIFT–QPE circuit whose unitary equals the stochastic product
    required by the algorithm in §1 (N = 2^m-1 samples, each drawn i.i.d.).

    Returns
    -------
    QuantumCircuit
        Fully-bound and ready-to-simulate trajectory.
    """
    # 1 ───── copy static scaffold ────────────────────────────────────────
    template_qc = template_circuit_cache[(h_signature, n_anc, ground_state)].copy()

    # 2 ───── draw the Pauli words  (independent for every placeholder) ──
    n_placeholders = 2**n_anc - 1
    pauli_labels = list(H.paulis.to_labels())
    pmf = np.abs(H.coeffs) / np.sum(np.abs(H.coeffs))  # probability mass function
    # print(f"Generating {n_placeholders} placeholders for {n_anc} ancilla qubits.\nDrawing from pmf: {pmf} \nHamiltonian: {H}.\nLabels: {pauli_labels}")
    words: List[str] = rng.choice(pauli_labels, p=pmf, size=n_placeholders)
    # print(f"Drawn words: {words}")

    new_data = []
    word_iterator = iter(words)
    c1, c2 = 0, 0
    for instruction in template_qc.data:
        if placeholder_label in instruction.operation.name:
            chosen_word = next(word_iterator)
            new_gate = exponentialed_hamiltonian_terms_cache[chosen_word]
            new_instruction = CircuitInstruction(new_gate, instruction.qubits, instruction.clbits)
            new_data.append(new_instruction)
            c1 += 1
        else:
            new_data.append(instruction)
            c2 += 1
    assert c1 > 0, "No placeholders were replaced in the template circuit."
    assert next(word_iterator, None) is None, "Not all words consumed"


    template_qc.data = new_data
    # print(f"New circuit data: {template_qc.data}")
    print(f"Replaced {c1} placeholders with new gates, kept {c2} original instructions.")
    lam = np.sum(np.abs(H.coeffs))
    tau_val = (lam * total_time) / n_qdrift_segments if n_qdrift_segments > 0 else 0
    tau_param = exponentialed_hamiltonian_terms_cache[placeholder_label].params[0]
    
    qdrift_trajectory = template_qc.assign_parameters({tau_param: tau_val})
    print(f"Built qDRIFT trajectory for {h_signature} with {n_anc} ancillas.")

    return qdrift_trajectory