# phase estimation code
from qiskit import ClassicalRegister, transpile, QuantumCircuit
from qiskit.circuit.library import PhaseEstimation, QFT
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.circuit.library import UnitaryGate
import numpy as np


def phase_estimation(num_eval_qubits, unitary):
    """
    Phase estimation of the quantum state.
    :param num_eval_qubits: Number of evaluation qubits with initial state 0
    :param unitary: Unitary operator
    :return: DataBin object of the result
    """
    # Choose the backend
    backend = FakeManilaV2()

    # Classical register bits for measurement result
    cl_register = ClassicalRegister(num_eval_qubits, "cl1")

    # Unitary gate representation of the unitary matrix/operator
    unitary_circuit = UnitaryGate(unitary)

    # Phase estimation circuit
    pe_circuit = PhaseEstimation(num_eval_qubits, unitary_circuit, iqft=None,
                                 name='QPE')

    # Add the classical register to the Phase Estimation circuit
    pe_circuit.add_register(cl_register)

    # Optimize the circuit for the backend topology
    transpiled_circuit = transpile(pe_circuit, backend)
    transpiled_circuit.measure(range(num_eval_qubits), cl_register)

    # Sampler. Measures the probability of certain outputs.
    sampler = SamplerV2(backend)

    job = sampler.run([transpiled_circuit])
    result = job.result()[0]

    return result.data


def manual_phase_estimation(num_eval_qubits, unitary, state_qubits, state_prep):
    """
    Phase estimation with manually constructed circuit
    Args:
        num_eval_qubits: Number of evaluation qubits with initial state 0
        unitary: The unitary operator
        state_qubits: The number of qubits for preparing the state
        state_prep: The preparation circuit of the state

    Returns: DataBin object of the result

    """
    backend = FakeManilaV2()
    unitary_circuit = UnitaryGate(unitary)
    controlled_u = unitary_circuit.control(1)
    t = num_eval_qubits
    qpe = QuantumCircuit(t + state_qubits, 3)
    qpe.append(state_prep, [t + i for i in range(state_qubits)])
    for qubit in range(t):
        qpe.h(qubit)

    repetitions = 1
    for counting_qubit in range(t):
        for i in range(repetitions):
            qpe.append(controlled_u, [counting_qubit, t])
            # qpe.cp(math.pi/8, counting_qubit, t)
        repetitions *= 2

    myQft = QFT(num_qubits=t, do_swaps=True, inverse=True, insert_barriers=True,
                name="myQFT")
    qpe.append(myQft, [0, 1, 2])

    for n in range(3):
        qpe.measure(n, n)

    qpe.draw()

    transpiled_circuit = transpile(qpe, backend)
    sampler = SamplerV2(backend)

    job = sampler.run([transpiled_circuit])

    result = job.result()[0]
    print(qpe)

    return result


if __name__ == '__main__':
    num_eval_qubits = 2

    unitary = np.array([[1, 0], [0, np.exp(np.pi * 1j / 4)]])
    state_qubits = 1
    state_prep = QuantumCircuit(state_qubits)
    state_prep.x(0)
    result = manual_phase_estimation(3, unitary, state_qubits, state_prep).data.c.array
    print(result)

