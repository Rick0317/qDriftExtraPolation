# Baby's first Phase Estimation Algorithm
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
import numpy as np
from numpy import linalg
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.quantum_info import SparsePauliOp, Operator

from typing import Optional, Union

def iterative_phase_estimation(U: Union[UnitaryGate, np.array], 
                               num_evals: int, 
                               eigenstate: Optional[Union[QuantumCircuit, np.array]],
                               backend,
                               num_qubits=1) -> float:
    
    # Data type management:

    # Convert U to a gate if it's a numpy array
    if isinstance(U, np.ndarray):
        U = UnitaryGate(U)
    
    elif isinstance(U, SparsePauliOp):
        op = Operator(U)
        U = UnitaryGate(op)
    
    elif isinstance(U, Operator):
        U = UnitaryGate(U)

    # Prepare the initial eigenstate
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circuit = QuantumCircuit(num_qubits, name='eigenstate')
        # the eigenstate is a vector of complex numbers that determine the state of the qubits other than the control qubit
        eigenstate_circuit.initialize(eigenstate)

    else:
        eigenstate_circuit = eigenstate

    # Actually create the circuit
    qc = QuantumCircuit(num_qubits + 1,1)
    qc.append(eigenstate_circuit, range(1, num_qubits + 1)) # Apply the eigenstate to the qubits
    qc.h(0) # Apply Hadamard to the control qubit
    u = U.control(1) # Create the controlled-U gate
    qc.append(u, range(num_qubits + 1)) # Apply the controlled-U gate
    qc.h(0) # Apply Hadamard to the control qubit
    qc.measure(0, cbit=0) # Measure the control qubit

    # Execute the circuit num_evals times
    num_zeros = 0
    tot = 0
    for i in range(num_evals):
        # Execute the circuit
        transp_qc = transpile(qc, backend=backend)
        job = backend.run(transp_qc, shots=1)
        result = job.result().get_counts()
        # print(result)
        # Check if the control qubit was measured to be 0
        if '0' in result:
            num_zeros += result['0']
        tot += 1
    return num_zeros / tot, qc


def iterative_phase_estimation_v2(U: Union[UnitaryGate, np.array], 
                               num_evals: int, 
                               eigenstate: Optional[Union[QuantumCircuit, np.array]],
                               backend,
                               num_qubits=1,
                               print_results=False) -> float:
    '''
    This function is a modified version of the iterative_phase_estimation function that uses the Qiskit Runtime Service'''
    
    # Data type management:
    # Convert U to a gate if it's a numpy array
    if isinstance(U, np.ndarray):
        U = UnitaryGate(U)
    
    elif isinstance(U, SparsePauliOp):
        op = Operator(U)
        U = UnitaryGate(op)
    
    elif isinstance(U, Operator):
        U = UnitaryGate(U)


    # Prepare the initial eigenstate
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circuit = QuantumCircuit(num_qubits, name='eigenstate')
        # the eigenstate is a vector of complex numbers that determine the state of the qubits other than the control qubit
        eigenstate_circuit.initialize(eigenstate)
    else:
        eigenstate_circuit = eigenstate

    # Actually create the circuit
    qc = QuantumCircuit(num_qubits + 1,1)
    qc.append(eigenstate_circuit, range(1, num_qubits + 1)) # Apply the eigenstate to the qubits
    qc.h(0) # Apply Hadamard to the control qubit
    u = U.control(1) # Create the controlled-U gate
    qc.append(u, range(num_qubits + 1)) # Apply the controlled-U gate
    qc.h(0) # Apply Hadamard to the control qubit
    qc.measure(0, cbit=0) # Measure the control qubit

    # Execute the circuit num_evals times
    job = backend.run(transpile(qc, backend), dynamic=False, shots=num_evals, memory=True, meas_level=0, meas_return='single')
    num_zeros = 0
    result = job.result()
    counts = result.get_counts()
    if print_results:
        print(counts)
    if '0' in counts:
        num_zeros = counts['0']
    tot = sum(counts.values())
    return num_zeros / tot, qc
        




    

