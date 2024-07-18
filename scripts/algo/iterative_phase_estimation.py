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
        

        
if __name__ == "__main__":
    # Example usage: 2 qubit Hadamard gate
    U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    eigenstate = np.array([1,0,0,0])
    num_evals = 100

    # Run the algorithm. Determine how many times the control qubit was measured to be 0.

    # Now, to exectue a qiksit circuit, qe need to specify a few things:
    # 1. The backend: this is the simulator or the real quantum computer that will run the circuit
    '''service = QiskitRuntimeService()
    available_backends = service.backends()
    filtered_backends = service.backends(available=True, min_num_qubits=2, simulator=True)
    selected_backend = service.least_busy(filtered_backends)'''
    # Since I dont have an account, I will use the Aer simulator
    selected_backend = Aer.get_backend('qasm_simulator')

    # 2. Experiment: this is the circuit that we want to run. It is a QuantumCircuit object
    phase_estimate, qc = iterative_phase_estimation(U, num_evals, eigenstate, selected_backend, num_qubits=2)
    qc.draw('mpl')
    print(phase_estimate)


    

