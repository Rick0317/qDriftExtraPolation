# Baby's first Phase Estimation Algorithm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
import numpy as np
from numpy import linalg
from qiskit import Aer, execute


from typing import Optional, Union

def iterative_phase_estimation(U: Union[UnitaryGate, np.array], 
                               num_evals: int, 
                               eigenstate: Optional[Union[QuantumCircuit, np.array]]) -> float:
    
    # Data type management:

    # Convert U to a gate if it's a numpy array
    if isinstance(U, np.ndarray):
        U = UnitaryGate(U)

    # Prepare the initial eigenstate
    if isinstance(eigenstate, np.ndarray):
        eigenstate_circuit = QuantumCircuit(1)
        eigenstate_circuit.initialize(eigenstate, 0)
    else:
        eigenstate_circuit = eigenstate

    # Actually create the circuit
    qc = QuantumCircuit(2,1)
    qc.append(eigenstate_circuit, [1]) # Prepare the eigenstate
    qc.h(0) # Apply Hadamard to the control qubit
    u = U.control(1) # Create the controlled-U gate
    qc.append(u, [0, 1]) # Apply the controlled-U gate
    qc.h(0) # Apply Hadamard to the control qubit
    qc.measure(0, cbit=0) # Measure the control qubit

    qc.draw('mpl')

    # Execute the circuit num_evals times
    num_zeros = 0
    tot = 0
    for i in range(num_evals):
        # Execute the circuit
        result = qc.measure_all()
        num_zeros += result.get('0', 0)
        tot += 1
    return num_zeros / tot
        

        
if __name__ == "__main__":
    # Example usage
    U = np.array([[1, 0], [0, -1]])
    eigenstate = np.array([1, 0])
    num_evals = 10

    # Run the algorithm. Determine how many times the control qubit was measured to be 0
    phase_estimate = iterative_phase_estimation(U, num_evals, eigenstate)


    

