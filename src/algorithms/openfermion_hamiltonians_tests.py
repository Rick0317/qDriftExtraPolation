import csv
import openfermion as of
import openfermionpsi4 as ofpsi4
from qdrift_qpe import babys_first_qpe
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from tqdm import tqdm
import os


from openfermion import MolecularData
from openfermion import geometry_from_pubchem
from openfermion import get_fermion_operator, get_sparse_operator, jordan_wigner

import time
import pickle
import numpy as np
import random
from scipy.linalg import expm
import scipy
import matplotlib.pyplot as plt


from qiskit.quantum_info import Pauli


MOLECULE_LIST = ["H2", "F2", "HCl", "LiH", "H2O", "CH2", "O2", "BeH2","H2S",
                 "NH3", "N2", "CH4", "C2", "LiF", "PH3", "LiCL", "Li2O"]

def get_geometry(molecule_name, verbose=True):
    if verbose and (molecule_name not in MOLECULE_LIST):
        print(f"Warning: {molecule_name} is not one of the molecules used in the paper" + 
               "- that's not wrong, but just know it's not recreating the published results!")
    
    if molecule_name=="C2":
        # C2 isn't in PubChem - don't know why.
        geometry = [('C', [0.0, 0.0, 0.0]), ('C', [0.0, 0.0, 1.26])]
    else:
        if molecule_name=="Li2O":
            # Li2O returns a different molecule - again, don't know why.
            molecule_name = "Lithium Oxide"
        geometry = geometry_from_pubchem(molecule_name)
        
    return geometry

def prepare_psi4(molecule_name,
                 geometry = None,
                 multiplicity = None,
                 charge = None,
                 basis = None):

    if multiplicity is None:
        multiplicity = 1 if molecule_name not in ["O2","CH2"] else 3
    if charge is None:
        charge = 0
    if basis is None:
        basis = 'sto-3g'

    if multiplicity == 1:
        reference = 'rhf'
        guess = 'sad'
    else:
        reference = 'rohf'
        guess  = 'gwh'
        
    if geometry is None:
        geometry = get_geometry(molecule_name)

    geo_str = ""
    for atom, coords in geometry:
        geo_str += f"\n\t{atom}"
        for ord in coords:
            geo_str += f" {ord}"
        geo_str += ""

    psi4_str =f'''
molecule {molecule_name} {{{geo_str}
    {charge} {multiplicity}
    symmetry c1
}}
set basis       {basis}
set reference   {reference}

set globals {{
    basis {basis}
    freeze_core false
    fail_on_maxiter true
    df_scf_guess false
    opdm true
    tpdm true
    soscf false
    scf_type pk
    maxiter 1e6
    num_amps_print 1e6
    r_convergence 1e-6
    d_convergence 1e-6
    e_convergence 1e-6
    ints_tolerance EQUALITY_TOLERANCE
    damping_percentage 0
}}

hf = energy("scf")

# cisd = energy("cisd")
ccsd = energy("ccsd")
ccsdt = energy("ccsd(t)")
fci = energy("fci")

print("Results for {molecule_name}.dat\\n")

print("""Geometry : {geo_str}\\n""")

print("HF : %10.6f" % hf)
# print("CISD : %10.6f" % cisd)
print("CCSD : %10.6f" % ccsd)
print("CCSD(T) : %10.6f" % ccsdt)
print("FCI : %10.6f" % fci)
    '''

    fname = f'{molecule_name}.dat'
    with open(fname, 'w+') as psi4_file:
        psi4_file.write(psi4_str)
    print(f"Created {fname}.")
    print(f"To solve molecule, run 'psi4 {fname}' from command line.")


def create_molecule_data(molecule_name,
                         geometry = None,
                         multiplicity = None,
                         charge = None,
                         basis = None,
                         save_name=None):

    if multiplicity is None:
        multiplicity = 1 if molecule_name not in ["O2","CH2"] else 3
    if charge is None:
        charge = 0
    if basis is None:
        basis = 'sto-3g'
    if save_name is None:
        save_name = molecule_name
        
    if geometry is None:
        geometry = get_geometry(molecule_name)
        
    molecule = MolecularData(geometry,
                             basis = basis,                      
                             multiplicity = multiplicity,
                             charge = charge,
                             filename=save_name
                             )
    
    # 1. Solve molecule and print results.
    
    print("Solving molecule with psi4", end="...")
    t_start=time.time()
    
    molecule = ofpsi4.run_psi4(molecule,
                                run_scf=True,
                                run_mp2=True,
                                run_cisd=True,
                                run_ccsd=True,
                                run_fci=True,
                                memory=16000,
                                delete_input=True,
                                delete_output=True,
                                verbose=True)
    print("done in {:.2f} seconds".format(time.time()-t_start))
    
    print(f'{molecule_name} has:')
    print(f'\tgeometry of {molecule.geometry},')
    print(f'\t{molecule.n_electrons} electrons in {2*molecule.n_orbitals if molecule.n_orbitals is not None else "n/a"} spin-orbitals,')
    print(f'\tHartree-Fock energy of {molecule.hf_energy if molecule.hf_energy is not None else "n/a"} Hartree,')
    print(f'\tCISD energy of {molecule.cisd_energy if molecule.cisd_energy is not None else "n/a"} Hartree,')
    print(f'\tCCSD energy of {molecule.ccsd_energy if molecule.ccsd_energy is not None else "n/a"} Hartree,')
    print(f'\tFCI energy of {molecule.fci_energy if molecule.fci_energy is not None else "n/a"} Hartree.')

    # 2. Save molecule.
    
    # molecule.filename=save_name
    molecule.save()
    
    print(f"Molecule saved to {save_name}.hdf.")
    
    # 3. Convert molecular Hamiltonian to qubit Hamiltonian.
    print("Converting molecular Hamiltonian to qubit Hamiltonian", end="...")
    
    active_space_start=0
    active_space_stop=molecule.n_orbitals if molecule.n_orbitals is not None else 2*molecule.n_electrons if molecule.n_electrons is not None else 0

    # Get the Hamiltonian in an active space.
    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=None,
        active_indices=range(active_space_start, active_space_stop))

    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qubit_hamiltonian.compress()
    
    print("done in {:.2f} seconds".format(time.time()-t_start))
    
    # 3. Save qubit Hamiltonian.

    with open(save_name+"_qubit_hamiltonian.pkl",'wb') as f:
        pickle.dump(qubit_hamiltonian,f)
        
    print(f"Qubit Hamiltonian saved to {save_name+'_qubit_hamiltonian.pkl'}.")
    

def load_qubit_hamiltonian(file_path):
    """
    Load the qubit Hamiltonian from the given pickle file and convert it to Qiskit-compatible format.
    """
    with open(file_path, 'rb') as f:
        qubit_hamiltonian = pickle.load(f)
    
    # Convert OpenFermion Hamiltonian to Qiskit format
    hamiltonian_terms = []
    n_qubits = len(max(qubit_hamiltonian.terms.keys(), key=len))

    for term, coeff in qubit_hamiltonian.terms.items():
         
        pauli_str = ["I"  for _ in range(n_qubits)]
        
        for tensor_term in term:
            pauli_str[tensor_term[0]] = tensor_term[1]
            hamiltonian_terms.append((coeff.real, Pauli("".join(pauli_str))))
    
    return hamiltonian_terms

def run_quantum_workflow(molecule_file, time, num_samples, eigenstate, shots, qubits):
    """
    Load Hamiltonian, run qdrift_channel, and perform extrapolation.
    """
    hamiltonian_terms = load_qubit_hamiltonian(molecule_file)
    
    # Run QDrift simulation
    results, v_lst = qdrift_channel(hamiltonian_terms, time, num_samples, eigenstate, shots, qubits)
    
    # Perform extrapolation
    extrapolation(time, hamiltonian_terms, eigenstate, expected=0, shots=shots, num_qubits=qubits, num_terms=len(hamiltonian_terms))
    
    print("Quantum workflow completed successfully.")

def qdrift_channel(hamiltonian_terms, time, num_samples, eigenstate, shots, qubits):
    """
    Simulates the QDrift algorithm.
    """
    lam = sum(abs(term[0]) for term in hamiltonian_terms)
    tau = time * lam / num_samples
    backend = Aer.get_backend('qasm_simulator')
    
    results = []
    prob_weights = [abs(coeff) for coeff, _ in hamiltonian_terms]
    
    for _ in tqdm(range(num_samples)):
        idx = random.choices(range(len(hamiltonian_terms)), weights=prob_weights, k=1)[0]
        h_j = hamiltonian_terms[idx][1]
        unitary = expm(1j * tau * h_j.to_matrix())
        
        qc = babys_first_qpe(unitary, num_evals=1, eigenstate=eigenstate, backend=backend, num_qubits=qubits)
        job = backend.run(transpile(qc, backend), shots=shots)
        counts = job.result().get_counts()
        if '0' in counts:
            results.append(np.arcsin(1 - 2 * counts['0'] / shots))
    
    return results, []

def extrapolation(time, hamiltonian, eigenstate, expected, shots, num_qubits, num_terms):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    domain = np.linspace(-1, 1, 200)
    axs_x, axs_y = 0, 0
    
    for n in range(2, 10, 2):
        data_points = []
        nodes = np.cos((2 * np.arange(1, n+1) - 1) / (2 * n) * np.pi)
        
        for node in nodes[:n//2]:
            num_samples = int(np.ceil(np.abs(time / node)))
            qpe_results, _ = qdrift_channel(hamiltonian, time, num_samples, eigenstate, shots, num_qubits)
            data_points.append(sum(qpe_results) / time)
        
        data_points.extend(data_points[::-1])
        axs[axs_x, axs_y].plot(domain, np.interp(domain, nodes, data_points), label=f'{n} nodes')
        axs[axs_x, axs_y].axhline(expected, color='r', linestyle='--')
        axs[axs_x, axs_y].set_title(f'Extrapolation with {n} nodes')
        axs_x, axs_y = (axs_x + axs_y) % 2, (axs_y + 1) % 2
        
        print(f"Estimated eigenvalues with {n} nodes: {data_points}")
        with open("extrapolation_results.csv", "a") as file:
            writer = csv.writer(file)
            writer.writerow([num_qubits, num_terms, sum(data_points)/len(data_points), n, time, shots])
    plt.savefig('extrapolation_results.png')
    plt.show()



if __name__ == "__main__":
    molecule_name = "LiH"
    # prepare_psi4(molecule_name)
    # run in the command line: psi4 CH4.dat
    # os.system(f"psi4 {molecule_name}.dat")
    create_molecule_data(molecule_name)

    """
    hamiltonian_terms = load_qubit_hamiltonian("N2_qubit_hamiltonian.pkl")
    num_qubits = len(hamiltonian_terms[0][1].to_label())
    num_terms = len(hamiltonian_terms)
    print(f"Number of qubits: {num_qubits}\nNumber of terms: {num_terms}")
    H = sum(coeff * term.to_matrix(True) for coeff, term in tqdm(hamiltonian_terms))
    print(f"H: {H}")
    eigenvalues, eigenvectors = scipy.linalg.eigh(H)
    eigenstate = eigenvectors[:, np.argmin(eigenvalues)]
    print(f"Eigenstate: {eigenstate}")
    results, v_lst = qdrift_channel(hamiltonian_terms, time=0.1, num_samples=2, eigenstate=eigenstate, shots=4096, qubits=num_qubits)
    print(f"Results: {results}")
    """




