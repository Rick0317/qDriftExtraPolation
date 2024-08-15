import numpy as np
from qiskit_aer import Aer
import scipy
import json
from ..algo.iterative_phase_estimation import iterative_phase_estimation_v2
import pandas as pd
import plotly.express as px

def generate_random_hermitian(print_output=False):
    # Generate a random 2x2 Hermitian matrix with real, non-negative eigenvalues
    while True:
        a = np.random.rand()
        b = np.random.rand()
        c = np.random.rand()
        d = np.random.rand()
        
        # Construct the Hermitian matrix
        hermitian_matrix = np.array([[a, b + 1j * c], [b - 1j * c, d]])
        
        # Calculate eigenvalues
        eigenvalues, _ = np.linalg.eig(hermitian_matrix)
        
        # Check if the matrix is Hermitian and has real, non-negative eigenvalues
        if np.all(np.isreal(eigenvalues)) and np.all(eigenvalues >= 0):
            if print_output:
                print(f'Hermitian matrix: \n{hermitian_matrix}')
                print(f'Eigenvalues: {eigenvalues}')
            return hermitian_matrix
        
if __name__ == "__main__":

    NUM_TRIALS = 100
    PHASE_ESTIMATION_MEAUREMENTS = 10000
    NUM_RAND_MATRICES = 10
    backend = Aer.get_backend('qasm_simulator')
    random_hermitians = [generate_random_hermitian() for _ in range(NUM_RAND_MATRICES)]
    all_estimation_errors = {}

    for i, test_hermitian in enumerate(random_hermitians):

        exact_eigenvalue, eigenvectors= scipy.linalg.eig(test_hermitian)
        U = scipy.linalg.expm(1j * test_hermitian) # unitary operator corresponding to the Hermitian matrix
        eigenstate = eigenvectors[:, np.argmin(exact_eigenvalue)] # pick eigenstate that corresponds to smallest eigenvalue
        estimated_eigenvalues = []
        estimation_errors = []

        for _ in range(NUM_TRIALS):
            p_0, qc = iterative_phase_estimation_v2(U=U, eigenstate=eigenstate, num_evals=PHASE_ESTIMATION_MEAUREMENTS, backend=backend, num_qubits=1)
            print(p_0)
            estimated_eigenvalue = 2 * np.arccos(np.sqrt(p_0)) # convert estimated probability of measuring 0 to estimated eigenvalue
            estimated_eigenvalues.append(estimated_eigenvalue)
            estimation_errors.append(np.abs(estimated_eigenvalue - min(exact_eigenvalue))) # error = difference between smallest true eigenvalue and estimated eigenvalue
        
        print(f'Hermitian matrix {i + 1} / {NUM_RAND_MATRICES}:\n{test_hermitian} \nAverage estimation error: {np.mean(estimation_errors)}\n\n')
        all_estimation_errors[i] = estimation_errors

    # Write all_estimation_errors to json file
    with open('estimation_errors.json', 'w') as f:
        json.dump(all_estimation_errors, f)
    print('All estimation errors written to estimation_errors.json')

    # create pandas dataframe with the matrices, the true eigenvalues,  the average estimation errors, and the maximum estimation errors
    # write the dataframe to a csv file
    # create a histogram of the estimation errors
    # create a scatter plot of the true eigenvalues vs the estimation errors
    # create a scatter plot of the true eigenvalues vs the maximum estimation errors
    # create a scatter plot of the true eigenvalues vs the average estimation errors

    df = pd.DataFrame(columns=['Matrix', 'True Eigenvalue', 'Average Estimation Error', 'Max Estimation Error'])
    for i, (test_hermitian, estimation_errors) in enumerate(zip(random_hermitians, all_estimation_errors.values())):
        exact_eigenvalue, _ = scipy.linalg.eig(test_hermitian)
        df.loc[i] = [test_hermitian, min(exact_eigenvalue), np.mean(estimation_errors), max(estimation_errors)]
    
    df.to_csv('estimation_errors.csv')
    print('Dataframe written to estimation_errors.csv')

    fig = px.histogram(x=estimation_errors, title='Histogram of Estimation Errors')
    fig.write_html('histogram.html')
    print('Histogram of estimation errors written to histogram.html')











