import sys
import os
import numpy as np
from qiskit_aer import Aer
import scipy
import json
import pandas as pd
import plotly.express as px
import tqdm
import openpyxl
from openpyxl.styles import PatternFill
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # this is ugly af but noting else worked for some reason
from algo.iterative_phase_estimation import iterative_phase_estimation_v2

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

    for i, test_hermitian in tqdm.tqdm(enumerate(random_hermitians), desc='Running phase estimation on random Hermitian matrices', total=NUM_RAND_MATRICES):

        exact_eigenvalue, eigenvectors= scipy.linalg.eig(test_hermitian)
        U = scipy.linalg.expm(1j * test_hermitian) # unitary operator corresponding to the Hermitian matrix
        eigenstate = eigenvectors[:, np.argmin(exact_eigenvalue)] # pick eigenstate that corresponds to smallest eigenvalue
        estimated_eigenvalues = []
        estimation_errors = []

        for _ in range(NUM_TRIALS):
            p_0, qc = iterative_phase_estimation_v2(U=U, eigenstate=eigenstate, num_evals=PHASE_ESTIMATION_MEAUREMENTS, backend=backend, num_qubits=1)
            # print(p_0)
            estimated_eigenvalue = 2 * np.arccos(np.sqrt(p_0)) # convert estimated probability of measuring 0 to estimated eigenvalue
            estimated_eigenvalues.append(estimated_eigenvalue)
            estimation_errors.append(np.abs(estimated_eigenvalue - min(exact_eigenvalue))) # error = difference between smallest true eigenvalue and estimated eigenvalue
        
        print(f'Hermitian matrix {i + 1} / {NUM_RAND_MATRICES}:\n{test_hermitian} \nAverage estimation error: {np.mean(estimation_errors)}\n\n')
        all_estimation_errors[i] = estimation_errors

    # Write all_estimation_errors to json file
    with open('estimation_errors.json', 'w') as f:
        json.dump(all_estimation_errors, f)
    print('All estimation errors written to estimation_errors.json')

    df = pd.DataFrame(columns=['Matrix', 'True Eigenvalue', 'Average Estimation Error', 'Max Estimation Error'])
    for i, (test_hermitian, estimation_errors) in enumerate(zip(random_hermitians, all_estimation_errors.values())):
        exact_eigenvalue, _ = scipy.linalg.eig(test_hermitian)
        df.loc[i] = [test_hermitian, min(exact_eigenvalue), np.mean(estimation_errors), max(estimation_errors)]
    
    # Write df to excel file
    with pd.ExcelWriter('estimation_errors.xlsx') as writer:
        df.to_excel(writer, sheet_name='Estimation Errors', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Estimation Errors']
        red_fill = PatternFill(start_color='FFFF0000', end_color='FFFF0000', fill_type='solid')
        for i in range(2, len(df) + 2):
            worksheet[f'C{i}'].fill = red_fill
            worksheet[f'D{i}'].fill = red_fill
            worksheet[f'E{i}'].fill = red_fill

    # Create a histogram of the estimation errors
    all_errors = [error for errors in all_estimation_errors.values() for error in errors]
    fig_hist = px.histogram(all_errors, nbins=30, title='Histogram of Estimation Errors')
    fig_hist.update_layout(xaxis_title='Estimation Error', yaxis_title='Frequency')
    fig_hist.show()

    # Create scatter plots
    true_eigenvalues = [min(scipy.linalg.eig(matrix)[0]) for matrix in random_hermitians]
    average_errors = [np.mean(errors) for errors in all_estimation_errors.values()]
    max_errors = [max(errors) for errors in all_estimation_errors.values()]

    fig_avg = px.scatter(x=true_eigenvalues, y=average_errors, title='True Eigenvalues vs Average Estimation Errors')
    fig_avg.update_layout(xaxis_title='True Eigenvalue', yaxis_title='Average Estimation Error')
    fig_avg.show()

    fig_max = px.scatter(x=true_eigenvalues, y=max_errors, title='True Eigenvalues vs Max Estimation Errors')
    fig_max.update_layout(xaxis_title='True Eigenvalue', yaxis_title='Max Estimation Error')
    fig_max.show()











