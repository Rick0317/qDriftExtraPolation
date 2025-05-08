import sys
import os
import numpy as np
from qiskit_aer import Aer
import scipy
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tqdm
import openpyxl
from openpyxl.styles import PatternFill
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # this is ugly af but noting else worked for some reason
from scripts.algo import iterative_phase_estimation_v2

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


def probability_of_estimation_error(delta, n):
    # P [ˆθ − θ ≥ δ ] ≤ 2e^{−2nδ^2}
    # returns upper bound for probability that the estimation error is greater than delta
    return 2 * np.exp(-2 * n * delta**2)

def upper_bound_linear(p0, delta):
    # returns upper bound on the error in phase estimation given the error in p0 estimation
    return delta / np.sqrt(p0 - 3*p0**2) if p0 - 3*p0**2 > 0 else np.inf

def upper_bound_nonlinear(p0, delta):
    # returns upper bound on the error in phase estimation given the error in p0 estimation
    return delta / np.sqrt(-(p0 + delta - 1) * (p0 + delta )) if -(p0 + delta - 1) * (p0 + delta ) > 0 else np.inf

if __name__ == "__main__":

    NUM_TRIALS = 10000
    NUM_RAND_MATRICES = 15
    backend = Aer.get_backend('qasm_simulator')
    random_hermitians = [generate_random_hermitian() for _ in range(NUM_RAND_MATRICES)]
    all_estimation_errors = {}

    for i in [5,50,5000,10000]:
        PHASE_ESTIMATION_MEAUREMENTS = i
        all_p0_estimation_errors = []
        fig = go.Figure()
        phase_error_plot = go.Figure()

        for i, test_hermitian in tqdm.tqdm(enumerate(random_hermitians), desc='Running phase estimation on random Hermitian matrices', total=NUM_RAND_MATRICES):

            exact_eigenvalue, eigenvectors= scipy.linalg.eig(test_hermitian)
            exact_p0 = np.cos(min(exact_eigenvalue) / 2) ** 2 # we kind of arbitrarily choose the smallest eigenvalue as the one we want to estimate
            U = scipy.linalg.expm(1j * test_hermitian) # unitary operator corresponding to the Hermitian matrix
            eigenstate = eigenvectors[:, np.argmin(exact_eigenvalue)] # pick eigenstate that corresponds to smallest eigenvalue

            estimated_eigenvalues = []
            estimation_errors = []
            p0_estimation_errors = []

            for _ in range(NUM_TRIALS):

                p_0, qc = iterative_phase_estimation_v2(U=U, eigenstate=eigenstate, num_evals=PHASE_ESTIMATION_MEAUREMENTS, backend=backend, num_qubits=1)
                estimated_eigenvalue = 2 * np.arccos(np.sqrt(p_0)) # convert estimated probability of measuring 0 to estimated eigenvalue
                estimated_eigenvalues.append(float(estimated_eigenvalue))
                estimation_errors.append(np.abs(estimated_eigenvalue - min(exact_eigenvalue))) # error = difference between smallest true eigenvalue and estimated eigenvalue
                p0_estimation_errors.append(np.abs(p_0 - exact_p0))

            all_p0_estimation_errors.extend(p0_estimation_errors)

            # Define the number of bins and the bin edges
            num_bins = 100
            bin_edges = np.linspace(0, 1, num_bins + 1)

            # Manually bin the data and normalize the bin counts with respect to the most frequent bin
            hist, bin_edges = np.histogram(p0_estimation_errors, bins=bin_edges)
            max_count = np.max(hist)
            relative_freq = hist / max_count
            fig.add_trace(go.Bar(x=bin_edges[:-1], y=relative_freq, name=f'Matrix {i + 1}'))

            # now, plot error in estimated eigenvalues vs error in estimated p_0
            phase_error_plot.add_trace(go.Scatter(x=estimation_errors, y=p0_estimation_errors, mode='markers', name=f'Matrix {i + 1}'))


        # Generate data for the function that upperbound p0 error
        delta_values = np.linspace(-0.00001, 1, 100)
        prob_values = probability_of_estimation_error(delta_values, n=PHASE_ESTIMATION_MEAUREMENTS)
        fig.add_trace(go.Scatter(x=delta_values, y=prob_values, mode='lines', name='Probability of Estimation Error'))


        # Add label for n
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f'n = {PHASE_ESTIMATION_MEAUREMENTS}',
            showarrow=False,
            font=dict(size=16)
        )
        # Add label for number of trials
        fig.add_annotation(
            x=0.5,
            y=0.4,
            text=f'Number of trials = {NUM_TRIALS}',
            showarrow=False,
            font=dict(size=16)
        )

        # Update the layout
        fig.update_layout(
            title=f'Relative frequency of p_0 estimation errors for PHASE_ESTIMATION_MEASUREMENTS = {PHASE_ESTIMATION_MEAUREMENTS}',
            xaxis_title='p_0 estimation error',
            yaxis_title='Relative Frequency',
            xaxis=dict(range=[0, 0.5])
        )

        # Save the plot as a png
        fig.write_image(f'p0_estimation_errors_{PHASE_ESTIMATION_MEAUREMENTS}.png')
        fig.show()

        # generate plot for error in estimated eigenvalues vs error in estimated p_0
        phase_error_plot.update_layout(
            title=f'Error in estimated eigenvalues vs error in estimated p_0 for PHASE_ESTIMATION_MEASUREMENTS = {PHASE_ESTIMATION_MEAUREMENTS}',
            xaxis_title='Error in estimated eigenvalues',
            yaxis_title='Error in estimated p_0',
        )
        phase_error_plot.write_image(f'phase_error_plot_{PHASE_ESTIMATION_MEAUREMENTS}.png')
        phase_error_plot.show()

        # also show both analytical bounds
        # upper_bounds_linear = [upper_bound_linear(og_p0, delta) for delta, og_p0 in zip(delta_values, prob_values)]






    '''
    # Create a pandas dataframe to store the results
    df = pd.DataFrame(columns=['Matrix', 'True Eigenvalue', 'Average Estimation Error', 'Max Estimation Error', 'Variance in MLE estimates of p_0', 'Fisher Information lower Bound'])
    for i, (test_hermitian, estimation_errors) in enumerate(zip(random_hermitians, all_estimation_errors.values())):
        exact_eigenvalue, _ = scipy.linalg.eig(test_hermitian)
        df.loc[i] = [test_hermitian, min(exact_eigenvalue), np.mean(estimation_errors), max(estimation_errors), all_estimation_variances[i][0], all_estimation_variances[i][1]]
    
    # Write df to excel file
    with pd.ExcelWriter('estimation_errors.xlsx') as writer:
        df.to_excel(writer, sheet_name='Estimation Errors', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Estimation Errors']
    '''












