import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.algo.chebyshev import chebyshev_nodes

if __name__ == "__main__":
    tau_list = 0.0292 + 0.012 * np.array(chebyshev_nodes(10)[:5])
    alpha = 4.4
    t_list = tau_list / alpha

    # Read the CSV
    df = pd.read_csv('qdrift_ising_model_sweep_data_all_2025-06-05.csv')
    num_ancilla = 10
    num_circuits = 1

    df_filtered = df[
        (df['Num Ancilla'] == num_ancilla) &
        (df['Num Random Circuits'] == num_circuits) &
        (df['Time'] >= 0.0292) &
        (df['Time'] <= 0.0411)
        ]
    # Find closest Time for each t in t_list and extract Estimated Eigenvalue
    closest_eigenvalues = []
    actual_times = []
    for t in t_list:
        idx = (df_filtered['Time'] - t).abs().idxmin()
        closest_eigenvalues.append(df_filtered.loc[idx, 'Estimated Eigenvalue'])
        actual_times.append(df_filtered.loc[idx, 'Time'])

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(tau_list, closest_eigenvalues, 'o-', label='Estimated Eigenvalue')
    plt.axhline(3.1240998703626572, color='r', linestyle='--', label='Exact energy')
    plt.xlabel(r'$\tau$')
    plt.ylabel('Estimated Eigenvalue')
    plt.title(rf'Estimated Eigenvalue vs $\tau$ Ancilla: {num_ancilla} #Circuit: {num_circuits}')
    plt.grid(True)
    plt.legend()
    plt.savefig('estimated_eigenvalue_vs_t.png')
    # Optionally, print the actual times used
    for t, actual, eig in zip(tau_list, actual_times, closest_eigenvalues):
        print(f"Requested t: {t:.5f}, Closest Time in CSV: {actual:.5f}, Estimated Eigenvalue: {eig}")


