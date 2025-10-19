import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read the CSV file
df = pd.read_csv(r'C:\Users\rodcs\Desktop\my-repos\qDriftExtraPolation\src\parameter_sweep\qdrift_qpe_fc_parameter_sweep_2025-09-29.csv')
df = df[df["time"] > df["time"].min()]  # filter for specific ancilla count if needed
df = df[df["time"] < df["time"].max()]  # filter for specific ancilla count if needed
print("Original dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())

# Group by n_shots
grouped = df.groupby('n_shots')

# Store cleaned data
cleaned_dfs = []

for n_shots_value, group in grouped:
    print(f"\n{'='*60}")
    print(f"Processing group: n_shots = {n_shots_value}")
    print(f"Group size: {len(group)}")
    
    # Make a copy to work with
    group_clean = group.copy()
    
    # Filter out non-positive values for log-log regression
    group_clean = group_clean[
        (group_clean['time'] > 0) & 
        (group_clean['estimation_error'].abs() > 0)
    ].copy()
    
    # Identify duplicated time values
    time_counts = group_clean['time'].value_counts()
    duplicated_times = time_counts[time_counts > 1].index.tolist()
    
    if duplicated_times:
        print(f"Found {len(duplicated_times)} duplicated time values")
        
        # For each duplicated time value, remove duplicates to maximize R²
        for dup_time in duplicated_times:
            # Get all rows with this time value
            dup_rows = group_clean[group_clean['time'] == dup_time]
            n_duplicates = len(dup_rows)
            
            if n_duplicates > 1:
                print(f"\n  Time value {dup_time:.6f} has {n_duplicates} duplicates")
                
                # Try removing each duplicate and calculate R² for remaining data
                best_r2 = -np.inf
                best_idx_to_keep = None
                
                for idx_to_keep in dup_rows.index:
                    # Create temporary dataset removing all duplicates except one
                    temp_df = group_clean[
                        (group_clean['time'] != dup_time) | 
                        (group_clean.index == idx_to_keep)
                    ].copy()
                    
                    # Fit linear regression on log-log scale
                    X = np.log10(temp_df['time'].values).reshape(-1, 1)
                    y = np.log10(temp_df['estimation_error'].abs().values)
                    
                    if len(X) > 1:  # Need at least 2 points for regression
                        model = LinearRegression()
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        r2 = r2_score(y, y_pred)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_idx_to_keep = idx_to_keep
                
                # Remove all duplicates except the one that maximizes R²
                indices_to_remove = dup_rows.index[dup_rows.index != best_idx_to_keep]
                group_clean = group_clean.drop(indices_to_remove)
                print(f"  Kept index {best_idx_to_keep}, removed {len(indices_to_remove)} duplicates")
                print(f"  Best R² after removal: {best_r2:.6f}")
    else:
        print("No duplicated time values found")
    
    # Perform final linear regression on cleaned data (log-log scale)
    X = np.log10(group_clean['time'].values).reshape(-1, 1)
    y = np.log10(group_clean['estimation_error'].abs().values)
    
    if len(X) > 1:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        print(f"\nFinal log-log linear regression results:")
        print(f"  Power law exponent (slope): {model.coef_[0]:.6f}")
        print(f"  Log10(coefficient): {model.intercept_:.6f}")
        print(f"  Coefficient: {10**model.intercept_:.6e}")
        print(f"  R² score: {r2:.6f}")
        print(f"  Final group size: {len(group_clean)}")
        print(f"  Power law: error ≈ {10**model.intercept_:.6e} * time^{model.coef_[0]:.6f}")
    
    cleaned_dfs.append(group_clean)

# Combine all cleaned groups
df_cleaned = pd.concat(cleaned_dfs, ignore_index=True)

print(f"\n{'='*60}")
print(f"Final cleaned dataset shape: {df_cleaned.shape}")
print(f"Rows removed: {len(df) - len(df_cleaned)}")

# Save cleaned data
output_filename = 'qdrift_qpe_cleaned.csv'
df_cleaned.to_csv(output_filename, index=False)
print(f"\nCleaned data saved to: {output_filename}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY BY n_shots:")
print("="*60)
for n_shots_value in sorted(df_cleaned['n_shots'].unique()):
    group = df_cleaned[df_cleaned['n_shots'] == n_shots_value]
    X = group[['time']].values
    y = group['estimation_error'].values
    
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    
    print(f"\nn_shots = {n_shots_value}:")
    print(f"  Number of points: {len(group)}")
    print(f"  R² score: {r2:.6f}")
    print(f"  Regression equation: error = {model.coef_[0]:.6e} * time + {model.intercept_:.6e}")