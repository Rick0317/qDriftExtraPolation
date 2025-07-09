import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def loglog_error_vs_time_with_regression(
    df: pd.DataFrame,
    title_suffix: str,
    true_eigenval: float = None,
    log_transform_data: bool = True,
    show_plot: bool = True,
    agg: bool = False
) -> dict:
    """
    If log_transform_data is True, performs log-log regression per Num Ancilla group
    and plots transformed data + regression lines.
    
    If False, plots raw data using log-log scales (no transformation, no regression).
    Returns R² values per group if regression was performed; otherwise returns None.
    """

    if log_transform_data:
        df = df.copy()
        df["log_time"] = np.log10(df["Time"])
        df["log_error"] = np.log10(df["Eigenvalue Error"]) if not agg else np.log10(df["Mean_Error"])
        x_col = "log_time"
        y_col = "log_error"

        r2_scores = {}
        if show_plot:
            plt.figure(figsize=(8, 6))
        grouped = df.groupby("Num Ancilla")
        for group_name, group_data in grouped:
            X = group_data[[x_col]]
            y = group_data[y_col]

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            r2 = r2_score(y, predictions)
            r2_scores[group_name] = r2

            slope = model.coef_[0]
            intercept = model.intercept_
            if show_plot:
                # Scatter and regression
                plt.scatter(X, y, label=f"Ancilla={group_name}", alpha=0.6)
                x_vals = np.linspace(X.min().values[0], X.max().values[0], 100).reshape(-1, 1)
                x_vals_df = pd.DataFrame(x_vals, columns=[x_col])
                y_vals = model.predict(x_vals_df)
                plt.plot(x_vals, y_vals, linestyle='--')

                # Annotate with R²
                x_mid = X.mean().values[0]
                y_mid = model.predict([[x_mid]])[0]
                plt.text(x_mid, y_mid, f"$R^2$={r2:.2f}", fontsize=8)

        if show_plot:
            plt.xlabel("log(Time)")
            plt.ylabel("log(Error)")
            plt.title(f"Log-Log Regression: {title_suffix}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return r2_scores

    else:
        # Just plot raw data using log-log axes
        grouped = df.groupby("Num Ancilla")
        if show_plot:
            plt.figure(figsize=(8, 6))
            for group_name, group_data in grouped:
                plt.scatter(group_data["Time"], group_data["Eigenvalue Error"] if not agg else group_data["Mean_Error"],
                            label=f"Ancilla={group_name}", alpha=0.6)

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Time")
            plt.ylabel("Eigenvalue Error")
            plt.title(f"Log-Log Plot Only (No Regression): {title_suffix}")
            plt.legend()
            plt.grid(True, which='both', ls='--')
            plt.tight_layout()
            plt.show()

        return None

def search_best_time_interval_r2(
    df: pd.DataFrame,
    min_duration: float,
    title_suffix: str = "",
    log_transform_data: bool = True,
    show_plot: bool = True,
    agg: bool = False
):
    best_r2 = -np.inf
    best_df = None
    best_range = (None, None)

    times = sorted(df["Time"].unique())
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            t_min = times[i]
            t_max = times[j]
            if t_max - t_min < min_duration:
                continue

            sub_df = df[(df["Time"] >= t_min) & (df["Time"] <= t_max)]

            r2s = loglog_error_vs_time_with_regression(
                sub_df,
                title_suffix=f"{title_suffix} [{t_min:.2e}, {t_max:.2e}]",
                log_transform_data=log_transform_data,
                show_plot=show_plot,
                agg=agg
            )

            if r2s is not None:
                r2_of_largest_qubits = r2s.get(sub_df["Num Ancilla"].max(), -np.inf)
                if r2_of_largest_qubits > best_r2:
                    best_r2 = r2_of_largest_qubits
                    best_df = sub_df
                    best_range = (t_min, t_max)

    print(f"\n✅ Best R² sum = {best_r2:.4f} in interval {best_range}")
    return best_df, best_range, best_r2



