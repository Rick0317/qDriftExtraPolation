import numpy as np

def general_expr(p0, n):
    return p0 + np.sqrt(p0 * (1 - p0)/n)


def qDrift_theoretical_upper_bound(eig, t, N, total_eval):
    lam = 1
    delta = np.sin(lam * eig * t / N) / (2 * np.sqrt(total_eval))
    p = np.cos(lam * eig * t / (2 * N)) ** 2
    print(f"The value of $\delta$ {delta}")
    print(f"The value of P(0): {p}")
    denominator = 2 * np.sqrt(- N * (p + delta - 1) * (p + delta))
    return delta / denominator


if __name__ == '__main__':
    print(general_expr(0.99991, 10000))
    eig = 0.3389256773827304
    t = 1
    N = 100
    total_eval = 10000

    upperbound = qDrift_theoretical_upper_bound(eig, t, N, total_eval)
    print(upperbound)
