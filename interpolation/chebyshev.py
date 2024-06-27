import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def chebyshev_nodes(n: int):
    """Returns a list of n chebyshev nodes on the open interval (-1,1)

    :param n (int): a number of chebyshev nodes
    :return: a list of n chebyshev nodes
    """
    nodes = [np.cos((2 * k + 1)/(2 * n) * np.pi) for k in range(n)]
    return nodes


def chebyshev_barycentric_weight(j: int, n: int):
    """Returns the value of j-th barycentric weight for the n Chebyshev points of the first kind
    
    :precondition: j = 0, ..., n-1
    
    :param j (int): j-th barycentric weight
    :param n (int): a number of Chebyshev nodes
    :return: the j-th barycentric weight for the Chebyshev points of the first kind
    """
    assert 0 <= j and j <= n-1, "j out of range"
    return (-1) ** j * np.sin((2 * j + 1) * np.pi / (2 * n))


def chebyshev_barycentric_interp(x: float, n: int, f: callable):
    """Returns the value of barycentric interpolation at x using n Chebyshev nodes

    :param x (float): a point where the terms are evaluated
    :param n (int): a number of the Chebyshev nodes
    :param f (callable): a function that is to be interpolated
    """
    nodes = chebyshev_nodes(n)
    terms = [chebyshev_barycentric_weight(j, n)/(x - nodes[j]) for j in range(n)]
    denom = np.sum(terms)
    for i in range(len(terms)):
        terms[i] = terms[i] * f(nodes[i])
    numer = np.sum(terms)
    return numer/denom


if __name__ == '__main__':
    n = 11
    domain = np.linspace(-1, 1, 200)
    f = lambda x: 1 / (1 + 25 * x ** 2)

    plt.plot(domain,f(domain))
    plt.plot(domain, [chebyshev_barycentric_interp(x, n, f) for x in domain])
