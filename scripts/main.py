from scripts.algo.qdrift import qdrift
from scripts.database import Hubbard, DataManager, Tensor
from scripts.algo.phase_estimation import phase_estimation
import numpy as np

h = Hubbard(2)  # create a Hubbard model of two spatial orbitals
h.decompose(1, 3)  # decompose into one- and two-body terms with coefficients of 1s
hd = h.get_decomp()  # get decomposition info
obt = hd.get_term('obt')  # get the one-body term

data = DataManager('../data/')  # Create DataManager instance
data.save('hubbard', "h_2_1_3", h)  # save the hubbard model

ld = data.load('hubbard', "h_2_1_3")  # load the hubbard model

v: list[Tensor] = qdrift(h, 100)
u = v[-1]
for i in range(len(v) - 2, -1, -1):
    np.multiply(u, v[i])

phase_estimation(4, v)