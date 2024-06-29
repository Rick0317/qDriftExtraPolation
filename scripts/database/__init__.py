from scripts.database.database import *

if __name__ == "__main__":
    h = Hubbard(2)  # create a Hubbard model of two spatial orbitals
    h.decompose(1, 1)  # decompose into one- and two-body terms with coefficients of 1s
    hd = h.get_decomp()  # get decomposition info
    obt = hd.get_term('obt')  # get the one-body term
    print(obt.matrix)

    data = DataManager()  # Create DataManager instance
    data.save('hubbard', "h_2", h)  # save the hubbard model

    ld = data.load('hubbard', "h_2")  # load the hubbard model
    print(ld.get_decomp().get_term('obt').matrix)
