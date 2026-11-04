
"""
Generate Hamiltonians for the supersymmetric SYK model
and save them to file for later analysis.
"""

import fermions
import supersyk
import numpy as np
import scipy.io


"""
Pre-generate a Hamiltonian for the supersymmetric SYK model
with the given parameters and save it to file.
"""
def hamiltonian(N, J, seed, filename=""):

    if filename == "":
        filename = f"hamiltonian_N={N}_J={J:.2f}"

    rep = fermions.representation(N)
    Q = supersyk.supercharge(rep, N, J, seed)
    H = supersyk.hamiltonian(Q)

    scipy.io.mmwrite(filename, H, symmetry="hermitian")


"""
If the script is run directly, generate and save
a random matrix for the given parameters.
"""
if __name__ == "__main__":

    N = 12
    J = 100
    seed = 42

    print("Parameters:")
    print("N =", N)
    print("J =", J)
    print("dim(H) =", 2**N)

    print("Generating the Hamiltonian and saving it to file...")
    hamiltonian(N, J, seed)

    print("Success")
