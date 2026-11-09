
"""
Generate Hamiltonians for the supersymmetric SYK model
and save them to file for later analysis.
"""

import sys

import fermions
import supersyk
import scipy.io


def hamiltonian(N, J, seed, filename=""):

    """
    Pre-generate a Hamiltonian for the supersymmetric SYK model
    with the given parameters and save it to file.
    """

    if filename == "":
        filename = f"hamiltonian_N={N}_J={J:.2f}"

    rep = fermions.representation(N)
    Q = supersyk.supercharge(rep, N, J, seed)
    H = supersyk.hamiltonian(Q)

    scipy.io.mmwrite(filename, H, symmetry="hermitian")



if __name__ == "__main__":

    """
    If the script is run directly, generate and save
    a random matrix for the given parameters,
    passed as command line arguments: N, J, seed.
    """

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    J = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    print("Parameters:")
    print("N =", N)
    print("J =", J)
    print("dim(H) =", 2**N)

    print("Generating the Hamiltonian and saving it to file...")
    hamiltonian(N, J, seed)

    print("Success")
