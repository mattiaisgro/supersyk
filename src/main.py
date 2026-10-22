
"""
Main program of the study on the spectrum of the N=2 supersymmetric SYK model.
"""

import fermions
import supersyk
import observables

import matplotlib.pyplot as plt
import numpy as np


"""
Specify the observable under study.
"""
def observable(psi, psi_bar, P):

    return P @ psi[0] @ psi_bar[1] @ P



if __name__ == "__main__":

    N = 12
    J = 1.0
    seed = 42

    print("Parameters:")
    print("N =", N)
    print("J =", J)
    print("dim(H) =", 2**N)

    print("Generating representation and Hamiltonian...")
    couplings = supersyk.coupling(N, J, seed)
    rep = fermions.representation(N)
    Q = supersyk.supercharge(rep, couplings)
    H = supersyk.hamiltonian(Q)

    print("Diagonalizing the Hamiltonian...")
    eigenvalues_, eigenvectors = np.linalg.eigh(H)
    eigenvalues = np.array([E.real for E in eigenvalues_])


    # Construct observables and project them onto the ground subspace
    print("Diagonalizing the projected observable...")
    P = observables.projector(eigenvalues, eigenvectors)
    psi, psi_bar = rep

    operator = observable(psi, psi_bar, P)
    op_eigenvalues = np.linalg.eigvals(operator)


    print("Energy spectrum:")
    print(eigenvalues)

    # Count the number of degenerate ground states
    ground_deg = 0
    for E in eigenvalues:
        if abs(E) < 1e-12:
            ground_deg += 1

    print("Ground state degeneracy: ", ground_deg)


    # Plot full spectrum histogram
    fig, ax = plt.subplots()
    ax.hist(eigenvalues, bins = 35)
    ax.set_xlabel("Energy $E$")
    ax.set_ylabel("Number of states")
    ax.set_title(f"Spectrum of the $\mathcal{{N}}=2$ supersymm. SYK model ($N={N}$, $J={J:.2f}$)")

    # Plot histogram of non-zero energies
    fig, ax = plt.subplots()
    ax.hist(
        [eigenvalues[i].real for i in range(len(eigenvalues)) if abs(eigenvalues[i]) > 1e-12],
        bins=35
    )
    ax.set_xlabel("Energy $E$")
    ax.set_ylabel("Number of states")
    ax.set_title(f"Excitations of the $\mathcal{{N}}=2$ supersymm. SYK model ($N={N}$, $J={J:.2f}$)")

    # Plot histogram of the eigenvalues of the projected operator
    fig, ax = plt.subplots()
    ax.hist(
        [op_eigenvalues[i].real for i in range(len(op_eigenvalues)) if abs(op_eigenvalues[i]) > 1e-12],
        bins=35
    )
    ax.set_xlabel("Eigenvalues of $P \psi_1 \psi_2^\dagger P$")
    ax.set_ylabel("Number of states")
    ax.set_title(f"Projected operator spectrum ($N={N}$, $J={J:.2f}$)")

    plt.show()
