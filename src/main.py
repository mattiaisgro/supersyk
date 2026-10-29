
"""
Main program for the study of the N=2 supersymmetric SYK model.
"""

import fermions
import supersyk
import observables

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse


"""
Simulation parameters
"""
N = 12
J = 100.0
seed = 42
tol = 1e-12


"""
Construct the operators to analyze, using their name
as key and matrix representation as value.
"""
def operators(psi, psi_bar):

    return {
        f"\psi_1 \psi_2^\dagger": psi[0] @ psi_bar[1],
        f"\psi_1 \psi_2": psi[0] @ psi[1],
        f"\psi_1 \psi_2^\dagger \psi_3 \psi_4^\dagger": psi[0] @ psi_bar[1] @ psi[2] @ psi_bar[3],
    }



if __name__ == "__main__":

    print("Parameters:")
    print("N =", N)
    print("J =", J)
    print("dim(H) =", 2**N)

    print("Generating representation and Hamiltonian...")
    rep = fermions.representation(N)
    Q = supersyk.supercharge(rep, N, J, seed)
    H = supersyk.hamiltonian(Q)

    print("Diagonalizing the Hamiltonian...")
    eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())

    # Construct observables and project them onto the ground subspace
    print("Diagonalizing the projected observable...")
    P = observables.projector(eigenvalues, eigenvectors)
    psi, psi_bar = rep

    # Observables under study
    operators = operators(psi, psi_bar)
    operators_eigs = {}

    for name, O in operators.items():
        O_proj = observables.project(observables.symmetrize(O), P)
        eigs = np.linalg.eigvalsh(O_proj)
        operators_eigs[name] = eigs
        deg = np.sum(np.abs(eigs) < tol)
        print(f"Zero eigenvalues for {name}:", deg, f"({deg / len(eigs):.2%})")

    # Count the number of degenerate ground states
    ground_deg = np.sum(np.abs(eigenvalues) < tol)
    print("Ground state degeneracy: ", ground_deg, f"({ground_deg / len(eigenvalues):.2%})")


    # Plot full spectrum histogram
    fig, ax = plt.subplots()
    ax.hist(eigenvalues, bins=35, alpha=0.75)
    ax.set_xlabel("Energy $E$")
    ax.set_ylabel("Counts")
    ax.set_title(f"Spectrum of the $\mathcal{{N}}=2$ supersymm. SYK model ($N={N}$, $J={J:.2f}$)")

    # Plot histogram of non-zero energies
    fig, ax = plt.subplots()
    ax.hist(
        eigenvalues[np.abs(eigenvalues) > 1e-06].real,
        bins=35, alpha=0.75
    )
    ax.set_xlabel("Energy $E$")
    ax.set_ylabel("Counts")
    ax.set_title(f"Excitations of the $\mathcal{{N}}=2$ supersymm. SYK model ($N={N}$, $J={J:.2f}$)")

    # Plot histogram of the eigenvalues of the projected operator
    for name, O in operators.items():

        eigs = operators_eigs[name]
        fig, ax = plt.subplots()
        ax.hist(
            eigs[np.abs(eigs) > tol].real,
            bins=30, alpha=0.75, label=f"$P {name} P$"
        )
        ax.set_xlabel("Eigenvalues")
        ax.set_ylabel("Counts")
        ax.set_title(f"Projected operator spectrum ($N={N}$, $J={J:.2f}$)")
        ax.legend()


    # Plot level spacing statistics
    for name, O in operators.items():

        eigs = operators_eigs[f"\psi_1 \psi_2^\dagger"].real
        eigs = np.unique(eigs[np.abs(eigs) > tol])

        fig, ax = plt.subplots()
        s = observables.level_spacing(eigs)
        ax.hist(s, density=True, bins=20, alpha=0.75, range=(0, 4))
        ax.set_xlabel("Level spacing $s$")
        ax.set_ylabel("$P(s)$")
        ax.set_title(f"Level spacing statistics for $P {name} P$ ($N={N}$, $J={J:.2f}$)")

    plt.show()
