
"""
Main program of the study on the spectrum of the N=2 supersymmetric SYK model.
"""

import fermions
import supersyk

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    N = 12
    J = 1.0
    seed = 42

    print("Parameters:")
    print("N =", N)
    print("J =", J)
    print("dim(H) =", 2**N)

    couplings = supersyk.coupling(N, J, seed)
    rep = fermions.representation(N)
    Q = supersyk.supercharge(rep, couplings)
    H = supersyk.hamiltonian(Q)

    eigenvalues_res = np.linalg.eigvals(H)
    eigenvalues = np.array([E.real for E in eigenvalues_res])

    print("Energy spectrum:")
    print(eigenvalues)

    # Count the number of degenerate ground states
    ground_deg = 0
    for E in eigenvalues:
        if abs(E) < 1e-12:
            ground_deg += 1

    print("Ground state degeneracy: ", ground_deg)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        eigenvalues,
        linewidth=0.6,
        alpha=0.95,
    )

    ax.set_xlabel("Energy $E$")
    ax.set_ylabel("Number of states")
    ax.set_title(f"Spectrum of the $\mathcal{{N}}=2$ supersymmetric SYK model ($N={N}$, $J={J:.2f}$)")

    plt.show()
