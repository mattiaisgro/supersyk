
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
J = 25
seed = 42
tol = 1e-12

plots = [
    "energy_spectrum",
    "operators_spectrum",
    "spacings",
    #"fermion_number"
]


"""
Construct the operators to analyze, using their name
as key and matrix representation as value.
"""
def operators(psi, psi_bar):

    return {
        f"\psi_1 \psi_2^\dagger": observables.symmetrize( psi[0] @ psi_bar[1] ),
        f"\psi_1 \psi_2": observables.symmetrize( psi[0] @ psi[1] ),
        f"\psi_1 \psi_2^\dagger \psi_3 \psi_4^\dagger": observables.symmetrize( psi[0] @ psi_bar[1] @ psi[2] @ psi_bar[3] ),
        f"R-charge": observables.fermion_number(psi, psi_bar)
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


    # Before diagonalizing the Hamiltonian, check if an already
    # computed spectrum has been previously cached.
    filename = f"hamiltonian_N={N}_J={J:.2f}_seed={seed}.npz"
    eigenvalues, eigenvectors = None, None

    try:
        data = np.load(filename)
        eigenvalues = data["eigenvalues"]
        eigenvectors = data["eigenvectors"]
        print(f"Loaded cached Hamiltonian spectrum from {filename}.")

    except FileNotFoundError:
        print(f"Diagonalizing the Hamiltonian...")
        eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
        np.savez(filename, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
        print(f"Saved Hamiltonian spectrum to {filename}.")


    # Construct observables and project them onto the ground subspace
    print("Diagonalizing the projected observable...")
    P = observables.projector(eigenvalues, eigenvectors)
    psi, psi_bar = rep

    # Observables under study
    operators = operators(psi, psi_bar)
    operators_eigs = {}

    for name, O in operators.items():
        O_proj = observables.project(O, P)
        eigs = np.linalg.eigvals(O_proj)
        operators_eigs[name] = eigs
        deg = np.sum(np.abs(eigs) < tol)
        print(f"Zero eigenvalues for {name}:", deg, f"({deg / len(eigs):.2%})")

    # Count the number of degenerate ground states
    ground_deg = np.sum(np.abs(eigenvalues) < tol)
    print("Ground state degeneracy: ", ground_deg, f"({ground_deg / len(eigenvalues):.2%})")


    # Plot full spectrum histogram
    if "energy_spectrum" in plots:

        fig, ax = plt.subplots()
        ax.hist(eigenvalues, bins=35, alpha=0.75)
        ax.set_xlabel("Energy $E$")
        ax.set_ylabel("Counts")
        ax.set_title(f"Spectrum of the $\mathcal{{N}}=2$ supersymm. SYK model ($N={N}$, $J={J:.2f}$)")

        # Plot only non-zero energies
        fig, ax = plt.subplots()
        ax.hist(
            eigenvalues[np.abs(eigenvalues) > 1e-06].real,
            bins=35, alpha=0.75
        )
        ax.set_xlabel("Energy $E$")
        ax.set_ylabel("Counts")
        ax.set_title(f"Excitations of the $\mathcal{{N}}=2$ supersymm. SYK model ($N={N}$, $J={J:.2f}$)")

    # Plot histogram of the non-zero eigenvalues of the projected operator
    if "operators_spectrum" in plots:
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
    if "spacings" in plots:
        for name, O in operators.items():

            eigs = np.abs(operators_eigs[name])
            eigs = eigs[np.abs(eigs) > tol]
            s = observables.level_spacing(eigs)

            fig, ax = plt.subplots()
            ax.hist(s, density=True, bins=20, alpha=0.75, range=(0, 4))
            ax.set_xlabel("Level spacing $s$")
            ax.set_ylabel("$P(s)$")
            ax.set_title(f"Level spacing statistics for $P {name} P$ ($N={N}$, $J={J:.2f}$)")


    # Separate eigenstates with respect to fermion number
    if "fermion_number" in plots:

        number = observables.fermion_number(psi, psi_bar)
        spectrum = {}
        for i in range(eigenvectors.shape[1]):

            vec = eigenvectors[:, i]

            # Rounding is needed to avoid numerical errors,
            # but not all values might be integers!
            n = round(np.abs(vec.conj().T @ number @ vec))

            if n not in spectrum:
                spectrum[n] = []

            spectrum[n].append(eigenvalues[i].real)

        # Plot the spectrum for each fermion number sector
        fig, ax = plt.subplots()
        ax.set_title(f"Spectrum of the $\mathcal{{N}}=2$ supersymm. SYK model ($N={N}$, $J={J:.2f}$)")
        ax.set_xlabel("Energy $E$")
        ax.set_ylabel("Counts")
        for i in sorted(spectrum.keys()):
            eigs = np.array(spectrum[i])
            ax.hist(eigs[np.abs(eigs) > tol].real, bins=30, alpha=0.75, label=f"$n_f={i}$")
            ax.legend()

    plt.show()
