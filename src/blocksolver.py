
"""
Specialized solver for the block-diagonal Hamiltonian
of the supersymmetric SYK model, which exploits the
R-charge symmetry to reduce the size of the eigensystem.
"""

import fermions
import supersyk
import observables

import numpy as np
import matplotlib.pyplot as plt


def trace(A):

    """
    Compute the normalized trace of a matrix.
    """

    return np.sum(A.diagonal()) / A.shape[0]



class BlockSolution:

    """
    An eigensystem solution for a given R-charge sector,
    making it possible to project operators onto its subspace.
    """

    def __init__(self, eigenvals, eigenvecs, indices):

        self.eigenvals = eigenvals
        self.eigenvecs = eigenvecs
        self.indices = indices

    def project(self, O):

        """
        Project an operator O onto the R-charge sector corresponding to this BlockSolution.
        """

        O_sector = O[self.indices, :][:, self.indices]
        P = self.eigenvecs

        return P.conj().T @ O_sector @ P


class BlockSolver:

    """
    A block-diagonal solver for a given R-charge sector
    of the supersymmetric SYK model.
    """

    def __init__(self, N, J, seed, rep=None):

        """
        Initialize the BlockSolver with the Hamiltonian H and R-charge operator R.
        """

        if rep is None:
            rep = fermions.representation(N)

        Q = supersyk.supercharge(rep, N, J, seed)
        self.H = supersyk.hamiltonian(Q)
        self.R = observables.r_charge(rep[0], rep[1])

        self.sectors = {}


    def H_block(self, R_sector):

        """
        Construct the Hamiltonian block corresponding to the given R-charge sector.
        """

        # Get the unique R-charge values and their corresponding indices,
        # where an index in R_indices points to the R-charge value in R_vals
        R_vals, R_indices = np.unique(self.R.diagonal(), return_inverse=True)

        # Check that the requested R-charge sector is valid
        if R_sector not in R_vals:
            raise ValueError(
                f"R-charge {R_sector} was not found in the R-charge spectrum. "
                f"Available sectors: {R_vals}"
            )
    
        # Get the Hilbert space indices of the selected R-charge sector
        sector_index = np.where(R_vals == R_sector)[0][0]
        indices = np.where(R_indices == sector_index)[0]

        # Construct the Hamiltonian block corresponding to the R-charge sector
        H_block = self.H[indices, :][:, indices]

        return H_block


    def solve(self, R_sector, ground_only=True, tol=1e-12):

        """
        Solve the eigenvalue problem for the Hamiltonian H
        by block-diagonalizing it according to the R-charge symmetry,
        returning the eigenvalues and eigenvectors of the selected R-charge sector.

        The eigensystem is stored inside the solver for later use,
        such as projecting operators onto the R-charge sector.
        The solution for a given R-charge is available as solver.sectors[R_sector].

        If ground_only is True, only the ground state eigenvalue and eigenvector
        are returned, otherwise all eigenvalues and eigenvectors are returned.
        """

        H_block = self.H_block(R_sector)

        R_vals, R_indices = np.unique(self.R.diagonal(), return_inverse=True)
        sector_index = np.where(R_vals == R_sector)[0][0]
        indices = np.where(R_indices == sector_index)[0]

        # Solve the eigenvalue problem for the Hamiltonian block
        eigenvalues, eigenvectors = np.linalg.eigh(H_block.toarray())

        # Return only the ground state eigenvalues and eigenvectors
        if ground_only:
            ground_indices = np.where(np.abs(eigenvalues) < tol)[0]
            eigenvalues = eigenvalues[ground_indices]
            eigenvectors = eigenvectors[:, ground_indices]

        self.sectors[R_sector] = BlockSolution(eigenvalues, eigenvectors, indices)
        return eigenvalues, eigenvectors


if __name__ == "__main__":

    N = 14
    J = 25
    seed = 42

    R_charge = 0.0

    print("Parameters:")
    print(f"N = {N}")
    print(f"J = {J}")
    print(f"seed = {seed}")
    print(f"R-charge = {R_charge}")

    print("Generating fermion representation...")
    rep = fermions.representation(N)

    print("Constructing solver...")
    solver = BlockSolver(N, J, seed, rep)

    # Solve for the spectrum of the Hamiltonian in the R=0 sector
    print("Computing the spectrum of the R-charge sector...")
    spectrum, eigenvectors = solver.solve(R_sector=R_charge, ground_only=True)

    print("Sector size:", solver.sectors[R_charge].eigenvals.size)

    # Project the observable onto the R-charge sector
    O = observables.symmetrize( rep[0][0] @ rep[1][1] )
    O_projected = solver.sectors[R_charge].project(O)

    # Diagonalize the projected operator
    O_spectrum, _ = np.linalg.eigh(O_projected)
    O_levels = observables.level_spacing(O_spectrum)


    # Plots

    th_ground_deg = 3**(N//2 - 1)
    if R_charge == 0.0:
        th_ground_deg *= 2

    ground_deg = np.sum(np.abs(spectrum) < 1e-08)
    print("Ground state degeneracy:", ground_deg)
    print("Theoretical degeneracy:", th_ground_deg)

    fig, ax = plt.subplots()
    ax.hist(spectrum, bins=30, alpha=0.75)
    ax.axhline(th_ground_deg, color='red', linestyle='--', label=f"Th. Ground States")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Counts")
    ax.legend(loc='best')
    ax.set_title(f"Energy Spectrum for R={R_charge} ($N={N}$, $J={J:.2f}$)")

    fig, ax = plt.subplots()
    ax.hist(O_levels, bins=20, alpha=0.75, range=(0, 3.5), density=True)
    ax.set_xlabel("Level spacing")
    ax.set_ylabel("Counts")
    ax.set_title(f"Spectrum of $P \psi_1 \psi_2^\dagger P$ ($N={N}$, $J={J:.2f}$, R={R_charge})")
    plt.show()
